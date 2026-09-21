"""From-scratch posterior-TVM A/B training for Task1.

This module is deliberately independent from the older fixed-teacher TVM
experiments.  It owns the matched global random plan, local clean-token CE,
EMA-terminal posterior TVM map loss, empirical information clock, gradient
calibration, and strict finite-map sampler required by the dual-card contract.
"""

import copy
import json
import math
import os
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from algo import LangFlowFLMHybrid
from langflow_hybrid.ops import flm_vocab_gaussian_bias
from task1_tvm_ce import (
    calibrated_relative_distributions,
    finite_map_update,
    integer_quota_counts,
    tvm_quantity_from_logits_jvp,
)
from trainer_base import Loss


_CATEGORY_NAMES = ('short', 'medium', 'long', 'grid', 'full_span')
_CURRICULUM = (
    (0, 1000, (0.70, 0.20, 0.10, 0.00, 0.00)),
    (1000, 5000, (0.25, 0.30, 0.25, 0.10, 0.10)),
    (5000, 10000, (0.15, 0.25, 0.30, 0.20, 0.10)),
    (10000, 30000, (0.10, 0.20, 0.30, 0.25, 0.15)),
    (30000, 50001, (0.10, 0.15, 0.25, 0.30, 0.20)),
)
_GRID_B = (0.0, 0.339277, 0.581469, 0.724669, 0.95)
_GRID_U = (0.0, 0.2375, 0.475, 0.7125, 0.95)


def posterior_gamma(t, eps=1e-4):
    """Contract log-NSR coordinate; used as a feature, never as JVP input."""
    t = t.float()
    return 2.0 * torch.log((1.0 - t + eps) / (t + eps))


def posterior_time_features(r, eta, eps=1e-4):
    """Return [r, gamma(r)/8, eta, s-r, delta_gamma/8]."""
    r = r.float()
    eta = eta.float()
    s = r + (1.0 - r) * eta
    gamma_r = posterior_gamma(r, eps)
    gamma_s = posterior_gamma(s, eps)
    return torch.stack((
        r, gamma_r / 8.0, eta, s - r, (gamma_s - gamma_r) / 8.0),
        dim=-1)


def _isotonic_nonincreasing(values):
    """Small weighted PAVA implementation for a 1-D decreasing fit."""
    blocks = []
    for value in np.asarray(values, dtype=np.float64):
        blocks.append([float(value), 1, 1])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            blocks.append([
                (left[0] * left[1] + right[0] * right[1]) / weight,
                weight,
                left[2] + right[2],
            ])
    fitted = []
    for value, _, count in blocks:
        fitted.extend([value] * count)
    return np.asarray(fitted, dtype=np.float64)


class Task1PosteriorTVM(LangFlowFLMHybrid):
    """Matched A=no-SC / B=finite-map-only-SC posterior TVM model."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.variant = str(config.algo.posterior_tvm_variant)
        self.local_only = bool(getattr(
            config.algo, 'posterior_tvm_local_only', False))
        self.fixed_cdf_after_5k_path = getattr(
            config.algo, 'posterior_tvm_fixed_cdf_after_5k_path', None)
        self.terminal_t = float(config.algo.posterior_tvm_terminal_t)
        self.kappa = float(config.algo.posterior_tvm_kappa)
        self.map_subset_size = int(config.algo.posterior_tvm_map_subset_size)
        self.plan_seed = int(config.algo.posterior_tvm_plan_seed)
        self.target_decay_max = float(
            config.algo.posterior_tvm_target_ema_decay)
        self.metrics_backend = str(getattr(
            config.algo, 'posterior_tvm_metrics_backend', 'reference'))
        self.target_query_mode = str(getattr(
            config.algo, 'posterior_tvm_target_query_mode', 'no_grad'))
        self.eval_decay_max = float(config.training.ema)
        self.cdf_bank_size = int(config.algo.posterior_tvm_cdf_bank_size)
        self.cdf_node_count = int(config.algo.posterior_tvm_cdf_nodes)
        self.cdf_query_batch = int(config.algo.posterior_tvm_cdf_query_batch)
        self.cdf_denominator_min = float(
            config.algo.posterior_tvm_cdf_denominator_min)
        self.cdf_bootstrap_spread_max = float(
            config.algo.posterior_tvm_cdf_bootstrap_spread_max)
        self._validate_contract()

        self.training_target_ema = copy.deepcopy(self.backbone)
        self.training_target_ema.requires_grad_(False)
        self.training_target_ema.eval()
        self.register_buffer(
            'training_target_ema_updates',
            torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            'map_lambda_star', torch.ones((), dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'gradient_probe_ratios',
            torch.full((3,), float('nan'), dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'gradient_probe_count',
            torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            'selected_grid_index',
            torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            'sampler_cdf_version',
            torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            'sc_amplitude', torch.ones((), dtype=torch.float32),
            persistent=True)
        nodes = torch.linspace(0.0, self.terminal_t, 4097)
        with torch.no_grad():
            tau = self._t_to_tau(nodes).float()
            tau = (tau - tau[0]) / (tau[-1] - tau[0]).clamp_min(1e-8)
            cdf = 0.5 * nodes / self.terminal_t + 0.5 * tau
            cdf[0], cdf[-1] = 0.0, 1.0
        self.register_buffer('sampler_t_nodes', nodes, persistent=True)
        self.register_buffer('sampler_cdf_values', cdf, persistent=True)
        self._cdf_clean_tokens_cpu = None
        self._last_plan_audit = None
        self._posterior_validation_batch_index = 0
        self._validation_online_target_kl = []

    def _validate_contract(self):
        if self.variant not in {'A_no_sc', 'B_finite_sc'}:
            raise ValueError('posterior_tvm_variant must be A_no_sc or B_finite_sc.')
        finite_sc = bool(getattr(self.config.algo, 'finite_map_only_sc', False))
        if finite_sc != (self.variant == 'B_finite_sc'):
            raise ValueError('finite_map_only_sc must be enabled only for B.')
        if not bool(getattr(
                self.config.algo, 'posterior_tvm_time_conditioning', False)):
            raise ValueError('The dedicated posterior-TVM conditioner is required.')
        if not bool(getattr(self.config.model, 'qk_norm', False)):
            raise ValueError('QK RMSNorm must be enabled.')
        if not math.isclose(float(self.config.model.dropout), 0.0):
            raise ValueError('Posterior-TVM requires dropout=0.')
        if not math.isclose(self.terminal_t, 0.95):
            raise ValueError('Posterior-TVM terminal time must remain T=0.95.')
        if int(self.config.loader.global_batch_size) != 256:
            raise ValueError('Posterior-TVM scientific global batch must be 256.')
        expected_map_subset = 0 if self.local_only else 96
        if self.map_subset_size != expected_map_subset:
            raise ValueError(
                'Posterior-TVM map subset must be 0 for local-only and 96 '
                'otherwise.')
        if self.local_only:
            if self.variant != 'A_no_sc' or finite_sc:
                raise ValueError(
                    'Local-only control must use A_no_sc with no finite SC.')
            if not self.fixed_cdf_after_5k_path:
                raise ValueError(
                    'Local-only control requires the exact A@5k CDF artifact.')
        if not math.isclose(self.kappa, 0.02):
            raise ValueError('Posterior-TVM calibrated cap kappa must be 0.02.')
        if not math.isclose(self.target_decay_max, 0.99):
            raise ValueError('Training-target EMA must start at beta=0.99.')
        if self.metrics_backend not in {'reference', 'detached'}:
            raise ValueError(
                'posterior_tvm_metrics_backend must be reference or detached.')
        if self.target_query_mode not in {'no_grad', 'inference_clone'}:
            raise ValueError(
                'posterior_tvm_target_query_mode must be no_grad or '
                'inference_clone.')
        if not math.isclose(self.eval_decay_max, 0.999):
            raise ValueError('Evaluation EMA must remain beta=0.999.')

    @property
    def optimization_scale(self):
        return 1.0

    def task1_training_time_contract(self):
        return {
            'workflow': (
                'TASK1-MATCHED-LOCAL-ONLY-CONTROL'
                if self.local_only
                else 'TASK1-FROMSCRATCH-POSTERIOR-TVM-DUAL'),
            'variant': self.variant,
            'physical_path': 'x_t=t*one_hot(y)+(1-t)*epsilon',
            'terminal_t': self.terminal_t,
            'network_features': [
                'r', 'gamma_eps(r)/8', 'eta', 's-r',
                '(gamma_eps(s)-gamma_eps(r))/8'],
            'jvp_variable': 'eta=(s-r)/(1-r)',
            'local_sequences_per_step': 256,
            'map_sequences_per_step': self.map_subset_size,
            'training_target_ema': self.target_decay_max,
            'evaluation_ema': self.eval_decay_max,
            'strict_nfe': True,
        }

    def setup(self, stage):
        del stage
        self.training_target_ema.requires_grad_(False)
        self.training_target_ema.eval()
        self._apply_runtime_overrides()

    def _apply_runtime_overrides(self):
        override = getattr(
            self.config.algo, 'posterior_tvm_target_ema_decay_override', None)
        if override is not None:
            value = float(override)
            if value not in {0.98, 0.99}:
                raise ValueError('Target EMA override may only be 0.98 or 0.99.')
            self.target_decay_max = value
        grid_override = getattr(
            self.config.algo, 'posterior_tvm_selected_grid_override', None)
        if grid_override is not None:
            value = int(grid_override)
            if value not in {0, 1, 2}:
                raise ValueError('Selected grid override must be 0, 1, or 2.')
            self.selected_grid_index.fill_(value)
        sc_override = getattr(
            self.config.algo, 'posterior_tvm_sc_amplitude_override', None)
        if sc_override is not None:
            value = float(sc_override)
            if value not in {0.5, 1.0}:
                raise ValueError('SC amplitude override may only be 0.5 or 1.0.')
            self.sc_amplitude.fill_(value)
        lambda_override = getattr(
            self.config.algo, 'posterior_tvm_lambda_star_override', None)
        if lambda_override is not None:
            value = float(lambda_override)
            if not 0.1 <= value <= 10.0:
                raise ValueError('lambda-star override must lie in [0.1, 10].')
            self.map_lambda_star.fill_(value)

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._apply_runtime_overrides()
        return result

    def on_load_checkpoint(self, checkpoint):
        self.target_decay_max = float(checkpoint.get(
            'posterior_tvm_target_ema_decay', self.target_decay_max))
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['posterior_tvm_target_ema_decay'] = float(
            self.target_decay_max)
        checkpoint['posterior_tvm_random_stream_contract'] = {
            'base_seed': self.plan_seed,
            'stateless_by_optimizer_step_and_accumulation_slice': True,
            'streams': [
                'plan', 'local_noise', 'map_noise',
                'predecessor_noise', 'cdf'],
        }
        checkpoint['posterior_tvm_scientific_state'] = {
            'variant': self.variant,
            'selected_grid_index': int(self.selected_grid_index),
            'sampler_cdf_version': int(self.sampler_cdf_version),
            'map_lambda_star': float(self.map_lambda_star),
            'sc_amplitude': float(self.sc_amplitude),
        }

    def on_train_start(self):
        # B's extra zero-init module must not perturb the paired data sampler.
        torch.manual_seed(int(self.config.seed) + 20260920)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.seed) + 20260920)
        super().on_train_start()
        self.training_target_ema.eval()
        if not self.local_only:
            self._prepare_cdf_bank()

    def _prepare_cdf_bank(self):
        if self._cdf_clean_tokens_cpu is not None:
            return
        # limit_val_batches=0 is intentional during the stability benchmark,
        # so Lightning does not expose trainer.val_dataloaders at this hook.
        # Build the fixed held-out bank from the same manifest-declared packed
        # validation split instead of depending on trainer hook ordering.
        from packed_dataset import PackedTokenDataset
        dataset = PackedTokenDataset(self.config.data.packed_dir, 'validation')
        tokens = []
        for index in range(self.cdf_bank_size):
            item = dataset[index]
            value = item['input_ids'] if isinstance(item, dict) else item
            tokens.append(torch.as_tensor(value, dtype=torch.long))
        self._cdf_clean_tokens_cpu = torch.stack(tokens, dim=0).cpu()

    def _seed(self, stream, step, accumulation_step=0):
        stream_ids = {
            'plan': 101, 'local_noise': 211, 'map_noise': 307,
            'predecessor_noise': 401, 'cdf': 503,
        }
        return int(
            self.plan_seed
            + int(step) * 1000003
            + int(accumulation_step) * 10007
            + stream_ids[stream])

    def _generator(self, stream, step, accumulation_step=0):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self._seed(stream, step, accumulation_step))
        return generator

    def _inverse_cdf(self, uniform):
        values = self.sampler_cdf_values
        nodes = self.sampler_t_nodes
        uniform = uniform.float().clamp(0.0, 1.0)
        upper = torch.searchsorted(values, uniform, right=False).clamp(
            1, values.numel() - 1)
        lower = upper - 1
        c0, c1 = values[lower], values[upper]
        t0, t1 = nodes[lower], nodes[upper]
        weight = (uniform - c0) / (c1 - c0).clamp_min(1e-8)
        return t0 + weight * (t1 - t0)

    def _information_grid(self):
        quantiles = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            device=self.device, dtype=torch.float32)
        return self._inverse_cdf(quantiles)

    def candidate_grids(self):
        return (
            torch.tensor(_GRID_B, device=self.device),
            torch.tensor(_GRID_U, device=self.device),
            self._information_grid(),
        )

    def _curriculum_probabilities(self, step):
        override_name = (
            'posterior_tvm_curriculum_10_30_override'
            if 10000 <= int(step) < 30000
            else 'posterior_tvm_curriculum_30_50_override'
            if int(step) >= 30000 else None)
        if override_name is not None:
            override = getattr(self.config.algo, override_name, None)
            if override is not None:
                values = tuple(float(value) for value in override)
                if (len(values) != 5
                        or any(value < 0.0 for value in values)
                        or not math.isclose(sum(values), 1.0, abs_tol=1e-8)):
                    raise ValueError(
                        f'{override_name} must be five probabilities summing to 1.')
                return values
        for start, stop, probabilities in _CURRICULUM:
            if start <= int(step) < stop:
                return probabilities
        return _CURRICULUM[-1][2]

    def _global_plan(self, accumulation_step):
        step = int(self.global_step)
        global_size = int(self.config.loader.global_batch_size)
        micro_size = int(self.config.loader.batch_size)
        accumulation = int(self.trainer.accumulate_grad_batches)
        if global_size != micro_size * accumulation:
            raise ValueError('Single-card global batch must equal microbatch*accumulation.')
        generator = self._generator('plan', step)
        local_t = self._inverse_cdf(torch.rand(
            global_size, device=self.device, generator=generator))

        if self.local_only:
            start = int(accumulation_step or 0) * micro_size
            stop = start + micro_size
            return {
                'local_t': local_t[start:stop],
                'map_mask': torch.zeros(
                    micro_size, dtype=torch.bool, device=self.device),
            }

        permutation = torch.randperm(
            global_size, device=self.device, generator=generator)
        subset = permutation[:self.map_subset_size]
        map_mask = torch.zeros(global_size, dtype=torch.bool, device=self.device)
        map_mask[subset] = True
        probabilities = self._curriculum_probabilities(step)
        counts = integer_quota_counts(probabilities, self.map_subset_size)
        category_selected = torch.repeat_interleave(
            torch.arange(5, device=self.device),
            torch.tensor(counts, device=self.device))
        category_selected = category_selected[torch.randperm(
            self.map_subset_size, device=self.device, generator=generator)]
        category = torch.full(
            (global_size,), -1, dtype=torch.long, device=self.device)
        category[subset] = category_selected

        r = torch.zeros(global_size, device=self.device)
        s = torch.zeros(global_size, device=self.device)
        broad = map_mask & (category <= 2)
        broad_count = int(broad.sum())
        if broad_count:
            r_zero = torch.rand(
                broad_count, device=self.device, generator=generator) < 0.35
            broad_r = self._inverse_cdf(torch.rand(
                broad_count, device=self.device, generator=generator))
            broad_r[r_zero] = 0.0
            unit = torch.rand(
                broad_count, device=self.device, generator=generator)
            broad_category = category[broad]
            z = torch.empty_like(unit)
            z[broad_category == 0] = unit[broad_category == 0] * 0.2
            z[broad_category == 1] = 0.2 + unit[broad_category == 1] * 0.4
            z[broad_category == 2] = 0.6 + unit[broad_category == 2] * 0.4
            r[broad] = broad_r
            s[broad] = broad_r + z * (self.terminal_t - broad_r)

        full = map_mask & (category == 4)
        r[full] = 0.0
        s[full] = self.terminal_t

        grid_indices = (map_mask & (category == 3)).nonzero(
            as_tuple=True)[0].tolist()
        grids = self.candidate_grids()
        selected_main = int(self.selected_grid_index)
        other = [index for index in range(3) if index != selected_main]
        for global_index in grid_indices:
            draw = float(torch.rand((), device=self.device, generator=generator))
            grid_index = selected_main if draw < 0.6 else (
                other[0] if draw < 0.8 else other[1])
            grid = grids[grid_index].clone()
            if bool(torch.rand((), device=self.device, generator=generator) < 0.5):
                jitter = (
                    torch.rand(3, device=self.device, generator=generator)
                    * 0.04 - 0.02)
                grid[1:4] = grid[1:4] + jitter
            interval = int(torch.randint(
                0, 4, (), device=self.device, generator=generator))
            r[global_index], s[global_index] = grid[interval], grid[interval + 1]

        generated = torch.zeros(
            global_size, dtype=torch.bool, device=self.device)
        positive_r = map_mask & (r > 0.0)
        generated[positive_r] = torch.rand(
            int(positive_r.sum()), device=self.device,
            generator=generator) < 0.5
        predecessor_zero = torch.zeros_like(generated)
        predecessor_zero[generated] = torch.rand(
            int(generated.sum()), device=self.device,
            generator=generator) < 0.5
        a = torch.zeros_like(r)
        nonzero_a = generated & ~predecessor_zero
        a[nonzero_a] = torch.rand(
            int(nonzero_a.sum()), device=self.device,
            generator=generator) * r[nonzero_a]

        start = int(accumulation_step or 0) * micro_size
        stop = start + micro_size
        plan = {
            'local_t': local_t[start:stop],
            'map_mask': map_mask[start:stop],
            'category': category[start:stop],
            'r': r[start:stop],
            's': s[start:stop],
            'generated': generated[start:stop],
            'a': a[start:stop],
            'probabilities': probabilities,
            'counts': counts,
        }
        if int(accumulation_step or 0) == 0:
            self._last_plan_audit = {
                'step': step,
                'local_t': local_t.detach().cpu(),
                'subset': subset.detach().cpu(),
                'category': category.detach().cpu(),
                'r': r.detach().cpu(),
                's': s.detach().cpu(),
            }
        return plan

    def _noise(self, shape, stream, accumulation_step):
        return torch.randn(
            shape, device=self.device, dtype=torch.float32,
            generator=self._generator(
                stream, int(self.global_step), int(accumulation_step or 0)))

    def _logits(self, model, state, r, eta, cache=None,
                previous_eta=None, valid=None, use_jvp_attn=False):
        features = posterior_time_features(r, eta)
        state_embedding = model.vocab_embed(state)
        residual = model(
            state_embedding,
            r.float(),
            use_jvp_attn=use_jvp_attn,
            inputs_are_embeddings=True,
            posterior_time_features=features,
            finite_sc_cache=cache,
            finite_sc_previous_eta=previous_eta,
            finite_sc_valid=valid,
            finite_sc_eta=eta,
            finite_sc_amplitude=float(self.sc_amplitude))
        bias = flm_vocab_gaussian_bias(
            state, r, self._current_token_bias_weight(), self.flm_time_eps)
        return residual.float() + bias.float()

    def _target_probabilities(self, state, r, eta, cache=None,
                              previous_eta=None, valid=None):
        self.training_target_ema.eval()
        context = (
            torch.inference_mode()
            if self.target_query_mode == 'inference_clone'
            else torch.no_grad())
        with context:
            logits = self._logits(
                self.training_target_ema, state, r, eta,
                cache=cache, previous_eta=previous_eta, valid=valid)
            probabilities = F.softmax(logits.float(), dim=-1)
        if self.target_query_mode == 'inference_clone':
            # An inference tensor cannot be saved by the student's backward.
            # Clone after leaving inference_mode to return a normal tensor.
            probabilities = probabilities.clone()
        return probabilities

    def _map_state(self, tokens, plan, accumulation_step):
        r = plan['r']
        selected = plan['map_mask']
        tokens = tokens[selected]
        r = r[selected]
        a = plan['a'][selected]
        generated = plan['generated'][selected]
        shape = (tokens.shape[0], tokens.shape[1], self.vocab_size)
        map_noise = self._noise(shape, 'map_noise', accumulation_step)
        state_r = self.corrupt_vocab_state(tokens, r, noise=map_noise)
        hidden = int(self.config.model.hidden_size)
        cache = torch.zeros(
            tokens.shape[0], tokens.shape[1], hidden,
            device=self.device, dtype=torch.float32)
        previous_eta = torch.zeros_like(r)
        valid = torch.zeros_like(r)
        if bool(generated.any()):
            index = generated.nonzero(as_tuple=True)[0]
            predecessor_noise = self._noise(
                (index.numel(), tokens.shape[1], self.vocab_size),
                'predecessor_noise', accumulation_step)
            state_a = self.corrupt_vocab_state(
                tokens[index], a[index], noise=predecessor_noise)
            eta_previous = (
                (r[index] - a[index])
                / (1.0 - a[index]).clamp_min(1e-8))
            previous_probability = self._target_probabilities(
                state_a, a[index], eta_previous)
            state_r[index] = finite_map_update(
                state_a, previous_probability, a[index], r[index]).detach()
            if self.variant == 'B_finite_sc':
                cache[index] = self.training_target_ema.vocab_embed(
                    previous_probability).detach().float()
                previous_eta[index] = eta_previous
                valid[index] = 1.0
        return tokens, state_r, r, cache, previous_eta, valid

    def _map_objective(self, clean_tokens, plan, accumulation_step):
        if not bool(plan['map_mask'].any()):
            return clean_tokens.new_zeros((), dtype=torch.float32), None
        (map_tokens, state, r, cache,
         previous_eta, valid) = self._map_state(
            clean_tokens, plan, accumulation_step)
        s = plan['s'][plan['map_mask']]
        category = plan['category'][plan['map_mask']]
        eta = ((s - r) / (1.0 - r).clamp_min(1e-8)).clamp(0.0, 1.0)

        def logits_at_eta(eta_value):
            return self._logits(
                self.backbone, state, r, eta_value,
                cache=cache, previous_eta=previous_eta, valid=valid,
                use_jvp_attn=True)

        logits, tangent = torch.func.jvp(
            logits_at_eta, (eta,), (eta * (1.0 - eta),))
        probability, _, gate, quantity = tvm_quantity_from_logits_jvp(
            logits, tangent)
        endpoint = finite_map_update(state, probability, r, s)
        zeros_eta = torch.zeros_like(s)
        terminal = self._target_probabilities(
            endpoint.detach(), s, zeros_eta).detach()
        student_log, calibrated_target = calibrated_relative_distributions(
            probability, gate, terminal, self.kappa)
        calibrated_ce = -(calibrated_target * student_log).sum(dim=-1)
        negative_mass = F.relu(-quantity).sum(dim=-1)
        token_loss = calibrated_ce + 0.1 * negative_mass.square()
        metrics_context = (
            torch.no_grad()
            if self.metrics_backend == 'detached'
            else nullcontext())
        with metrics_context:
            raw_residual = quantity - terminal
            raw_l1 = raw_residual.abs().sum(dim=-1)
            raw_rms = raw_residual.square().mean(dim=-1).sqrt()
            velocity_l1 = raw_l1 / (1.0 - s).clamp_min(1e-8)[:, None]
            velocity_rms = raw_rms / (1.0 - s).clamp_min(1e-8)[:, None]
            calibrated_kl = (
                calibrated_target
                * (calibrated_target.clamp_min(
                    torch.finfo(torch.float32).tiny).log()
                   - student_log)).sum(dim=-1)
            cap_l1 = (student_log.exp() - quantity).abs().sum(dim=-1)
            target_entropy = -(
                terminal * terminal.clamp_min(
                    torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
            student_entropy = -(
                probability * probability.clamp_min(
                    torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
        metrics = {
            'calibrated_ce': calibrated_ce, 'calibrated_kl': calibrated_kl,
            'raw_l1': raw_l1, 'raw_rms': raw_rms,
            'velocity_l1': velocity_l1, 'velocity_rms': velocity_rms,
            'negative_mass': negative_mass, 'cap_l1': cap_l1,
            'target_entropy': target_entropy,
            'student_entropy': student_entropy,
        }
        for name, value in metrics.items():
            self.log(
                f'posterior_tvm/map/{name}_mean', value.detach().mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                f'posterior_tvm/map/{name}_p95',
                torch.quantile(value.detach().reshape(-1), 0.95),
                on_step=True, on_epoch=False, sync_dist=True)
        for category_index, category_name in enumerate(_CATEGORY_NAMES):
            selected = category == category_index
            if bool(selected.any()):
                self.log(
                    f'posterior_tvm/map/{category_name}/calibrated_kl',
                    calibrated_kl[selected].detach().mean(),
                    on_step=True, on_epoch=False, sync_dist=True)
                self.log(
                    f'posterior_tvm/map/{category_name}/raw_l1',
                    raw_l1[selected].detach().mean(),
                    on_step=True, on_epoch=False, sync_dist=True)
        map_mean = token_loss.mean()
        return map_mean, {
            'probability': probability.detach(),
            'state': state.detach(), 'r': r.detach(), 's': s.detach(),
            'cache': cache.detach(), 'previous_eta': previous_eta.detach(),
            'valid': valid.detach(), 'category': category.detach(),
        }

    def _lambda_map(self):
        step = int(self.global_step)
        warm = 0.05 + 0.95 * min(step / 2000.0, 1.0)
        star = float(self.map_lambda_star)
        if step >= 2000 and (star > 2.0 or star < 0.5) and step < 2100:
            fraction = (step - 2000) / 100.0
            star = 1.0 + fraction * (star - 1.0)
        return warm * star

    def _shared_parameters(self):
        return tuple(
            parameter for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
            and not name.startswith('finite_map_sc_projector.'))

    def _gradient_probe(self, local_loss, map_loss, completed_step):
        parameters = self._shared_parameters()
        gradients = {}
        for name, scalar in (('local', local_loss), ('map', map_loss)):
            raw = torch.autograd.grad(
                scalar, parameters, retain_graph=True, allow_unused=True)
            gradients[name] = tuple(
                None if value is None else value.detach().float().cpu()
                for value in raw)
            del raw
        norms = {}
        for name, values in gradients.items():
            squares = [value.square().sum() for value in values
                       if value is not None]
            norms[name] = torch.stack(squares).sum().sqrt() if squares else torch.zeros(())
        dot = torch.zeros(())
        for left, right in zip(gradients['local'], gradients['map']):
            if left is not None and right is not None:
                dot += (left * right).sum()
        cosine = dot / (norms['local'] * norms['map']).clamp_min(1e-20)
        ratio = norms['local'] / norms['map'].clamp_min(1e-20)
        index = (500, 1000, 2000).index(completed_step)
        self.gradient_probe_ratios[index] = ratio.to(self.device)
        self.gradient_probe_count.fill_(index + 1)
        if completed_step == 2000:
            median = torch.nanmedian(self.gradient_probe_ratios).item()
            self.map_lambda_star.fill_(min(max(0.5 * median, 0.1), 10.0))
        for name, value in (
                ('local_norm', norms['local']), ('map_norm', norms['map']),
                ('local_map_cosine', cosine), ('local_map_ratio', ratio)):
            self.log(
                f'posterior_tvm/gradient/{name}', value.to(self.device),
                on_step=True, on_epoch=False, sync_dist=False)

    def _loss(self, clean_tokens, valid_tokens,
              current_accumulation_step=None, train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        del xT, given_t, not_sampling_t
        accumulation_step = int(current_accumulation_step or 0)
        if not train_mode:
            batch_size = clean_tokens.shape[0]
            node_count = 16
            offset = self._posterior_validation_batch_index * batch_size
            node_index = (
                torch.arange(batch_size, device=self.device) + offset
            ) % node_count
            t = (node_index.float() + 0.5) / node_count * self.terminal_t
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                self.plan_seed + 900000
                + self._posterior_validation_batch_index)
            noise = torch.randn(
                (batch_size, clean_tokens.shape[1], self.vocab_size),
                device=self.device, dtype=torch.float32, generator=generator)
            state = self.corrupt_vocab_state(clean_tokens, t, noise=noise)
            logits = self._logits(
                self.backbone, state, t, torch.zeros_like(t))
            evaluation_probability = F.softmax(logits.float(), dim=-1)
            token_loss = F.cross_entropy(
                logits.transpose(1, 2), clean_tokens, reduction='none')
            mask = valid_tokens.float()
            nlls = (token_loss * mask).sum()
            count = mask.sum()
            target_probability = self._target_probabilities(
                state, t, torch.zeros_like(t))
            tiny = torch.finfo(torch.float32).tiny
            target_ce = -target_probability.clamp_min(tiny).log().gather(
                -1, clean_tokens.unsqueeze(-1)).squeeze(-1)
            target_entropy = -(
                target_probability
                * target_probability.clamp_min(tiny).log()).sum(dim=-1)
            target_top1 = (
                target_probability.argmax(dim=-1) == clean_tokens).float()
            eval_target_kl = (
                evaluation_probability
                * (evaluation_probability.clamp_min(tiny).log()
                   - target_probability.clamp_min(tiny).log())).sum(dim=-1)

            online_probability = None
            if self.ema is not None and self.ema.collected_params:
                self.ema.restore(self._get_parameters())
                online_logits = self._logits(
                    self.backbone, state, t, torch.zeros_like(t))
                online_probability = F.softmax(online_logits.float(), dim=-1)
                self.ema.copy_to(self._get_parameters())
            if online_probability is None:
                online_probability = evaluation_probability
            online_ce = -online_probability.clamp_min(tiny).log().gather(
                -1, clean_tokens.unsqueeze(-1)).squeeze(-1)
            online_target_kl = (
                online_probability
                * (online_probability.clamp_min(tiny).log()
                   - target_probability.clamp_min(tiny).log())).sum(dim=-1)
            valid = valid_tokens.bool()
            self._validation_online_target_kl.append(
                online_target_kl[valid].detach().float().cpu())
            batch_log = {'on_step': False, 'on_epoch': True,
                         'sync_dist': True, 'batch_size': batch_size}
            self.log(
                'posterior_tvm/heldout/eval_ema_local_ce',
                nlls / count.clamp_min(1.0), **batch_log)
            self.log(
                'posterior_tvm/heldout/online_local_ce',
                (online_ce * mask).sum() / count.clamp_min(1.0), **batch_log)
            self.log(
                'posterior_tvm/heldout/target_local_ce',
                (target_ce * mask).sum() / count.clamp_min(1.0), **batch_log)
            self.log(
                'posterior_tvm/heldout/target_entropy',
                (target_entropy * mask).sum() / count.clamp_min(1.0),
                **batch_log)
            self.log(
                'posterior_tvm/heldout/target_top1',
                (target_top1 * mask).sum() / count.clamp_min(1.0),
                **batch_log)
            self.log(
                'posterior_tvm/heldout/eval_target_kl_mean',
                (eval_target_kl * mask).sum() / count.clamp_min(1.0),
                **batch_log)
            for bin_index in range(8):
                selected = node_index // 2 == bin_index
                if bool(selected.any()):
                    selected_mask = mask[selected]
                    self.log(
                        f'posterior_tvm/heldout/local_ce_tbin_{bin_index}',
                        ((token_loss[selected] * selected_mask).sum()
                         / selected_mask.sum().clamp_min(1.0)),
                        **{**batch_log, 'batch_size': int(selected.sum())})
            return Loss(
                loss=nlls / count.clamp_min(1.0), nlls=nlls,
                prior_loss=0.0, num_tokens=count)

        plan = self._global_plan(accumulation_step)
        shape = (clean_tokens.shape[0], clean_tokens.shape[1], self.vocab_size)
        local_noise = self._noise(shape, 'local_noise', accumulation_step)
        local_state = self.corrupt_vocab_state(
            clean_tokens, plan['local_t'], noise=local_noise)
        local_logits = self._logits(
            self.backbone, local_state, plan['local_t'],
            torch.zeros_like(plan['local_t']))
        local_tokens = F.cross_entropy(
            local_logits.transpose(1, 2), clean_tokens, reduction='none')
        mask = valid_tokens.float()
        local_loss = (local_tokens * mask).sum() / mask.sum().clamp_min(1.0)
        if self.local_only:
            self.log('posterior_tvm/local_ce', local_loss.detach(),
                     on_step=True, on_epoch=False, sync_dist=True)
            self.log('loss', local_loss.detach(), prog_bar=True,
                     on_step=True, on_epoch=False, sync_dist=True)
            local_nlls = (local_tokens.detach() * mask).sum()
            return Loss(
                loss=local_loss, nlls=local_nlls,
                prior_loss=0.0, num_tokens=mask.sum())
        map_loss, _ = self._map_objective(
            clean_tokens, plan, accumulation_step)
        completed_step = int(self.global_step) + 1
        if (accumulation_step == 0
                and completed_step in {500, 1000, 2000}
                and map_loss.requires_grad):
            self._gradient_probe(local_loss, map_loss, completed_step)
        lambda_map = self._lambda_map()
        selected_count = int(plan['map_mask'].sum())
        accumulation = int(self.trainer.accumulate_grad_batches)
        map_global_scale = (
            accumulation * selected_count / float(self.map_subset_size))
        total = local_loss + lambda_map * map_global_scale * map_loss
        self.log('posterior_tvm/local_ce', local_loss.detach(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('posterior_tvm/map_loss', map_loss.detach(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('posterior_tvm/map_selected_in_slice',
                 total.new_tensor(float(selected_count)), on_step=True,
                 on_epoch=False, sync_dist=False)
        self.log('posterior_tvm/lambda_map',
                 total.new_tensor(lambda_map), on_step=True,
                 on_epoch=False, sync_dist=True)
        self.log('posterior_tvm/lambda_star', self.map_lambda_star.detach(),
                 on_step=True, on_epoch=False, sync_dist=True)
        self.log('loss', total.detach(), prog_bar=True,
                 on_step=True, on_epoch=False, sync_dist=True)
        # Lightning divides by accumulation. Each slice contains equal local
        # batch size; map_loss is reweighted by its actual selected fraction.
        local_nlls = (local_tokens.detach() * mask).sum()
        return Loss(
            loss=total, nlls=local_nlls,
            prior_loss=0.0, num_tokens=mask.sum())

    def on_validation_epoch_start(self):
        self._posterior_validation_batch_index = 0
        self._validation_online_target_kl = []
        super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        self._posterior_validation_batch_index = int(batch_idx)
        return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self._validation_online_target_kl:
            values = torch.cat(self._validation_online_target_kl)
            self.log(
                'posterior_tvm/heldout/online_target_kl_mean',
                values.mean().to(self.device), on_step=False,
                on_epoch=True, sync_dist=False)
            self.log(
                'posterior_tvm/heldout/online_target_kl_p95',
                torch.quantile(values, 0.95).to(self.device),
                on_step=False, on_epoch=True, sync_dist=False)
        super().on_validation_epoch_end()

    @torch.no_grad()
    def _update_training_target_ema(self):
        next_update = int(self.training_target_ema_updates) + 1
        decay = min(
            self.target_decay_max,
            (next_update + 1.0) / (next_update + 10.0))
        online = tuple(self.backbone.parameters())
        target = tuple(self.training_target_ema.parameters())
        if len(online) != len(target):
            raise RuntimeError('Online and target EMA parameter counts differ.')
        for target_parameter, online_parameter in zip(target, online):
            target_parameter.mul_(decay).add_(
                online_parameter.detach(), alpha=1.0 - decay)
        self.training_target_ema_updates.add_(1)
        self.training_target_ema.eval()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.local_only:
            completed_step = int(self.global_step) + 1
            if completed_step == 5000:
                self._load_fixed_cdf_after_5k()
            return
        self._update_training_target_ema()
        completed_step = int(self.global_step) + 1
        if completed_step in {5000, 10000}:
            self._update_empirical_cdf(completed_step)

    @torch.no_grad()
    def _load_fixed_cdf_after_5k(self):
        artifact = torch.load(
            str(self.fixed_cdf_after_5k_path), map_location='cpu',
            weights_only=False)
        nodes = torch.as_tensor(artifact['sampler_t_nodes']).float()
        values = torch.as_tensor(artifact['sampler_cdf_values']).float()
        if nodes.shape != self.sampler_t_nodes.shape or values.shape != nodes.shape:
            raise ValueError('A@5k fixed CDF artifact has an incompatible shape.')
        if (not torch.isfinite(nodes).all()
                or not torch.isfinite(values).all()
                or not torch.all(nodes[1:] > nodes[:-1])
                or not torch.all(values[1:] >= values[:-1])
                or not math.isclose(float(nodes[0]), 0.0, abs_tol=1e-7)
                or not math.isclose(
                    float(nodes[-1]), self.terminal_t, abs_tol=1e-6)
                or not math.isclose(float(values[0]), 0.0, abs_tol=1e-7)
                or not math.isclose(float(values[-1]), 1.0, abs_tol=1e-7)):
            raise ValueError('A@5k fixed CDF artifact failed validation.')
        self.sampler_t_nodes.copy_(nodes.to(self.sampler_t_nodes))
        self.sampler_cdf_values.copy_(values.to(self.sampler_cdf_values))
        self.sampler_cdf_version.fill_(1)

    @torch.no_grad()
    def _measure_cdf_curve(self, update_step):
        tokens_cpu = self._cdf_clean_tokens_cpu
        nodes = torch.linspace(
            0.0, self.terminal_t, self.cdf_node_count,
            device=self.device)
        per_sequence = torch.empty(
            self.cdf_node_count, self.cdf_bank_size,
            dtype=torch.float64)
        for node_index, t_value in enumerate(nodes):
            for start in range(0, self.cdf_bank_size, self.cdf_query_batch):
                stop = min(start + self.cdf_query_batch, self.cdf_bank_size)
                tokens = tokens_cpu[start:stop].to(self.device)
                t = torch.full(
                    (stop - start,), float(t_value), device=self.device)
                generator = torch.Generator(device=self.device)
                generator.manual_seed(
                    self._seed('cdf', update_step + node_index, start))
                noise = torch.randn(
                    (stop - start, tokens.shape[1], self.vocab_size),
                    device=self.device, dtype=torch.float32,
                    generator=generator)
                state = self.corrupt_vocab_state(tokens, t, noise=noise)
                probability = self._target_probabilities(
                    state, t, torch.zeros_like(t))
                ce = -probability.clamp_min(
                    torch.finfo(torch.float32).tiny).log().gather(
                        -1, tokens.unsqueeze(-1)).squeeze(-1).mean(dim=1)
                per_sequence[node_index, start:stop] = ce.double().cpu()
        return nodes.cpu().numpy(), per_sequence.numpy()

    @torch.no_grad()
    def _update_empirical_cdf(self, update_step):
        nodes, per_sequence = self._measure_cdf_curve(update_step)
        fitted = _isotonic_nonincreasing(per_sequence.mean(axis=1))
        denominator = fitted[0] - fitted[-1]
        accepted = bool(denominator >= self.cdf_denominator_min)
        reason = 'accepted'
        information = None
        bootstrap_spread = None
        if accepted:
            information = (fitted[0] - fitted) / denominator
            rng = np.random.default_rng(self.plan_seed + update_step)
            bootstrap = []
            for _ in range(32):
                indices = rng.integers(
                    0, self.cdf_bank_size, size=self.cdf_bank_size)
                curve = _isotonic_nonincreasing(
                    per_sequence[:, indices].mean(axis=1))
                span = curve[0] - curve[-1]
                if span < self.cdf_denominator_min:
                    accepted = False
                    reason = 'bootstrap_denominator_too_small'
                    break
                bootstrap.append((curve[0] - curve) / span)
            if accepted:
                bootstrap = np.stack(bootstrap)
                bootstrap_spread = float(np.max(
                    np.quantile(bootstrap, 0.9, axis=0)
                    - np.quantile(bootstrap, 0.1, axis=0)))
                if bootstrap_spread > self.cdf_bootstrap_spread_max:
                    accepted = False
                    reason = 'bootstrap_variation_too_large'
        else:
            reason = 'denominator_too_small_or_curve_flat'
        if accepted:
            new_cdf = 0.5 * nodes / self.terminal_t + 0.5 * information
            dense = np.interp(
                self.sampler_t_nodes.detach().cpu().numpy(), nodes, new_cdf)
            dense = np.maximum.accumulate(dense)
            dense[0], dense[-1] = 0.0, 1.0
            self.sampler_cdf_values.copy_(torch.as_tensor(
                dense, device=self.device, dtype=torch.float32))
            self.sampler_cdf_version.add_(1)
        if self.trainer.is_global_zero:
            path = os.path.join(
                self.config.checkpointing.save_dir, 'cdf_updates.jsonl')
            record = {
                'optimizer_step': update_step, 'accepted': accepted,
                'reason': reason, 'denominator': float(denominator),
                'bootstrap_spread': bootstrap_spread,
                'cdf_version': int(self.sampler_cdf_version),
                'nodes': nodes.tolist(), 'raw_ce': per_sequence.mean(axis=1).tolist(),
                'isotonic_ce': fitted.tolist(),
            }
            with open(path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, sort_keys=True) + '\n')

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if (parameter.ndim == 1 or name.endswith('.bias')
                    or 'vocab_embed' in name or 'norm' in name.lower()):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': 0.01},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=3e-4, betas=(0.9, 0.95), eps=float(self.config.optim.eps))

        def multiplier(step):
            step = int(step)
            if step < 1000:
                return max(step, 1) / 1000.0
            if step <= 30000:
                return 1.0
            progress = min(max((step - 30000) / 20000.0, 0.0), 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=multiplier)
        return [optimizer], [{
            'scheduler': scheduler, 'interval': 'step',
            'monitor': 'val/loss', 'name': 'trainer/lr'}]

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        gradients = [
            parameter.grad.detach().float().norm()
            for parameter in self.backbone.parameters()
            if parameter.grad is not None]
        total = torch.stack(gradients).norm() if gradients else torch.zeros(
            (), device=self.device)
        self.log('posterior_tvm/gradient/total_norm', total,
                 on_step=True, on_epoch=False, sync_dist=True)
        projector = self.backbone.finite_map_sc_projector
        if projector is not None:
            sc_gradients = [
                parameter.grad.detach().float().norm()
                for parameter in projector.parameters()
                if parameter.grad is not None]
            sc_total = (
                torch.stack(sc_gradients).norm() if sc_gradients
                else torch.zeros((), device=self.device))
            self.log('posterior_tvm/gradient/sc_projection_norm', sc_total,
                     on_step=True, on_epoch=False, sync_dist=True)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        grid = tuple(float(value) for value in getattr(
            self.config.algo, 'tvm_inference_grid_physical', []))
        if len(grid) < 2:
            raise ValueError('Evaluation requires an explicit physical-t grid.')
        if (not math.isclose(grid[0], 0.0, abs_tol=1e-8)
                or not math.isclose(grid[-1], self.terminal_t, abs_tol=1e-8)
                or any(left >= right for left, right in zip(grid, grid[1:]))):
            raise ValueError('Grid must increase exactly from 0 to T=0.95.')
        expected = len(grid) - 1
        if num_steps is not None and int(num_steps) != expected:
            raise ValueError(
                f'Configured grid has {expected} model calls, got {num_steps}.')
        state = torch.empty(
            num_samples, self.num_tokens, self.vocab_size,
            device=self.device, dtype=torch.float32)
        seed = int(getattr(
            self.config.sampling, 'task1_initial_noise_seed', self.plan_seed))
        offset = int(getattr(self, '_task1_sampling_sample_offset', 0))
        generator = torch.Generator(device=self.device)
        for index in range(num_samples):
            generator.manual_seed(seed + offset + index)
            state[index].normal_(generator=generator)
        self._task1_sampling_sample_offset = offset + num_samples
        hidden = int(self.config.model.hidden_size)
        cache = torch.zeros(
            num_samples, self.num_tokens, hidden,
            device=self.device, dtype=torch.float32)
        previous_eta = torch.zeros(num_samples, device=self.device)
        valid = torch.zeros(num_samples, device=self.device)
        calls = 0
        inference_mode = str(getattr(
            self.config.algo, 'posterior_tvm_inference_mode', 'finite_map'))
        if inference_mode not in {'finite_map', 'local_field'}:
            raise ValueError(
                'posterior_tvm_inference_mode must be finite_map or local_field.')
        for left, right in zip(grid, grid[1:]):
            r = torch.full((num_samples,), left, device=self.device)
            s = torch.full_like(r, right)
            eta = (s - r) / (1.0 - r).clamp_min(1e-8)
            query_eta = (
                torch.zeros_like(eta)
                if inference_mode == 'local_field' else eta)
            logits = self._logits(
                self.backbone, state, r, query_eta,
                cache=cache, previous_eta=previous_eta, valid=valid)
            calls += 1
            probability = F.softmax(logits.float(), dim=-1)
            state = finite_map_update(state, probability, r, s)
            if self.variant == 'B_finite_sc' and inference_mode == 'finite_map':
                cache = self.backbone.vocab_embed(
                    probability).detach().float()
                previous_eta = eta.detach()
                valid = torch.ones_like(valid)
        if calls != expected:
            raise RuntimeError('Strict NFE counter mismatch.')
        self.last_sampling_nfe = calls
        return state.argmax(dim=-1)
