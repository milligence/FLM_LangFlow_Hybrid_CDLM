"""Fixed-teacher TVM-CE finite-map student for Task1.

The implementation intentionally keeps the TVM differential objective separate
from the existing FMLM semigroup/self-distillation classes.  The student uses
the existing Task1 vocabulary-space backbone with a zero-initialized second
time-conditioning branch for the relative jump ``eta``.
"""

import copy
import json
import math
import os
import shutil

import hydra
import torch
import torch.nn.functional as F

from algo import LangFlowFLMHybrid
from langflow_hybrid.ops import (
    detached_self_conditioning_embedding,
    finite_sc_gate,
    finite_sc_update,
    flm_vocab_gaussian_bias,
)


def tvm_quantity_from_logits_jvp(logits, weighted_logit_jvp):
    """Return ``R, mu, g, Q`` for the relative-jump TVM parameterization."""
    logits = logits.float()
    weighted_logit_jvp = weighted_logit_jvp.float()
    probabilities = F.softmax(logits, dim=-1)
    mean_jvp = (probabilities * weighted_logit_jvp).sum(
        dim=-1, keepdim=True)
    gate = 1.0 + weighted_logit_jvp - mean_jvp
    quantity = probabilities * gate
    return probabilities, mean_jvp, gate, quantity


def _log_smoothcap(value, kappa):
    """Stable log(kappa * softplus(value / kappa))."""
    scaled = value.float() / float(kappa)
    log_softplus = torch.where(
        scaled < -20.0,
        scaled,
        F.softplus(scaled.clamp_min(-20.0)).log())
    return math.log(float(kappa)) + log_softplus


def _teacher_log_smoothcap(probability, reference, kappa,
                           batch_chunk_size=4):
    """Stable log(S_kappa(probability / reference)) with bounded peak memory."""
    tiny = torch.finfo(torch.float32).tiny
    log_kappa = math.log(float(kappa))
    log_twenty = math.log(20.0)
    if probability.shape != reference.shape:
        raise ValueError(
            'Teacher probability and reference shapes must match, got '
            f'{tuple(probability.shape)} and {tuple(reference.shape)}.')
    if probability.ndim == 0:
        raise ValueError('Teacher calibration expects a batch dimension.')
    chunk_size = max(1, int(batch_chunk_size))
    output = torch.empty_like(probability, dtype=torch.float32)
    for start in range(0, probability.shape[0], chunk_size):
        stop = min(start + chunk_size, probability.shape[0])
        log_ratio_over_kappa = (
            probability[start:stop].float().clamp_min(tiny).log()
            - reference[start:stop].float().clamp_min(tiny).log()
            - log_kappa)
        # For z >= 20, softplus(z) and z are equal to float32 precision.
        # Chunking prevents several full [batch, sequence, vocab] float32
        # temporaries from coexisting at the teacher-calibration peak.
        exact = log_ratio_over_kappa <= log_twenty
        bounded_z = log_ratio_over_kappa.clamp_max(log_twenty).exp()
        log_softplus = F.softplus(bounded_z).log()
        chunk = log_kappa + torch.where(
            exact, log_softplus, log_ratio_over_kappa)
        output[start:stop].copy_(chunk)
    return output


def calibrated_relative_distributions(probabilities, gate,
                                      teacher_probabilities, kappa):
    """Build calibrated student log-probabilities and detached teacher target."""
    if not math.isfinite(float(kappa)) or float(kappa) <= 0.0:
        raise ValueError(f'kappa must be finite and positive, got {kappa!r}.')
    tiny = torch.finfo(torch.float32).tiny
    log_reference = probabilities.float().clamp_min(tiny).log()

    student_logits = log_reference + _log_smoothcap(gate, kappa)
    student_log_probabilities = student_logits - torch.logsumexp(
        student_logits, dim=-1, keepdim=True)

    detached_reference = probabilities.detach().float()
    teacher_logits = _teacher_log_smoothcap(
        teacher_probabilities.detach(), detached_reference, kappa)
    teacher_logits.add_(detached_reference.clamp_min(tiny).log())
    teacher_probabilities_calibrated = F.softmax(
        teacher_logits, dim=-1).detach()
    return student_log_probabilities, teacher_probabilities_calibrated


def finite_map_update(state, probabilities, r, s, eps=1e-8):
    """Apply x_s = (1-eta) x_r + eta R for a configured interval."""
    denominator = (1.0 - r.float()).clamp_min(float(eps))
    eta = ((s.float() - r.float()) / denominator).clamp(0.0, 1.0)
    eta_view = eta.view(-1, *([1] * (state.ndim - 1)))
    return (1.0 - eta_view) * state + eta_view * probabilities


def finite_sc_regression_state(probabilities, sc_coordinate, residual,
                               carried_state, delta, d50):
    """SC endpoint used by ``L_H`` with the denoiser coordinate detached."""
    proposal = torch.matmul(
        probabilities.detach().float(), sc_coordinate.detach().float())
    proposal = proposal + residual.float()
    return finite_sc_update(carried_state.float(), proposal, delta, d50)


def gap_mode_probabilities(progress):
    """Linearly interpolate early and steady normalized-gap mixtures."""
    progress = float(max(0.0, min(1.0, progress)))
    initial = (0.15, 0.45, 0.30, 0.10)
    steady = (0.10, 0.20, 0.30, 0.40)
    return tuple(
        start + progress * (end - start)
        for start, end in zip(initial, steady))


def integer_quota_counts(probabilities, total):
    """Convert probabilities to deterministic integer quotas summing to total."""
    raw = [float(probability) * int(total) for probability in probabilities]
    counts = [math.floor(value) for value in raw]
    remainder = int(total) - sum(counts)
    order = sorted(
        range(len(raw)), key=lambda index: (raw[index] - counts[index], -index),
        reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def sample_global_interval_batch(global_batch_size, s_max, r0_count,
                                 curriculum_progress, device, seed):
    """Sample one shuffled global-batch interval plan with exact quotas."""
    size = int(global_batch_size)
    if not 0 <= int(r0_count) <= size:
        raise ValueError('r0_count must lie within the global batch.')
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    probabilities = gap_mode_probabilities(curriculum_progress)
    counts = integer_quota_counts(probabilities, size)
    mode = torch.repeat_interleave(
        torch.arange(4, device=device, dtype=torch.long),
        torch.tensor(counts, device=device, dtype=torch.long))
    mode = mode[torch.randperm(size, device=device, generator=generator)]

    r_zero = torch.zeros(size, device=device, dtype=torch.bool)
    r_zero[:int(r0_count)] = True
    r_zero = r_zero[
        torch.randperm(size, device=device, generator=generator)]
    r = torch.rand(size, device=device, generator=generator) * float(s_max)
    r[r_zero] = 0.0

    unit = torch.rand(size, device=device, generator=generator)
    z = torch.empty(size, device=device, dtype=torch.float32)
    z[mode == 0] = 0.0
    z[mode == 1] = unit[mode == 1] * 0.2
    z[mode == 2] = 0.2 + unit[mode == 2] * 0.4
    z[mode == 3] = 0.6 + unit[mode == 3] * 0.4
    s = r + z * (float(s_max) - r)
    return r, s, mode == 0, torch.tensor(
        probabilities, device=device, dtype=torch.float32), mode, counts


class Task1TVMCE(LangFlowFLMHybrid):
    """Two-time finite-map student trained by fixed-teacher TVM-CE."""

    _ENDPOINT_SC_MODES = {'zero_sc'}

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.teacher_model = None
        self._student_initialized_from_teacher = False
        self.kappa = float(config.algo.tvm_kappa)
        self.main_loss_type = str(config.algo.tvm_main_loss)
        self.negative_mass_weight = float(
            config.algo.tvm_negative_mass_weight)
        self.s_max = float(config.algo.tvm_teacher_s_max)
        self.r0_probability = float(config.algo.tvm_r0_probability)
        self.training_budget_steps = int(
            config.algo.tvm_training_budget_steps)
        curriculum_fraction = getattr(
            config.algo, 'tvm_curriculum_fraction', None)
        self.curriculum_fraction = (
            None if curriculum_fraction is None
            else float(curriculum_fraction))
        self.endpoint_weight = float(config.algo.tvm_endpoint_weight)
        self.endpoint_sc_mode = str(config.algo.tvm_endpoint_teacher_sc_mode)
        self._teacher_forward_calls = 0
        self._student_forward_calls = 0
        self._student_jvp_calls = 0
        self._validate_tvm_configuration()

    def _validate_tvm_configuration(self):
        if not bool(self.config.algo.double_temb):
            raise ValueError('Task1 TVM-CE requires double_temb=true.')
        if self.state_space != 'vocab' or self.corruption != 'flm_linear_gaussian':
            raise ValueError(
                'Task1 TVM-CE requires the Task1 vocabulary-space linear '
                'Gaussian path.')
        if not 0.0 < self.s_max < 1.0:
            raise ValueError('tvm_teacher_s_max must lie strictly below 1.')
        if self.main_loss_type not in {
                'calibrated_relative_ce', 'raw_q_mse'}:
            raise ValueError(
                f'Unsupported TVM main loss {self.main_loss_type!r}.')
        if not 0.0 <= self.r0_probability <= 1.0:
            raise ValueError('tvm_r0_probability must lie in [0, 1].')
        if self.training_budget_steps <= 0:
            raise ValueError('tvm_training_budget_steps must be positive.')
        if self.curriculum_fraction is None:
            raise ValueError(
                'The stage config must choose tvm_curriculum_fraction within '
                'the supplied [0.05, 0.10] range.')
        if not 0.05 <= self.curriculum_fraction <= 0.10:
            raise ValueError(
                'tvm_curriculum_fraction must remain within the supplied '
                '[0.05, 0.10] range.')
        if self.endpoint_sc_mode not in self._ENDPOINT_SC_MODES:
            raise ValueError(
                'Unsupported endpoint teacher SC mode '
                f'{self.endpoint_sc_mode!r}.')
        if self.endpoint_weight != 0.0:
            raise ValueError(
                'The Stage-1 pilot keeps endpoint auxiliary disabled; its '
                'selected teacher mode is zero_sc for the later endpoint stage.')

    def load_state_dict(self, state_dict, strict=True):
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith('teacher_model.')}
        return super().load_state_dict(state_dict, strict=strict)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        state_dict = checkpoint.get('state_dict', {})
        for key in tuple(state_dict):
            if key.startswith('teacher_model.'):
                del state_dict[key]

    def setup(self, stage: str):
        if self.teacher_model is None:
            teacher_path = str(self.config.algo.teacher_path)
            if not teacher_path:
                raise ValueError('Task1 TVM-CE requires teacher_path.')
            self.teacher_model = self._load_teacher_model(
                teacher_path, use_plain_config=True)
        if (stage == 'fit'
                and not self._is_resuming
                and not self._student_initialized_from_teacher):
            self._copy_teacher_weights_to_student(
                self.teacher_model.state_dict())
            self._synchronize_student_ema()
            self._student_initialized_from_teacher = True

    def task1_training_time_contract(self):
        return {
            'mode': 'tvm_normalized_gap_curriculum',
            'coordinate': 'physical_r_s_with_relative_eta',
            'start_time': {
                'r_zero_probability': self.r0_probability,
                'otherwise': f'uniform_0_{self.s_max}',
            },
            'normalized_gap': {
                'definition': 'z=(s-r)/(s_max-r)',
                'bands': {
                    'diagonal': [0.0, 0.0],
                    'short': [0.0, 0.2],
                    'medium': [0.2, 0.6],
                    'long': [0.6, 1.0],
                },
                'initial_probabilities': list(
                    gap_mode_probabilities(0.0)),
                'steady_probabilities': list(
                    gap_mode_probabilities(1.0)),
            },
            'curriculum_fraction': self.curriculum_fraction,
            'training_budget_steps': self.training_budget_steps,
            'preset_inference_grid': False,
            'main_loss': self.main_loss_type,
            'teacher_query_upper_bound': self.s_max,
            'endpoint_auxiliary_weight': self.endpoint_weight,
            'endpoint_teacher_sc_mode': self.endpoint_sc_mode,
        }

    def _synchronize_student_ema(self):
        if self.ema is None:
            return
        self.ema.shadow_params = [
            parameter.detach().clone()
            for parameter in self._get_parameters()
            if parameter.requires_grad]
        self.ema.num_updates = 0

    def _student_logits(
            self, state, r, eta, use_jvp_attn=False, model_r=None,
            x_self_cond=None):
        if model_r is None:
            q_r = self._t_to_tau(r.float().clamp(0.0, 1.0))
            model_r = self._task1_condition(q_r, r).to(self.device)
        state_embedding = self._state_embedding(state)
        residual_logits = self.backbone(
            state_embedding,
            self._process_sigma(model_r),
            self._process_sigma(eta),
            use_jvp_attn=use_jvp_attn,
            inputs_are_embeddings=True,
            x_self_cond=x_self_cond)
        bias = flm_vocab_gaussian_bias(
            state, r, self._current_token_bias_weight(), self.flm_time_eps)
        return residual_logits.float() + bias.float()

    @torch.no_grad()
    def _teacher_local_zero_sc(self, state, t):
        self.teacher_model.eval()
        q_t = self._t_to_tau(t.float().clamp(0.0, 1.0))
        model_t = self._task1_condition(q_t, t).to(self.device)
        teacher_state_embedding = self.teacher_model.vocab_embed(state)
        residual_logits = self.teacher_model(
            teacher_state_embedding,
            self._process_sigma(model_t),
            inputs_are_embeddings=True,
            x_self_cond=None)
        bias = flm_vocab_gaussian_bias(
            state, t,
            float(self.config.algo.tvm_teacher_bias_weight),
            self.flm_time_eps)
        return F.softmax(residual_logits.float() + bias.float(), dim=-1)

    def _sample_training_intervals(self, batch_size, accumulation_step=None):
        device = self.device
        curriculum_steps = max(
            1, round(self.training_budget_steps * self.curriculum_fraction))
        progress = min(float(self.global_step) / curriculum_steps, 1.0)
        if accumulation_step is None:
            global_size = int(batch_size)
            r0_count = round(global_size * self.r0_probability)
            return sample_global_interval_batch(
                global_size, self.s_max, r0_count, progress, device,
                seed=int(self.config.seed) + int(self.global_step) * 1000003)

        global_size = int(self.config.loader.global_batch_size)
        total_chunks = (
            int(self.trainer.num_nodes)
            * int(self.trainer.num_devices)
            * int(self.trainer.accumulate_grad_batches))
        if global_size % total_chunks != 0:
            raise ValueError(
                'TVM global batch must divide evenly across nodes, devices, '
                'and gradient-accumulation slices.')
        expected_batch = global_size // total_chunks
        if int(batch_size) != expected_batch:
            raise ValueError(
                f'TVM expected local microbatch {expected_batch}, got '
                f'{batch_size}; exact global quotas cannot be preserved.')
        chunk_index = (
            int(self.trainer.node_rank)
            * int(self.trainer.num_devices)
            * int(self.trainer.accumulate_grad_batches)
            + int(self.trainer.local_rank)
            * int(self.trainer.accumulate_grad_batches)
            + int(accumulation_step))
        plan = sample_global_interval_batch(
            global_size, self.s_max,
            round(global_size * self.r0_probability), progress, device,
            seed=int(self.config.seed) + int(self.global_step) * 1000003)
        start = chunk_index * expected_batch
        stop = start + expected_batch
        r, s, diagonal, probabilities, mode, counts = plan
        return (r[start:stop], s[start:stop], diagonal[start:stop],
                probabilities, mode[start:stop], counts)

    def loss(self, clean_tokens, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, train_mode
        del xT, given_t, not_sampling_t
        batch_size, length = clean_tokens.shape
        r, s, diagonal, mode_probabilities, gap_mode, gap_counts = (
            self._sample_training_intervals(
                batch_size, current_accumulation_step))
        state = self.corrupt_vocab_state(clean_tokens, r)
        eta = ((s - r) / (1.0 - r).clamp_min(1e-8)).clamp(0.0, 1.0)
        loss = state.new_zeros((batch_size, length), dtype=torch.float32)

        if diagonal.any():
            index = diagonal.nonzero(as_tuple=True)[0]
            self._student_forward_calls += 1
            logits = self._student_logits(
                state[index], r[index], eta[index], use_jvp_attn=False)
            self._teacher_forward_calls += 1
            target = self._teacher_local_zero_sc(state[index], r[index])
            loss[index] = -(target * F.log_softmax(
                logits.float(), dim=-1)).sum(dim=-1)

        off_diagonal = ~diagonal
        if off_diagonal.any():
            index = off_diagonal.nonzero(as_tuple=True)[0]
            state_off = state[index]
            r_off = r[index]
            s_off = s[index]
            eta_off = eta[index]
            q_r_off = self._t_to_tau(r_off.float().clamp(0.0, 1.0))
            model_r_off = self._task1_condition(q_r_off, r_off).to(
                self.device).detach()

            def logits_at_eta(eta_value):
                return self._student_logits(
                    state_off, r_off, eta_value, use_jvp_attn=True,
                    model_r=model_r_off)

            self._student_forward_calls += 1
            self._student_jvp_calls += 1
            logits, weighted_logit_jvp = torch.func.jvp(
                logits_at_eta,
                (eta_off,),
                (eta_off * (1.0 - eta_off),))
            probabilities, _, gate, quantity = (
                tvm_quantity_from_logits_jvp(
                    logits, weighted_logit_jvp))
            endpoint = finite_map_update(
                state_off, probabilities, r_off, s_off)
            self._teacher_forward_calls += 1
            teacher = self._teacher_local_zero_sc(
                endpoint.detach(), s_off).detach()
            student_log_probabilities, calibrated_teacher = (
                calibrated_relative_distributions(
                    probabilities, gate, teacher, self.kappa))
            calibrated_ce = -(
                calibrated_teacher * student_log_probabilities).sum(dim=-1)
            raw_residual = quantity - teacher
            if self.main_loss_type == 'calibrated_relative_ce':
                main_loss = calibrated_ce
            else:
                main_loss = raw_residual.square().sum(dim=-1)
            negative_mass = F.relu(-quantity).sum(dim=-1)
            loss[index] = (
                main_loss
                + self.negative_mass_weight * negative_mass.square())

            self.log(
                'tvm/raw_sum_error',
                (quantity.sum(dim=-1) - 1.0).abs().mean().detach(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/negative_mass', negative_mass.mean().detach(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/raw_q_teacher_l1',
                raw_residual.detach().abs().sum(dim=-1).mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            velocity_residual = (
                raw_residual.detach().abs().sum(dim=-1)
                / (1.0 - s_off).clamp_min(1e-8).unsqueeze(-1))
            self.log(
                'tvm/velocity_equivalent_l1', velocity_residual.mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            calibrated_student = student_log_probabilities.detach().exp()
            self.log(
                'tvm/student_cap_l1',
                (calibrated_student - quantity.detach()).abs().sum(
                    dim=-1).mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/teacher_calibration_l1',
                (calibrated_teacher - teacher).abs().sum(dim=-1).mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/calibrated_ce', calibrated_ce.detach().mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            calibrated_kl = (
                calibrated_teacher
                * (calibrated_teacher.clamp_min(
                    torch.finfo(torch.float32).tiny).log()
                   - student_log_probabilities.detach())).sum(dim=-1)
            self.log(
                'tvm/calibrated_kl', calibrated_kl.mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            teacher_log = teacher.clamp_min(
                torch.finfo(torch.float32).tiny).log()
            self.log(
                'tvm/teacher_entropy',
                -(teacher * teacher_log).sum(dim=-1).mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/teacher_top1', teacher.max(dim=-1).values.mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'tvm/jvp_norm',
                weighted_logit_jvp.detach().square().mean().sqrt(),
                on_step=True, on_epoch=False, sync_dist=True)

        self.log(
            'loss', loss.detach().mean(), prog_bar=True,
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm/r_zero_fraction', (r == 0.0).float().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm/r_mean', r.mean(), on_step=True, on_epoch=False,
            sync_dist=True)
        self.log(
            'tvm/s_mean', s.mean(), on_step=True, on_epoch=False,
            sync_dist=True)
        self.log(
            'tvm/eta_mean', eta.mean(), on_step=True, on_epoch=False,
            sync_dist=True)
        self.log(
            'tvm/endpoint_loss', loss.new_zeros(()),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm/sequences_seen_nominal',
            loss.new_tensor(float(
                (int(self.global_step) + 1)
                * int(self.config.loader.global_batch_size))),
            on_step=True, on_epoch=False, sync_dist=False)
        self.log(
            'tvm/teacher_forward_calls_per_rank',
            loss.new_tensor(float(self._teacher_forward_calls)),
            on_step=True, on_epoch=False, sync_dist=False)
        self.log(
            'tvm/student_forward_calls_per_rank',
            loss.new_tensor(float(self._student_forward_calls)),
            on_step=True, on_epoch=False, sync_dist=False)
        self.log(
            'tvm/student_jvp_calls_per_rank',
            loss.new_tensor(float(self._student_jvp_calls)),
            on_step=True, on_epoch=False, sync_dist=False)
        for index, name in enumerate(
                ('diagonal', 'short', 'medium', 'long')):
            self.log(
                f'tvm/gap_mode_sampled_{name}',
                (gap_mode == index).float().mean(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                f'tvm/gap_mode_probability_{name}',
                mode_probabilities[index],
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                f'tvm/gap_mode_global_quota_{name}',
                loss.new_tensor(float(gap_counts[index])),
                on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        gradients = [
            parameter.grad.detach().float().norm(2)
            for parameter in self._get_parameters()
            if parameter.grad is not None]
        total = torch.stack(gradients).norm(2) if gradients else torch.zeros(
            (), device=self.device)
        eta_gradients = [
            parameter.grad.detach().float().norm(2)
            for parameter in self.backbone.sigma_map_prime.parameters()
            if parameter.grad is not None]
        eta_total = (
            torch.stack(eta_gradients).norm(2) if eta_gradients
            else torch.zeros((), device=self.device))
        self.log(
            'tvm/total_grad_norm', total, on_step=True, on_epoch=False,
            sync_dist=True)
        self.log(
            'tvm/eta_conditioning_grad_norm', eta_total,
            on_step=True, on_epoch=False, sync_dist=True)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        inference_mode = str(getattr(
            self.config.algo, 'tvm_inference_mode', 'finite_map'))
        self_conditioning_mode = str(getattr(
            self.config.algo, 'tvm_inference_self_conditioning', 'off'))
        if inference_mode not in {'finite_map', 'local_eta_zero'}:
            raise ValueError(
                'tvm_inference_mode must be finite_map or local_eta_zero, '
                f'got {inference_mode!r}.')
        if self_conditioning_mode not in {'off', 'rolling'}:
            raise ValueError(
                'tvm_inference_self_conditioning must be off or rolling, '
                f'got {self_conditioning_mode!r}.')
        if (inference_mode == 'finite_map'
                and self_conditioning_mode != 'off'):
            raise ValueError(
                'Rolling self-conditioning is only defined for the '
                'local_eta_zero diagnostic.')
        grid = tuple(float(value) for value in getattr(
            self.config.algo, 'tvm_inference_grid_physical', []))
        if len(grid) < 2:
            raise ValueError(
                'Evaluation must supply an explicit inference grid; broad '
                'map training does not assume one.')
        if (not math.isclose(grid[0], 0.0, abs_tol=1e-8)
                or not math.isclose(grid[-1], 1.0, abs_tol=1e-8)
                or any(left >= right for left, right in zip(grid, grid[1:]))):
            raise ValueError(
                'tvm_inference_grid_physical must increase from 0 to 1.')
        expected_steps = len(grid) - 1
        if num_steps is not None and int(num_steps) != expected_steps:
            raise ValueError(
                f'Configured TVM grid has {expected_steps} intervals, '
                f'but num_steps={num_steps}.')
        state = torch.randn(
            (num_samples, self.num_tokens, self.vocab_size),
            device=self.device, dtype=self.dtype)
        x_self_cond = None
        for left, right in zip(grid, grid[1:]):
            r = torch.full(
                (num_samples,), left, device=self.device,
                dtype=torch.float32)
            s = torch.full_like(r, right)
            step_eta = (
                (s - r) / (1.0 - r).clamp_min(1e-8)).clamp(0.0, 1.0)
            eta_condition = (
                torch.zeros_like(step_eta)
                if inference_mode == 'local_eta_zero'
                else step_eta)
            logits = self._student_logits(
                state, r, eta_condition, use_jvp_attn=False,
                x_self_cond=x_self_cond)
            probabilities = F.softmax(logits.float(), dim=-1)
            if self_conditioning_mode == 'rolling':
                x_self_cond = detached_self_conditioning_embedding(
                    probabilities, self.embed_probabilities)
            state = finite_map_update(state, probabilities, r, s)
        self.last_sampling_nfe = expected_steps
        return state.argmax(dim=-1)

class Task1TVMSCRepair(Task1TVMCE):
    """Gap-aware finite-map SC repair initialized from the 10k TVM student."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.backbone.sc_residual_head is None:
            raise ValueError(
                'SC repair requires finite_sc_residual_head=true.')
        self.sc_d50 = float(config.algo.tvm_sc_gate_d50)
        self.sc_pair_weight = float(config.algo.tvm_sc_pair_weight)
        self.sc_gradient_probe_interval = int(
            config.algo.tvm_sc_gradient_probe_interval)
        self.sc_gradient_probe_batch_size = int(
            config.algo.tvm_sc_gradient_probe_batch_size)
        self.sc_gradient_probe_inline = bool(getattr(
            config.algo, 'tvm_sc_gradient_probe_inline', True))
        self.sc_aware_fraction = float(config.algo.tvm_sc_aware_fraction)
        self.sc_short_weight_floor = float(
            config.algo.tvm_sc_short_weight_floor)
        self.student_init_path = str(config.algo.student_init_path)
        self.sc_pair_bank_path = str(config.algo.tvm_sc_pair_bank_path)
        self.sc_step0_checkpoint_path = str(
            config.algo.tvm_sc_step0_checkpoint_path)
        hidden_size = int(config.model.hidden_size)
        self.register_buffer(
            'sc_coordinate',
            torch.zeros(self.vocab_size, hidden_size, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'sc_coordinate_ready', torch.tensor(False), persistent=True)
        self._sc_pair_bank = None
        self._sc_state_variance = None
        self._sc_state_global_variance = None
        self._validate_sc_repair_configuration()

    def _validate_sc_repair_configuration(self):
        finite_sc_gate(torch.zeros(()), self.sc_d50)
        if not 0.0 < self.sc_aware_fraction < 1.0:
            raise ValueError('tvm_sc_aware_fraction must lie in (0, 1).')
        if self.sc_pair_weight < 0.0:
            raise ValueError('SC repair pair weight must be non-negative.')
        if self.sc_short_weight_floor <= 0.0:
            raise ValueError('SC short-gap weight floor must be positive.')
        if (self.sc_gradient_probe_interval <= 0
                or self.sc_gradient_probe_batch_size <= 0):
            raise ValueError(
                'SC gradient-probe sizes must be positive.')

    def _load_teacher_without_student_sc_head(self, teacher_path):
        enabled = bool(self.config.algo.finite_sc_residual_head)
        self.config.algo.finite_sc_residual_head = False
        try:
            return self._load_teacher_model(
                teacher_path, use_plain_config=True)
        finally:
            self.config.algo.finite_sc_residual_head = enabled

    def _load_student_ema_initialization(self, path):
        if not path or not os.path.isfile(path):
            raise ValueError(
                f'SC repair requires the 10k student checkpoint: {path!r}.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(self.backbone, checkpoint)
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            'rotary_emb.inv_freq',
            'sc_residual_head.0.weight',
            'sc_residual_head.0.bias',
            'sc_residual_head.2.weight',
            'sc_residual_head.2.bias',
        }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'Unexpected 10k EMA initialization mismatch: '
                f'missing={sorted(unexpected_missing)}, '
                f'unexpected={sorted(incompatible.unexpected_keys)}.')
        with torch.no_grad():
            self.backbone.sc_residual_head[-1].weight.zero_()
            self.backbone.sc_residual_head[-1].bias.zero_()

    def _load_sc_pair_bank(self):
        if not self.sc_pair_bank_path or not os.path.isfile(
                self.sc_pair_bank_path):
            raise ValueError(
                'SC repair requires a generated reference pair bank at '
                f'{self.sc_pair_bank_path!r}.')
        payload = torch.load(
            self.sc_pair_bank_path, map_location='cpu', weights_only=False)
        required = {'state_r', 'h_r', 'r', 's', 'r_ref', 'h_t',
                    'state_variance', 'metadata'}
        missing = required - set(payload)
        if missing:
            raise ValueError(
                f'SC pair bank is missing fields: {sorted(missing)}.')
        metadata = payload['metadata']
        if not math.isclose(
                float(metadata['d50']), self.sc_d50,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                'SC pair bank d50 does not match the selected gate: '
                f'{metadata["d50"]} != {self.sc_d50}.')
        size = int(payload['r'].shape[0])
        for name in ('state_r', 'h_r', 's', 'r_ref', 'h_t'):
            if int(payload[name].shape[0]) != size:
                raise ValueError(
                    f'SC pair bank field {name} has inconsistent size.')
        self._sc_pair_bank = {
            name: payload[name].contiguous()
            for name in ('state_r', 'h_r', 'r', 's', 'r_ref', 'h_t')
        }
        self._sc_state_variance = payload['state_variance'].float()
        self._sc_state_global_variance = float(
            self._sc_state_variance.mean())

    def setup(self, stage: str):
        if self.teacher_model is None:
            teacher_path = str(self.config.algo.teacher_path)
            if not teacher_path:
                raise ValueError('SC repair requires teacher_path.')
            self.teacher_model = self._load_teacher_without_student_sc_head(
                teacher_path)
        if not bool(self.sc_coordinate_ready):
            with torch.no_grad():
                self.sc_coordinate.copy_(
                    self.teacher_model.vocab_embed.normalized_weight()
                    .detach().float())
                self.sc_coordinate_ready.fill_(True)
        if (stage == 'fit'
                and not self._is_resuming
                and not self._student_initialized_from_teacher):
            self._load_student_ema_initialization(self.student_init_path)
            self._synchronize_student_ema()
            self._student_initialized_from_teacher = True
        if stage in {'fit', 'diagnostic'} and self._sc_pair_bank is None:
            self._load_sc_pair_bank()

    def configure_optimizers(self):
        head_ids = {
            id(parameter)
            for parameter in self.backbone.sc_residual_head.parameters()
            if parameter.requires_grad}
        sc_projection_ids = {
            id(parameter)
            for parameter in self.backbone.self_cond_proj.parameters()
            if parameter.requires_grad}
        base_parameters = []
        sc_projection_parameters = []
        head_parameters = []
        for parameter in self._get_optimizer_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in head_ids:
                head_parameters.append(parameter)
            elif id(parameter) in sc_projection_ids:
                sc_projection_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {'params': base_parameters,
                 'lr': float(self.config.algo.tvm_sc_backbone_lr),
                 'name': 'backbone_existing'},
                {'params': sc_projection_parameters,
                 'lr': float(self.config.algo.tvm_sc_input_projection_lr),
                 'name': 'self_conditioning_input_projection'},
                {'params': head_parameters,
                 'lr': float(self.config.algo.tvm_sc_residual_head_lr),
                 'name': 'finite_sc_residual_head'},
            ],
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)
        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }]

    def on_train_start(self):
        super().on_train_start()
        path = self.sc_step0_checkpoint_path
        if not path or self.global_step != 0 or self._is_resuming:
            return
        if os.path.exists(path):
            raise FileExistsError(
                f'Refusing to overwrite SC-repair step-0 checkpoint: {path}')
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.trainer.save_checkpoint(path, weights_only=False)

    def _teacher_local(self, state, t, h):
        self.teacher_model.eval()
        q_t = self._t_to_tau(t.float().clamp(0.0, 1.0))
        model_t = self._task1_condition(q_t, t).to(self.device)
        teacher_state_embedding = self.teacher_model.vocab_embed(state)
        residual_logits = self.teacher_model(
            teacher_state_embedding,
            self._process_sigma(model_t),
            inputs_are_embeddings=True,
            x_self_cond=h)
        bias = flm_vocab_gaussian_bias(
            state, t,
            float(self.config.algo.tvm_teacher_bias_weight),
            self.flm_time_eps)
        return F.softmax(residual_logits.float() + bias.float(), dim=-1)

    def _map_teacher_local(self, state, t, h):
        """Return the terminal teacher used only by the text map loss."""
        return self._teacher_local(state, t, h)

    def _student_sc_outputs(
            self, state, r, eta, use_jvp_attn=False, model_r=None,
            h=None):
        if model_r is None:
            q_r = self._t_to_tau(r.float().clamp(0.0, 1.0))
            model_r = self._task1_condition(q_r, r).to(self.device)
        state_embedding = self._state_embedding(state)
        residual_logits, hidden = self.backbone(
            state_embedding,
            self._process_sigma(model_r),
            self._process_sigma(eta),
            use_jvp_attn=use_jvp_attn,
            inputs_are_embeddings=True,
            x_self_cond=h,
            return_hidden=True)
        with torch.amp.autocast(
                device_type=hidden.device.type, dtype=torch.bfloat16):
            sc_residual = self.backbone.sc_residual_head(hidden)
        bias = flm_vocab_gaussian_bias(
            state, r, self._current_token_bias_weight(), self.flm_time_eps)
        return (
            residual_logits.float() + bias.float(),
            sc_residual.float(),
            hidden)

    def _finite_sc_state(self, probabilities, h, residual, delta):
        proposal = torch.matmul(
            probabilities.float(), self.sc_coordinate.float())
        proposal = proposal + residual.float()
        return finite_sc_update(h.float(), proposal, delta, self.sc_d50)

    def finite_step(self, state, h, r, s):
        eta = ((s.float() - r.float())
               / (1.0 - r.float()).clamp_min(1e-8)).clamp(0.0, 1.0)
        logits, residual, hidden = self._student_sc_outputs(
            state, r, eta, use_jvp_attn=False, h=h)
        probabilities = F.softmax(logits.float(), dim=-1)
        endpoint = finite_map_update(state, probabilities, r, s)
        next_h = self._finite_sc_state(
            probabilities, h, residual, s.float() - r.float())
        return probabilities, endpoint, next_h, residual, hidden

    def _sample_training_intervals(self, batch_size, accumulation_step=None):
        """Keep the inherited 10k student's steady broad-map distribution."""
        device = self.device
        if accumulation_step is None:
            global_size = int(batch_size)
            r0_count = round(global_size * self.r0_probability)
            return sample_global_interval_batch(
                global_size, self.s_max, r0_count, 1.0, device,
                seed=int(self.config.seed) + int(self.global_step) * 1000003)

        global_size = int(self.config.loader.global_batch_size)
        total_chunks = (
            int(self.trainer.num_nodes)
            * int(self.trainer.num_devices)
            * int(self.trainer.accumulate_grad_batches))
        if global_size % total_chunks != 0:
            raise ValueError(
                'SC-repair global batch must divide evenly across devices '
                'and gradient-accumulation slices.')
        expected_batch = global_size // total_chunks
        if int(batch_size) != expected_batch:
            raise ValueError(
                f'SC repair expected local microbatch {expected_batch}, '
                f'got {batch_size}.')
        chunk_index = (
            int(self.trainer.node_rank)
            * int(self.trainer.num_devices)
            * int(self.trainer.accumulate_grad_batches)
            + int(self.trainer.local_rank)
            * int(self.trainer.accumulate_grad_batches)
            + int(accumulation_step))
        plan = sample_global_interval_batch(
            global_size, self.s_max,
            round(global_size * self.r0_probability), 1.0, device,
            seed=int(self.config.seed) + int(self.global_step) * 1000003)
        start = chunk_index * expected_batch
        stop = start + expected_batch
        r, s, diagonal, probabilities, mode, counts = plan
        return (r[start:stop], s[start:stop], diagonal[start:stop],
                probabilities, mode[start:stop], counts)

    def _sample_sc_pairs(self, count, accumulation_step):
        size = int(self._sc_pair_bank['r'].shape[0])
        generator = torch.Generator(device='cpu')
        generator.manual_seed(
            int(self.config.seed)
            + int(self.global_step) * 1000003
            + int(accumulation_step or 0) * 9176)
        indices = torch.randint(
            size, (int(count),), generator=generator, device='cpu')
        batch = {
            name: tensor.index_select(0, indices).to(
                self.device, dtype=torch.float32, non_blocking=True)
            for name, tensor in self._sc_pair_bank.items()
        }
        batch['r_ref'] = batch['r_ref'].clamp_min(
            torch.finfo(torch.float32).tiny)
        batch['r_ref'] = batch['r_ref'] / batch['r_ref'].sum(
            dim=-1, keepdim=True)
        return batch

    @torch.no_grad()
    def build_sc_reference_pair_bank(self, output_path):
        """Precompute matched-start continuousized teacher reference pairs."""
        trajectory_count = int(
            self.config.algo.tvm_sc_pair_bank_trajectories)
        pairs_per_trajectory = int(
            self.config.algo.tvm_sc_pairs_per_trajectory)
        resolution = int(self.config.algo.tvm_sc_reference_steps)
        reset_fraction = float(self.config.algo.tvm_sc_reset_fraction)
        short_fraction = float(
            self.config.algo.tvm_sc_short_gap_fraction)
        branch_batch_size = int(getattr(
            self.config.algo, 'tvm_sc_pair_bank_branch_batch_size', 4))
        if (trajectory_count <= 0 or pairs_per_trajectory <= 0
                or resolution <= 0 or branch_batch_size <= 0):
            raise ValueError('SC pair-bank sizes must be positive.')
        if not 0.0 <= reset_fraction < 0.5:
            raise ValueError(
                'Reset pairs must remain a minority of the SC-aware bank.')
        if not 0.0 <= short_fraction <= 1.0:
            raise ValueError('Short-gap fraction must lie in [0, 1].')
        if os.path.exists(output_path):
            raise FileExistsError(
                f'Refusing to overwrite SC pair bank: {output_path}')

        cpu_generator = torch.Generator(device='cpu')
        cpu_generator.manual_seed(int(self.config.seed) + 73001)
        pair_specs = []
        for trajectory_index in range(trajectory_count):
            for pair_index in range(pairs_per_trajectory):
                r_index = int(torch.randint(
                    1, resolution, (1,), generator=cpu_generator))
                r = self.s_max * r_index / resolution
                room = self.s_max - r
                is_short = bool(
                    torch.rand((), generator=cpu_generator) < short_fraction)
                if is_short:
                    log_ratio = (
                        math.log(1e-2)
                        + float(torch.rand((), generator=cpu_generator))
                        * (math.log(4.0) - math.log(1e-2)))
                    delta = min(room, self.sc_d50 * math.exp(log_ratio))
                else:
                    draw = float(torch.rand((), generator=cpu_generator))
                    if draw < 2.0 / 9.0:
                        z = 0.2 * float(torch.rand(
                            (), generator=cpu_generator))
                    elif draw < 5.0 / 9.0:
                        z = 0.2 + 0.4 * float(torch.rand(
                            (), generator=cpu_generator))
                    else:
                        z = 0.6 + 0.4 * float(torch.rand(
                            (), generator=cpu_generator))
                    delta = room * max(z, 1e-6)
                delta = max(min(delta, room), min(room, 1e-8))
                pair_specs.append({
                    'trajectory_index': trajectory_index,
                    'pair_index': pair_index,
                    'r_index': r_index,
                    'r': float(r),
                    's': float(r + delta),
                    'short_gap': is_short,
                    'reset_start': bool(
                        torch.rand((), generator=cpu_generator)
                        < reset_fraction),
                })

        state = torch.empty(
            trajectory_count, self.num_tokens, self.vocab_size,
            device=self.device, dtype=torch.float32)
        device_generator = torch.Generator(device=self.device)
        for trajectory_index in range(trajectory_count):
            device_generator.manual_seed(
                int(self.config.seed) + 81001 + trajectory_index)
            state[trajectory_index].normal_(generator=device_generator)
        h = torch.zeros(
            trajectory_count, self.num_tokens,
            int(self.config.model.hidden_size),
            device=self.device, dtype=torch.float32)
        starts = [None] * len(pair_specs)
        specs_by_start = {}
        for flat_index, spec in enumerate(pair_specs):
            specs_by_start.setdefault(spec['r_index'], []).append(flat_index)

        base_grid = torch.linspace(
            0.0, self.s_max, resolution + 1,
            device=self.device, dtype=torch.float32)
        for step in range(resolution):
            if step in specs_by_start:
                for flat_index in specs_by_start[step]:
                    trajectory_index = pair_specs[flat_index][
                        'trajectory_index']
                    starts[flat_index] = (
                        state[trajectory_index].to(
                            device='cpu', dtype=torch.bfloat16).clone(),
                        h[trajectory_index].to(
                            device='cpu', dtype=torch.bfloat16).clone())
            current = base_grid[step].expand(trajectory_count)
            following = base_grid[step + 1].expand(trajectory_count)
            probability = self._teacher_local(state, current, h)
            proposal = torch.matmul(
                probability.float(), self.sc_coordinate.float())
            h = finite_sc_update(
                h, proposal, following - current, self.sc_d50)
            state = finite_map_update(
                state, probability, current, following)
        if any(start is None for start in starts):
            raise RuntimeError('Failed to capture every pair-bank start state.')

        output = {
            'state_r': [],
            'h_r': [],
            'r': [],
            's': [],
            'r_ref': [],
            'h_t': [],
        }
        for chunk_start in range(0, len(pair_specs), branch_batch_size):
            chunk_specs = pair_specs[
                chunk_start:chunk_start + branch_batch_size]
            chunk_starts = starts[
                chunk_start:chunk_start + branch_batch_size]
            branch_state = torch.stack([
                item[0] for item in chunk_starts]).to(
                    self.device, dtype=torch.float32)
            branch_h = torch.stack([
                item[1] for item in chunk_starts]).to(
                    self.device, dtype=torch.float32)
            for local_index, spec in enumerate(chunk_specs):
                if spec['reset_start']:
                    branch_h[local_index].zero_()
            r = torch.tensor(
                [spec['r'] for spec in chunk_specs],
                device=self.device, dtype=torch.float32)
            s = torch.tensor(
                [spec['s'] for spec in chunk_specs],
                device=self.device, dtype=torch.float32)
            steps = torch.clamp(
                torch.ceil((s - r) * resolution).long(), min=1)
            equivalent = torch.zeros_like(branch_state)
            total_eta = torch.zeros(
                len(chunk_specs), device=self.device, dtype=torch.float32)
            maximum_steps = int(steps.max())
            for branch_step in range(maximum_steps):
                active = (branch_step < steps).nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    break
                fraction_left = (
                    branch_step / steps[active].float())
                fraction_right = (
                    (branch_step + 1) / steps[active].float())
                current = r[active] + (s[active] - r[active]) * fraction_left
                following = (
                    r[active] + (s[active] - r[active]) * fraction_right)
                probability = self._teacher_local(
                    branch_state[active], current, branch_h[active])
                local_eta = ((following - current)
                             / (1.0 - current).clamp_min(1e-8))
                new_total_eta = (
                    total_eta[active]
                    + (1.0 - total_eta[active]) * local_eta)
                numerator = (
                    ((1.0 - local_eta) * total_eta[active])[:, None, None]
                    * equivalent[active]
                    + local_eta[:, None, None] * probability)
                equivalent[active] = numerator / new_total_eta.clamp_min(
                    1e-8)[:, None, None]
                total_eta[active] = new_total_eta
                proposal = torch.matmul(
                    probability.float(), self.sc_coordinate.float())
                branch_h[active] = finite_sc_update(
                    branch_h[active], proposal,
                    following - current, self.sc_d50)
                branch_state[active] = finite_map_update(
                    branch_state[active], probability, current, following)

            output['state_r'].extend([item[0] for item in chunk_starts])
            output['h_r'].extend([
                (torch.zeros_like(item[1])
                 if spec['reset_start'] else item[1])
                for item, spec in zip(chunk_starts, chunk_specs)])
            output['r'].append(r.cpu())
            output['s'].append(s.cpu())
            output['r_ref'].extend([
                tensor.to(device='cpu', dtype=torch.bfloat16).clone()
                for tensor in equivalent])
            output['h_t'].extend([
                tensor.to(device='cpu', dtype=torch.bfloat16).clone()
                for tensor in branch_h])

        payload = {
            'state_r': torch.stack(output['state_r']),
            'h_r': torch.stack(output['h_r']),
            'r': torch.cat(output['r']),
            's': torch.cat(output['s']),
            'r_ref': torch.stack(output['r_ref']),
            'h_t': torch.stack(output['h_t']),
        }
        teacher_states = torch.cat(
            (payload['h_r'].float(), payload['h_t'].float()), dim=0)
        payload['state_variance'] = teacher_states.var(
            dim=(0, 1), unbiased=False)
        payload['metadata'] = {
            'schema_version': 1,
            'reference': 'continuousized_frozen_mse_teacher',
            'd50': self.sc_d50,
            'alpha': math.log(2.0) / self.sc_d50,
            'reference_steps_per_unit_interval': resolution,
            'trajectory_count': trajectory_count,
            'pairs_per_trajectory': pairs_per_trajectory,
            'pair_count': len(pair_specs),
            'reset_fraction_realized': (
                sum(spec['reset_start'] for spec in pair_specs)
                / len(pair_specs)),
            'short_gap_fraction_realized': (
                sum(spec['short_gap'] for spec in pair_specs)
                / len(pair_specs)),
            'seed': int(self.config.seed),
            'storage_dtype': 'bfloat16_dense_vocab_state',
            'pair_specs': pair_specs,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(payload, output_path)
        return payload['metadata']

    @torch.no_grad()
    def run_sc_diagnostics(self, output_path, sample_count=8,
                           batch_size=2):
        """Measure functional SC use and finite-map continuity on pair data."""
        size = min(int(sample_count), int(self._sc_pair_bank['r'].shape[0]))
        if size <= 0 or int(batch_size) <= 0:
            raise ValueError('SC diagnostics require positive sample counts.')
        tiny = torch.finfo(torch.float32).tiny
        functional_kl_sum = 0.0
        text_endpoint_kl_sum = 0.0
        normalized_state_error_sum = 0.0
        site_count = 0
        proposal_rows = []
        gap_functional_sums = {
            name: {'functional_kl': 0.0, 'state_error': 0.0, 'sites': 0}
            for name in ('lt_0p1', '0p1_to_1', '1_to_4', 'gt_4')}
        slope_sums = {ratio: 0.0 for ratio in (1e-2, 1e-1, 1.0, 4.0)}
        slope_counts = {ratio: 0 for ratio in slope_sums}
        insertion_sums = {
            ratio: {'text_l1': 0.0, 'state_distance': 0.0, 'count': 0}
            for ratio in (1e-2, 1e-1, 1.0)}

        for start in range(0, size, int(batch_size)):
            stop = min(start + int(batch_size), size)
            batch = {
                name: tensor[start:stop].to(
                    self.device, dtype=torch.float32)
                for name, tensor in self._sc_pair_bank.items()
            }
            reference = batch['r_ref'].clamp_min(tiny)
            reference = reference / reference.sum(dim=-1, keepdim=True)
            probability, _, predicted_h, residual, _ = self.finite_step(
                batch['state_r'], batch['h_r'], batch['r'], batch['s'])
            reference_endpoint = finite_map_update(
                batch['state_r'], reference, batch['r'], batch['s'])
            teacher_reference = self._teacher_local(
                reference_endpoint, batch['s'], batch['h_t'])
            teacher_prediction = self._teacher_local(
                reference_endpoint, batch['s'], predicted_h)
            functional_kl = (
                teacher_reference
                * (teacher_reference.clamp_min(tiny).log()
                   - teacher_prediction.clamp_min(tiny).log())).sum(dim=-1)
            text_endpoint_kl = (
                reference
                * (reference.log()
                   - probability.clamp_min(tiny).log())).sum(dim=-1)
            state_error = self._state_distance(
                predicted_h, batch['h_t'])
            functional_kl_sum += float(functional_kl.sum().cpu())
            text_endpoint_kl_sum += float(text_endpoint_kl.sum().cpu())
            normalized_state_error_sum += float(state_error.sum().cpu())
            site_count += int(functional_kl.numel())
            gap_ratio = (batch['s'] - batch['r']) / self.sc_d50
            gap_masks = {
                'lt_0p1': gap_ratio < 0.1,
                '0p1_to_1': (gap_ratio >= 0.1) & (gap_ratio < 1.0),
                '1_to_4': (gap_ratio >= 1.0) & (gap_ratio <= 4.0),
                'gt_4': gap_ratio > 4.0,
            }
            for name, mask in gap_masks.items():
                if not bool(mask.any()):
                    continue
                selected_kl = functional_kl[mask]
                selected_state = state_error[mask]
                gap_functional_sums[name]['functional_kl'] += float(
                    selected_kl.sum().cpu())
                gap_functional_sums[name]['state_error'] += float(
                    selected_state.sum().cpu())
                gap_functional_sums[name]['sites'] += int(
                    selected_kl.numel())

            proposal = (
                torch.matmul(probability.float(), self.sc_coordinate.float())
                + residual.float())
            gate = finite_sc_gate(
                batch['s'] - batch['r'], self.sc_d50).to(self.device)
            for local_index in range(stop - start):
                proposal_rows.append({
                    'pair_index': start + local_index,
                    'delta': float((batch['s'] - batch['r'])[
                        local_index].cpu()),
                    'gate': float(gate[local_index].cpu()),
                    'proposal_minus_h_rms': float((
                        proposal[local_index] - batch['h_r'][local_index]
                    ).square().mean().sqrt().cpu()),
                    'carried_update_rms': float((
                        predicted_h[local_index] - batch['h_r'][local_index]
                    ).square().mean().sqrt().cpu()),
                })

            for ratio in slope_sums:
                delta = torch.full_like(batch['r'], self.sc_d50 * ratio)
                delta = torch.minimum(
                    delta, (self.s_max - batch['r']).clamp_min(1e-8))
                valid = delta > 0.0
                if not bool(valid.any()):
                    continue
                _, _, short_h, _, _ = self.finite_step(
                    batch['state_r'][valid], batch['h_r'][valid],
                    batch['r'][valid], batch['r'][valid] + delta[valid])
                slope = (
                    (short_h - batch['h_r'][valid]).square()
                    .mean(dim=(1, 2)).sqrt() / delta[valid])
                slope_sums[ratio] += float(slope.sum().cpu())
                slope_counts[ratio] += int(slope.numel())

            direct_probability, direct_endpoint, direct_h, _, _ = (
                self.finite_step(
                    batch['state_r'], batch['h_r'],
                    batch['r'], batch['s']))
            del direct_probability
            for ratio, totals in insertion_sums.items():
                epsilon = torch.full_like(
                    batch['r'], self.sc_d50 * ratio)
                epsilon = torch.minimum(
                    epsilon, 0.5 * (batch['s'] - batch['r']))
                valid = epsilon > 0.0
                if not bool(valid.any()):
                    continue
                midpoint = batch['r'][valid] + epsilon[valid]
                _, first_endpoint, first_h, _, _ = self.finite_step(
                    batch['state_r'][valid], batch['h_r'][valid],
                    batch['r'][valid], midpoint)
                _, inserted_endpoint, inserted_h, _, _ = self.finite_step(
                    first_endpoint, first_h, midpoint, batch['s'][valid])
                totals['text_l1'] += float((
                    inserted_endpoint - direct_endpoint[valid]
                ).abs().mean(dim=(1, 2)).sum().cpu())
                totals['state_distance'] += float(
                    self._state_distance(
                        inserted_h, direct_h[valid]).mean(dim=1).sum().cpu())
                totals['count'] += int(valid.sum())

        result = {
            'schema_version': 1,
            'checkpoint_global_step': int(getattr(
                self, '_loaded_checkpoint_global_step', self.global_step)),
            'weights': ('online' if self.config.eval.disable_ema else 'ema'),
            'sample_count': size,
            'posterior_site_count': site_count,
            'gate': {
                'd50': self.sc_d50,
                'alpha': math.log(2.0) / self.sc_d50,
                'epsilon_floor': False,
            },
            'functional_kl_nats': functional_kl_sum / site_count,
            'd_use_nats': functional_kl_sum / site_count,
            'text_endpoint_kl_diag_nats': text_endpoint_kl_sum / site_count,
            'normalized_sc_state_mse': (
                normalized_state_error_sum / site_count),
            'functional_by_gap': [
                {
                    'gap_bin': name,
                    'functional_kl_nats': (
                        values['functional_kl'] / values['sites']),
                    'normalized_sc_state_mse': (
                        values['state_error'] / values['sites']),
                    'posterior_site_count': values['sites'],
                }
                for name, values in gap_functional_sums.items()
                if values['sites']
            ],
            'proposal_compensation': proposal_rows,
            'near_diagonal_slope': [
                {
                    'delta_over_d50': ratio,
                    'mean_norm_h_change_over_delta': (
                        slope_sums[ratio] / slope_counts[ratio]),
                    'sample_count': slope_counts[ratio],
                }
                for ratio in slope_sums if slope_counts[ratio]
            ],
            'inserted_short_interval_sensitivity': [
                {
                    'epsilon_over_d50': ratio,
                    'mean_text_state_l1': values['text_l1'] / values['count'],
                    'mean_sc_state_distance': (
                        values['state_distance'] / values['count']),
                    'sample_count': values['count'],
                }
                for ratio, values in insertion_sums.items()
                if values['count']
            ],
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as handle:
            import json
            json.dump(result, handle, indent=2)
            handle.write('\n')
        return result

    def _sc_loss_weights(self):
        step = float(self.global_step)
        return self.sc_pair_weight * min(step / 250.0, 1.0)

    def _state_distance(self, left, right):
        variance = self._sc_state_variance.to(
            left.device, dtype=torch.float32)
        denominator = (
            variance + 0.01 * self._sc_state_global_variance)
        return ((left.float() - right.float()).square()
                / denominator.view(1, 1, -1)).mean(dim=-1)

    def _log_sc_gap_gradient_proxies(
            self, sc_loss, hidden, delta):
        if (not torch.is_grad_enabled()
                or not sc_loss.requires_grad
                or not hidden.requires_grad):
            return
        ratio = delta.float() / self.sc_d50
        bins = (
            ('lt_0p1', ratio < 0.1),
            ('0p1_to_1', (ratio >= 0.1) & (ratio < 1.0)),
            ('1_to_4', (ratio >= 1.0) & (ratio <= 4.0)),
            ('gt_4', ratio > 4.0),
        )
        for name, mask in bins:
            if not bool(mask.any()):
                continue
            gradient = torch.autograd.grad(
                sc_loss[mask].mean(), hidden,
                retain_graph=True, allow_unused=True)[0]
            value = (
                hidden.new_zeros((), dtype=torch.float32)
                if gradient is None else gradient.float().norm(2))
            self.log(
                f'tvm_sc/sc_hidden_gradient_{name}', value.detach(),
                on_step=True, on_epoch=False, sync_dist=True)

    def _gradient_parameter_groups(self):
        dedicated = {
            'sc_input_projection': tuple(
                self.backbone.self_cond_proj.parameters()),
            'output_head': tuple(self.backbone.output_layer.parameters()),
            'sc_residual_head': tuple(
                self.backbone.sc_residual_head.parameters()),
            'eta_conditioning': tuple(
                self.backbone.sigma_map_prime.parameters()),
        }
        dedicated_ids = {
            id(parameter)
            for parameters in dedicated.values()
            for parameter in parameters}
        dedicated['shared_backbone'] = tuple(
            parameter for parameter in self.backbone.parameters()
            if id(parameter) not in dedicated_ids)
        return dedicated

    def _gradient_geometry(self, losses):
        groups = self._gradient_parameter_groups()
        parameters = tuple(
            parameter for parameter in self.backbone.parameters()
            if parameter.requires_grad)
        parameter_positions = {
            id(parameter): index
            for index, parameter in enumerate(parameters)}
        gradients = {}
        for loss_name, scalar_loss in losses.items():
            raw_gradients = torch.autograd.grad(
                scalar_loss, parameters, retain_graph=True,
                allow_unused=True)
            gradients[loss_name] = tuple(
                None if gradient is None else gradient.detach().float().cpu()
                for gradient in raw_gradients)
            del raw_gradients

        for group_name, group_parameters in groups.items():
            positions = [
                parameter_positions[id(parameter)]
                for parameter in group_parameters
                if id(parameter) in parameter_positions]
            group_vectors = {}
            for loss_name, loss_gradients in gradients.items():
                selected = {
                    position: loss_gradients[position].detach().float()
                    .reshape(-1)
                    for position in positions
                    if loss_gradients[position] is not None}
                if selected:
                    norm = torch.stack([
                        value.square().sum()
                        for value in selected.values()
                    ]).sum().sqrt()
                else:
                    norm = torch.zeros(())
                group_vectors[loss_name] = (selected, norm)
                self.log(
                    f'tvm_sc/gradient_probe/{group_name}/{loss_name}_norm',
                    norm.to(self.device), on_step=True, on_epoch=False,
                    sync_dist=True)

            names = tuple(group_vectors)
            for left_index, left_name in enumerate(names):
                for right_name in names[left_index + 1:]:
                    left_vectors, left_norm = group_vectors[left_name]
                    right_vectors, right_norm = group_vectors[right_name]
                    dot = torch.zeros(())
                    for position in left_vectors.keys() & right_vectors.keys():
                        dot = dot + (
                            left_vectors[position]
                            * right_vectors[position]).sum()
                    denominator = left_norm * right_norm
                    cosine = torch.where(
                        denominator > 0.0,
                        dot / denominator,
                        torch.zeros_like(dot))
                    self.log(
                        'tvm_sc/gradient_probe/'
                        f'{group_name}/{left_name}_{right_name}_cosine',
                        cosine.to(self.device), on_step=True, on_epoch=False,
                        sync_dist=True)

    def _gradient_probe_extra_losses(self):
        return {}

    def _run_gradient_conflict_probe(self):
        size = min(
            self.sc_gradient_probe_batch_size,
            int(self._sc_pair_bank['r'].shape[0]))
        batch = {
            name: tensor[:size].to(self.device, dtype=torch.float32)
            for name, tensor in self._sc_pair_bank.items()}
        eta = ((batch['s'] - batch['r'])
               / (1.0 - batch['r']).clamp_min(1e-8)).clamp(0.0, 1.0)
        q_r = self._t_to_tau(batch['r'].clamp(0.0, 1.0))
        model_r = self._task1_condition(q_r, batch['r']).to(
            self.device).detach()

        def outputs_at_eta(eta_value):
            return self._student_sc_outputs(
                batch['state_r'], batch['r'], eta_value,
                use_jvp_attn=True, model_r=model_r, h=batch['h_r'])

        outputs, tangents = torch.func.jvp(
            outputs_at_eta, (eta,), (eta * (1.0 - eta),))
        logits, residual, hidden = outputs
        probability, _, gate, _ = tvm_quantity_from_logits_jvp(
            logits, tangents[0])
        endpoint = finite_map_update(
            batch['state_r'], probability, batch['r'], batch['s'])
        endpoint_h = self._finite_sc_state(
            probability, batch['h_r'], residual,
            batch['s'] - batch['r'])
        with torch.no_grad():
            teacher = self._map_teacher_local(
                endpoint.detach(), batch['s'], endpoint_h.detach())
        student_log, calibrated_teacher = calibrated_relative_distributions(
            probability, gate, teacher, self.kappa)
        map_loss = -(
            calibrated_teacher * student_log).sum(dim=-1).mean()

        with torch.amp.autocast(
                device_type=hidden.device.type, dtype=torch.bfloat16):
            isolated_residual = self.backbone.sc_residual_head(hidden)
        isolated_h = finite_sc_regression_state(
            probability, self.sc_coordinate, isolated_residual.float(),
            batch['h_r'], batch['s'] - batch['r'], self.sc_d50)
        state_error = self._state_distance(isolated_h, batch['h_t'])
        sc_gate = finite_sc_gate(
            batch['s'] - batch['r'], self.sc_d50).to(self.device)
        h_loss = (
            state_error
            / (sc_gate[:, None].square()
               + self.sc_short_weight_floor ** 2)).mean()
        losses = {'map': map_loss, 'h': h_loss}

        losses.update(self._gradient_probe_extra_losses())
        self._gradient_geometry(losses)

    def loss(self, clean_tokens, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        run_gradient_probe = bool(train_mode)
        del output_tokens, xT, given_t, not_sampling_t
        batch_size, length = clean_tokens.shape
        trace_first = (
            isinstance(self, Task1TVMJointJ0)
            and int(self.global_step) == 0
            and int(current_accumulation_step or 0) == 0)
        r, s, diagonal, mode_probabilities, gap_mode, gap_counts = (
            self._sample_training_intervals(
                batch_size, current_accumulation_step))
        if trace_first:
            print('J0_MAP_PHASE intervals_sampled', flush=True)
        state = self.corrupt_vocab_state(clean_tokens, r)
        if trace_first:
            print('J0_MAP_PHASE state_corrupted', flush=True)
        h = torch.zeros(
            batch_size, length, int(self.config.model.hidden_size),
            device=self.device, dtype=torch.float32)

        sc_count = max(1, round(batch_size * self.sc_aware_fraction))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            int(self.config.seed) + int(self.global_step) * 65537
            + int(current_accumulation_step or 0) * 313)
        sc_index = torch.randperm(
            batch_size, generator=generator, device=self.device
        )[:sc_count].sort().values
        pair_batch = self._sample_sc_pairs(
            sc_count, current_accumulation_step)
        if trace_first:
            print('J0_MAP_PHASE pairs_sampled', flush=True)
        state[sc_index] = pair_batch['state_r']
        h[sc_index] = pair_batch['h_r']
        r[sc_index] = pair_batch['r']
        s[sc_index] = pair_batch['s']
        diagonal[sc_index] = False

        eta = ((s - r) / (1.0 - r).clamp_min(1e-8)).clamp(0.0, 1.0)
        loss = state.new_zeros((batch_size, length), dtype=torch.float32)

        if diagonal.any():
            if trace_first:
                print('J0_MAP_PHASE diagonal_start', flush=True)
            index = diagonal.nonzero(as_tuple=True)[0]
            logits, residual, _ = self._student_sc_outputs(
                state[index], r[index], eta[index], h=h[index])
            probabilities = F.softmax(logits.float(), dim=-1)
            endpoint = finite_map_update(
                state[index], probabilities, r[index], s[index])
            endpoint_h = self._finite_sc_state(
                probabilities, h[index], residual,
                s[index] - r[index])
            with torch.no_grad():
                teacher = self._map_teacher_local(
                    endpoint.detach(), s[index], endpoint_h.detach())
            loss[index] = -(teacher * F.log_softmax(
                logits.float(), dim=-1)).sum(dim=-1)
            if trace_first:
                print('J0_MAP_PHASE diagonal_complete', flush=True)

        off_diagonal = ~diagonal
        index = off_diagonal.nonzero(as_tuple=True)[0]
        state_off = state[index]
        h_off = h[index]
        r_off = r[index]
        s_off = s[index]
        eta_off = eta[index]
        q_r_off = self._t_to_tau(r_off.float().clamp(0.0, 1.0))
        model_r_off = self._task1_condition(q_r_off, r_off).to(
            self.device).detach()

        def outputs_at_eta(eta_value):
            return self._student_sc_outputs(
                state_off, r_off, eta_value, use_jvp_attn=True,
                model_r=model_r_off, h=h_off)

        if trace_first:
            print('J0_MAP_PHASE jvp_start', flush=True)
        outputs, tangents = torch.func.jvp(
            outputs_at_eta,
            (eta_off,),
            (eta_off * (1.0 - eta_off),))
        if trace_first:
            print('J0_MAP_PHASE jvp_complete', flush=True)
        logits, residual, hidden = outputs
        weighted_logit_jvp = tangents[0]
        probabilities, _, gate, quantity = tvm_quantity_from_logits_jvp(
            logits, weighted_logit_jvp)
        endpoint = finite_map_update(
            state_off, probabilities, r_off, s_off)
        endpoint_h = self._finite_sc_state(
            probabilities, h_off, residual, s_off - r_off)
        if trace_first:
            print('J0_MAP_PHASE teacher_start', flush=True)
        with torch.no_grad():
            teacher = self._map_teacher_local(
                endpoint.detach(), s_off, endpoint_h.detach()).detach()
        if trace_first:
            print('J0_MAP_PHASE teacher_complete', flush=True)
        student_log_probabilities, calibrated_teacher = (
            calibrated_relative_distributions(
                probabilities, gate, teacher, self.kappa))
        calibrated_ce = -(
            calibrated_teacher * student_log_probabilities).sum(dim=-1)
        raw_residual = quantity - teacher
        if self.main_loss_type == 'calibrated_relative_ce':
            main_loss = calibrated_ce
        else:
            main_loss = raw_residual.square().sum(dim=-1)
        negative_mass = F.relu(-quantity).sum(dim=-1)
        loss[index] = (
            main_loss
            + self.negative_mass_weight * negative_mass.square())

        sc_position = torch.searchsorted(index, sc_index)
        probability_sc = probabilities[sc_position]
        residual_sc = residual[sc_position]
        hidden_sc = hidden[sc_position]
        h_sc = h[sc_index]
        delta_sc = s[sc_index] - r[sc_index]
        with torch.amp.autocast(
                device_type=hidden_sc.device.type, dtype=torch.bfloat16):
            isolated_residual = self.backbone.sc_residual_head(hidden_sc)
        isolated_h = finite_sc_regression_state(
            probability_sc, self.sc_coordinate, isolated_residual.float(),
            h_sc, delta_sc, self.sc_d50)
        state_error = self._state_distance(
            isolated_h, pair_batch['h_t'])
        if trace_first:
            print('J0_MAP_PHASE h_complete', flush=True)
        sc_gate = finite_sc_gate(delta_sc, self.sc_d50).to(self.device)
        short_weight = 1.0 / (
            sc_gate.square() + self.sc_short_weight_floor ** 2)
        weighted_state_error = short_weight[:, None] * state_error
        reference = pair_batch['r_ref']
        reference_kl = (
            reference
            * (reference.log()
               - probability_sc.clamp_min(
                   torch.finfo(torch.float32).tiny).log())).sum(dim=-1)
        h_loss = weighted_state_error
        pair_weight = self._sc_loss_weights()
        independent_mean_scale = batch_size / sc_count
        loss[sc_index] += (
            pair_weight * independent_mean_scale * h_loss)

        if pair_weight > 0.0:
            self._log_sc_gap_gradient_proxies(
                weighted_state_error, hidden_sc, delta_sc)

        self.log(
            'loss', loss.detach().mean(), prog_bar=True,
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm/calibrated_ce', calibrated_ce.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm/negative_mass', negative_mass.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm_sc/h_lambda', loss.new_tensor(pair_weight),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm_sc/state_loss', weighted_state_error.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm_sc/reference_kl_diagnostic', reference_kl.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm_sc/gate_mean', sc_gate.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'tvm_sc/aware_fraction', loss.new_tensor(sc_count / batch_size),
            on_step=True, on_epoch=False, sync_dist=True)
        for mode_index, name in enumerate(
                ('diagonal', 'short', 'medium', 'long')):
            self.log(
                f'tvm/gap_mode_probability_{name}',
                mode_probabilities[mode_index],
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                f'tvm/gap_mode_global_quota_{name}',
                loss.new_tensor(float(gap_counts[mode_index])),
                on_step=True, on_epoch=False, sync_dist=False)
        if (run_gradient_probe
                and self.sc_gradient_probe_inline
                and int(self.global_step) % self.sc_gradient_probe_interval == 0
                and int(current_accumulation_step or 0) == 0):
            self._run_gradient_conflict_probe()
        return loss

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        for label, module in (
                ('sc_input_projection', self.backbone.self_cond_proj),
                ('sc_residual_head', self.backbone.sc_residual_head)):
            gradients = [
                parameter.grad.detach().float().norm(2)
                for parameter in module.parameters()
                if parameter.grad is not None]
            total = (
                torch.stack(gradients).norm(2) if gradients
                else torch.zeros((), device=self.device))
            self.log(
                f'tvm_sc/{label}_grad_norm', total,
                on_step=True, on_epoch=False, sync_dist=True)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        grid = tuple(float(value) for value in getattr(
            self.config.algo, 'tvm_inference_grid_physical', []))
        if len(grid) < 2:
            raise ValueError(
                'SC-repair evaluation requires an explicit physical-t grid.')
        if (not math.isclose(grid[0], 0.0, abs_tol=1e-8)
                or not math.isclose(grid[-1], 1.0, abs_tol=1e-8)
                or any(left >= right
                       for left, right in zip(grid, grid[1:]))):
            raise ValueError(
                'SC-repair inference grid must increase from 0 to 1.')
        expected_steps = len(grid) - 1
        if num_steps is not None and int(num_steps) != expected_steps:
            raise ValueError(
                f'SC-repair grid has {expected_steps} intervals, '
                f'but num_steps={num_steps}.')
        state_shape = (
            num_samples, self.num_tokens, self.vocab_size)
        initial_noise_seed = getattr(
            self.config.sampling, 'task1_initial_noise_seed', None)
        initial_noise_schedule = str(getattr(
            self.config.sampling, 'task1_initial_noise_schedule',
            'base_seed_plus_batch_index'))
        if (initial_noise_seed is not None
                and initial_noise_schedule == 'base_seed_plus_sample_index'):
            offset = int(getattr(
                self, '_task1_sampling_sample_offset', 0))
            state = torch.empty(
                state_shape, device=self.device, dtype=torch.float32)
            generator = torch.Generator(device=self.device)
            for local_index in range(num_samples):
                generator.manual_seed(
                    int(initial_noise_seed) + offset + local_index)
                state[local_index].normal_(generator=generator)
            self._task1_sampling_sample_offset = offset + num_samples
        else:
            state = torch.randn(
                state_shape, device=self.device, dtype=torch.float32)
        h = torch.zeros(
            num_samples, self.num_tokens, int(self.config.model.hidden_size),
            device=self.device, dtype=torch.float32)
        for left, right in zip(grid, grid[1:]):
            r = torch.full(
                (num_samples,), left, device=self.device,
                dtype=torch.float32)
            s = torch.full_like(r, right)
            _, state, h, _, _ = self.finite_step(state, h, r, s)
        self.last_sampling_nfe = expected_steps
        return state.argmax(dim=-1)


class Task1TVMJointJ0(Task1TVMSCRepair):
    """Frozen-teacher J0 with an independently reduced local clean-token CE."""

    _ENDPOINT_SC_MODES = {'zero_sc', 'predicted_finite_state'}

    _LOCAL_GROUP_NAMES = (
        'tau_zero',
        'uniform_tau_first_interval',
        'uniform_physical_t_first_interval',
        'uniform_tau_high_noise_remainder',
        'uniform_tau_0p1_0p4',
        'uniform_tau_0p4_0p7',
        'uniform_tau_0p7_1p0',
    )

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.local_weight = float(config.algo.tvm_local_weight)
        self.local_warmup_steps = int(
            config.algo.tvm_local_warmup_steps)
        self.local_sc_probability = float(
            config.algo.tvm_local_sc_probability)
        self.local_time_sampling = str(
            config.algo.tvm_local_time_sampling)
        self.local_validation_global_size = int(
            config.algo.tvm_local_validation_global_size)
        length = int(config.model.length)
        probe_size = int(self.sc_gradient_probe_batch_size)
        self.register_buffer(
            'joint_probe_tokens',
            torch.zeros(probe_size, length, dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'joint_probe_valid_tokens',
            torch.zeros(probe_size, length, dtype=torch.bool),
            persistent=True)
        self.register_buffer(
            'joint_probe_t',
            torch.zeros(probe_size, dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'joint_probe_ready', torch.tensor(False), persistent=True)
        self._joint_validation_batch_index = 0
        self._validate_joint_configuration()

    def _validate_joint_configuration(self):
        if self.local_time_sampling != 'v1_m_tau25_global256':
            raise ValueError('J0 local CE requires M-tau25 global sampling.')
        if self.local_weight < 0.0 or not self._local_warmup_is_valid():
            raise ValueError('J0 local CE weight and warm-in must be valid.')
        if not math.isclose(
                self.local_sc_probability, 0.25,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('J0 local-SC probability must remain 0.25.')
        if self.local_validation_global_size != 256:
            raise ValueError('J0 held-out local diagnostics require 256 samples.')
        if (str(self.config.mode) == 'train'
                and int(self.config.loader.global_batch_size) != 256):
            raise ValueError('J0 training requires global batch 256.')

    def _local_warmup_is_valid(self):
        return self.local_warmup_steps > 0

    def _load_student_ema_initialization(self, path):
        if not path or not os.path.isfile(path):
            raise ValueError(
                f'J0 requires the SC-repair step500 checkpoint: {path!r}.')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(self.backbone, checkpoint)
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        allowed_missing = {'rotary_emb.inv_freq'}
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'Unexpected step500 EMA initialization mismatch: '
                f'missing={sorted(unexpected_missing)}, '
                f'unexpected={sorted(incompatible.unexpected_keys)}.')

    def _process_model_input(self, clean_tokens, valid_tokens):
        return clean_tokens, valid_tokens, valid_tokens

    def _sc_loss_weights(self):
        return self.sc_pair_weight

    def _local_lambda(self):
        progress = min(
            float(self.global_step) / float(self.local_warmup_steps), 1.0)
        return self.local_weight * progress

    def _sample_training_local_times(
            self, batch_size, current_accumulation_step):
        if self.training_time_sampling != self.local_time_sampling:
            raise ValueError(
                'J0 training_time_sampling must match tvm_local_time_sampling.')
        return self._sample_task1_training_times(
            batch_size, current_accumulation_step)

    def _validation_local_plan(self, batch_size, batch_index):
        from task1_continuation import global_group_counts

        global_size = self.local_validation_global_size
        if global_size % int(batch_size) != 0:
            raise ValueError(
                'Held-out local diagnostic batch must divide 256 exactly.')
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.config.seed) + 910003)
        base = torch.tensor([
            group
            for group, count in enumerate(
                global_group_counts(self.local_time_sampling))
            for _ in range(count)
        ], dtype=torch.long, device=self.device)
        group_id = base[torch.randperm(
            global_size, generator=generator, device=self.device)]
        random = torch.rand(
            global_size, generator=generator,
            dtype=torch.float32, device=self.device)
        start = (int(batch_index) * int(batch_size)) % global_size
        indices = torch.arange(
            start, start + int(batch_size), device=self.device) % global_size
        group_id = group_id.index_select(0, indices)
        random = random.index_select(0, indices)
        return self._local_times_from_groups(group_id, random)

    def _local_times_from_groups(self, group_id, random):
        tau = torch.empty_like(random)
        physical_t = torch.empty_like(random)
        first_tau = 1.0 / 128.0
        first_physical_t = tau.new_tensor(
            self._task1_lut_anchor_physical_t[1])
        tau[group_id == 0] = 0.0
        mask = group_id == 1
        tau[mask] = random[mask] * first_tau
        physical_mask = group_id == 2
        physical_t[physical_mask] = random[physical_mask] * first_physical_t
        tau[physical_mask] = self._t_to_tau(
            physical_t[physical_mask]).float().clamp(0.0, first_tau)
        mask = group_id == 3
        tau[mask] = first_tau + random[mask] * (0.1 - first_tau)
        for group, (lower, upper) in {
                4: (0.1, 0.4), 5: (0.4, 0.7),
                6: (0.7, 1.0)}.items():
            mask = group_id == group
            tau[mask] = lower + random[mask] * (upper - lower)
        mapped = self._task1_physical_time(tau)
        physical_t[~physical_mask] = mapped[~physical_mask]
        physical_t[group_id == 0] = 0.0
        return tau, physical_t, group_id

    def _local_state(self, clean_tokens, t, noise=None):
        if noise is None:
            noise = torch.randn(
                (clean_tokens.shape[0], clean_tokens.shape[1],
                 self.vocab_size),
                dtype=torch.float32, device=self.device,
                generator=self._task1_training_generator('gaussian_noise'))
        return self.corrupt_vocab_state(clean_tokens, t, noise=noise)

    def _local_logits(self, state, t, sc_sequence_mask):
        batch_size, length = state.shape[:2]
        h = torch.zeros(
            batch_size, length, int(self.config.model.hidden_size),
            dtype=torch.float32, device=self.device)
        if bool(sc_sequence_mask.any()):
            index = sc_sequence_mask.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                initial_logits, _, _ = self._student_sc_outputs(
                    state[index], t[index], torch.zeros_like(t[index]),
                    h=h[index])
                initial_probability = F.softmax(
                    initial_logits.float(), dim=-1)
                h[index] = torch.matmul(
                    initial_probability,
                    self.sc_coordinate.float()).float().detach()
        logits, _, _ = self._student_sc_outputs(
            state, t, torch.zeros_like(t), h=h)
        return logits.float()

    def _local_branch(
            self, clean_tokens, valid_tokens, t, group_id,
            sc_sequence_mask, noise=None, log_prefix='joint/local'):
        state = self._local_state(clean_tokens, t, noise=noise)
        logits = self._local_logits(state, t, sc_sequence_mask)
        token_ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            clean_tokens.reshape(-1), reduction='none').reshape_as(
                clean_tokens).float()
        mask = valid_tokens.to(token_ce.dtype)
        mean_ce = (token_ce * mask).sum() / mask.sum().clamp_min(1.0)
        probability = F.softmax(logits, dim=-1)
        entropy = -(probability * probability.clamp_min(
            torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
        accuracy = (logits.argmax(dim=-1) == clean_tokens).float()
        self.log(
            f'{log_prefix}_ce', mean_ce.detach(), on_step=True,
            on_epoch=not self.training, sync_dist=True)
        self.log(
            f'{log_prefix}_posterior_entropy',
            (entropy * mask).sum().div(mask.sum().clamp_min(1.0)).detach(),
            on_step=True, on_epoch=not self.training, sync_dist=True)
        self.log(
            f'{log_prefix}_top1_accuracy',
            (accuracy * mask).sum().div(mask.sum().clamp_min(1.0)).detach(),
            on_step=True, on_epoch=not self.training, sync_dist=True)
        for group, name in enumerate(self._LOCAL_GROUP_NAMES):
            selected = group_id == group
            if not bool(selected.any()):
                continue
            selected_mask = mask[selected]
            selected_ce = token_ce[selected]
            value = (
                (selected_ce * selected_mask).sum()
                / selected_mask.sum().clamp_min(1.0))
            self.log(
                f'{log_prefix}_ce_by_time/{name}', value.detach(),
                on_step=True, on_epoch=not self.training, sync_dist=True)
        return token_ce, mean_ce

    def _remember_joint_probe(self, clean_tokens, valid_tokens, t):
        if bool(self.joint_probe_ready):
            return
        size = self.joint_probe_tokens.shape[0]
        if clean_tokens.shape[0] < size:
            raise ValueError('J0 microbatch is smaller than the probe batch.')
        with torch.no_grad():
            self.joint_probe_tokens.copy_(clean_tokens[:size])
            self.joint_probe_valid_tokens.copy_(valid_tokens[:size].bool())
            self.joint_probe_t.copy_(t[:size].float())
            self.joint_probe_ready.fill_(True)

    def _gradient_probe_extra_losses(self):
        if not bool(self.joint_probe_ready):
            raise RuntimeError('J0 local gradient probe was not initialized.')
        size = self.joint_probe_tokens.shape[0]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.config.seed) + 920003)
        noise = torch.randn(
            (size, self.joint_probe_tokens.shape[1], self.vocab_size),
            dtype=torch.float32, device=self.device, generator=generator)
        sc_count = int(round(size * self.local_sc_probability))
        sc_mask = torch.zeros(size, dtype=torch.bool, device=self.device)
        if sc_count:
            sc_mask[-sc_count:] = True
        _, local = self._local_branch(
            self.joint_probe_tokens, self.joint_probe_valid_tokens,
            self.joint_probe_t, torch.full(
                (size,), -1, dtype=torch.long, device=self.device),
            sc_mask, noise=noise, log_prefix='joint/probe_local')
        return {
            'local': local,
            'weighted_local': self._local_lambda() * local,
        }

    def on_validation_epoch_start(self):
        self._joint_validation_batch_index = 0
        super().on_validation_epoch_start()

    def loss(self, clean_tokens, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del xT, given_t, not_sampling_t
        if output_tokens is None:
            raise ValueError('J0 local CE requires the valid-token mask.')
        valid_tokens = output_tokens.bool()
        batch_size = clean_tokens.shape[0]

        if not train_mode:
            _, t, group_id = self._validation_local_plan(
                batch_size, self._joint_validation_batch_index)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(
                int(self.config.seed) + 930003
                + self._joint_validation_batch_index)
            noise = torch.randn(
                (batch_size, clean_tokens.shape[1], self.vocab_size),
                dtype=torch.float32, device=self.device,
                generator=generator)
            state = self._local_state(clean_tokens, t, noise=noise)
            zero_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device)
            all_sc_mask = torch.ones_like(zero_mask)
            zero_logits = self._local_logits(state, t, zero_mask)
            sc_logits = self._local_logits(state, t, all_sc_mask)
            zero_ce = F.cross_entropy(
                zero_logits.reshape(-1, zero_logits.shape[-1]),
                clean_tokens.reshape(-1), reduction='none').reshape_as(
                    clean_tokens).float()
            sc_ce = F.cross_entropy(
                sc_logits.reshape(-1, sc_logits.shape[-1]),
                clean_tokens.reshape(-1), reduction='none').reshape_as(
                    clean_tokens).float()
            sc_count = int(round(batch_size * self.local_sc_probability))
            mixed = zero_ce.clone()
            mixed_logits = zero_logits.clone()
            if sc_count:
                mixed[-sc_count:] = sc_ce[-sc_count:]
                mixed_logits[-sc_count:] = sc_logits[-sc_count:]
            mask = valid_tokens.to(mixed.dtype)
            denominator = mask.sum().clamp_min(1.0)
            for name, values in (
                    ('zero_sc', zero_ce), ('same_time_sc', sc_ce),
                    ('mixed_75_25', mixed)):
                self.log(
                    f'joint/heldout_local_ce/{name}',
                    ((values * mask).sum() / denominator).detach(),
                    on_step=False, on_epoch=True, sync_dist=True,
                    batch_size=batch_size)
            probability = F.softmax(mixed_logits, dim=-1)
            entropy = -(probability * probability.clamp_min(
                torch.finfo(torch.float32).tiny).log()).sum(dim=-1)
            accuracy = (probability.argmax(dim=-1) == clean_tokens).float()
            self.log(
                'joint/heldout_local_posterior_entropy',
                ((entropy * mask).sum() / denominator).detach(),
                on_step=False, on_epoch=True, sync_dist=True,
                batch_size=batch_size)
            self.log(
                'joint/heldout_local_top1_accuracy',
                ((accuracy * mask).sum() / denominator).detach(),
                on_step=False, on_epoch=True, sync_dist=True,
                batch_size=batch_size)
            for group, name in enumerate(self._LOCAL_GROUP_NAMES):
                selected = group_id == group
                if not bool(selected.any()):
                    continue
                selected_mask = mask[selected]
                self.log(
                    f'joint/heldout_local_ce_by_time/{name}',
                    ((mixed[selected] * selected_mask).sum()
                     / selected_mask.sum().clamp_min(1.0)).detach(),
                    on_step=False, on_epoch=True, sync_dist=True,
                    batch_size=int(selected.sum()))
            self._joint_validation_batch_index += 1
            return mixed

        _, t, group_id = self._sample_training_local_times(
            batch_size, current_accumulation_step)
        trace_first = (
            int(self.global_step) == 0
            and int(current_accumulation_step or 0) == 0)
        if trace_first:
            print('J0_PHASE local_branch_start', flush=True)
        self._remember_joint_probe(clean_tokens, valid_tokens, t)
        sc_count = int(round(batch_size * self.local_sc_probability))
        sc_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device)
        if sc_count:
            generator = self._task1_training_generator('self_conditioning')
            selected = torch.randperm(
                batch_size, generator=generator, device=self.device)[:sc_count]
            sc_mask[selected] = True
        _, local_ce = self._local_branch(
            clean_tokens, valid_tokens, t, group_id, sc_mask)
        if trace_first:
            print('J0_PHASE local_branch_complete', flush=True)
        local_lambda = self._local_lambda()
        weighted_local = local_lambda * local_ce
        if trace_first:
            print('J0_PHASE map_h_branch_start', flush=True)
        map_and_h = super().loss(
            clean_tokens, None, current_accumulation_step,
            train_mode=True)
        if trace_first:
            print('J0_PHASE map_h_branch_complete', flush=True)
        total = map_and_h + weighted_local
        self.log(
            'joint/local_lambda', map_and_h.new_tensor(local_lambda),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'joint/combined_loss', total.detach().mean(),
            on_step=True, on_epoch=False, sync_dist=True)
        return total
