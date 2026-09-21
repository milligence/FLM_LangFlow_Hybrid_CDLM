"""Model and Lightning core for the LangFlow-FLM hybrid."""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from task1_continuation import global_group_counts

from .ops import (
    candidate_bias_codebook,
    detached_self_conditioning_embedding,
    flm_corrupt_vocab,
    flm_linear_alpha_sigma,
    flm_linear_gamma,
    flm_model_time_condition,
    flm_vocab_gaussian_bias,
    langflow_alpha_sigma,
    langflow_corrupt_embedding,
    langflow_gumbel_gamma,
    langflow_token_bias_weight,
    probability_prediction_metrics,
)


class LangFlowModelMixin:
    """Configuration, embeddings, forward path, and paired CE/MSE loss."""

    def __init__(self, config, tokenizer):
        # GPT-2's temporary dataloader-only PAD token is never a clean target
        # in packed OWT-128 batches and is not part of the frozen scorer vocab.
        super().__init__(config, tokenizer, vocab_size=tokenizer.vocab_size)
        self.loss_type = str(config.algo.loss_type)
        self.state_space = str(getattr(
            config.algo, 'state_space', 'embedding'))
        self.corruption = str(getattr(
            config.algo, 'corruption', 'langflow_vp_gaussian'))
        self.flm_time_eps = float(getattr(
            config.algo, 'flm_time_eps', 1e-5))
        self.model_time_condition = str(getattr(
            config.algo, 'model_time_condition', 'gumbel_log_nsr'))
        self.training_time_sampling = str(getattr(
            config.algo, 'training_time_sampling', 'uniform_tau'))
        gumbel = getattr(config.algo, 'gumbel', None)
        self.gumbel_loc = float(getattr(gumbel, 'loc', 4.723))
        self.gumbel_scale = float(getattr(gumbel, 'scale', 0.852))
        self.gumbel_cutoff = float(getattr(gumbel, 'cutoff', 1e-5))
        self.self_condition_probability = float(
            config.algo.self_condition_probability)
        self.token_bias_warmup_steps = int(
            config.algo.token_bias_warmup_steps)
        self.token_bias_schedule = str(getattr(
            config.algo, 'token_bias_schedule', 'warmup'))
        self.classification_prototype_mode = str(getattr(
            config.algo, 'classification_prototype_mode', 'shared'))
        self.diagnostic_interval_steps = int(getattr(
            config.algo, 'diagnostic_interval_steps', 2000))
        self.gradient_route_interval_steps = int(getattr(
            config.algo, 'gradient_route_interval_steps', 25))
        self.optimization_diagnostic_interval_steps = int(getattr(
            config.algo, 'optimization_diagnostic_interval_steps', 10))
        self.codebook_gradient_mode = str(getattr(
            config.algo, 'codebook_gradient_mode', 'all'))
        if (self.codebook_gradient_mode == 'frozen'
                and self.backbone.vocab_embed.embedding.requires_grad):
            raise RuntimeError(
                'Frozen physical codebook must be configured before EMA setup.')
        geometry_count = min(4096, self.vocab_size)
        geometry_generator = torch.Generator(device='cpu')
        geometry_generator.manual_seed(0)
        geometry_indices = torch.randperm(
            self.vocab_size, generator=geometry_generator)[:geometry_count]
        initial_geometry = F.normalize(
            self.backbone.vocab_embed.embedding.detach().float()[
                geometry_indices], dim=-1)
        self.register_buffer(
            '_geometry_sample_indices', geometry_indices, persistent=False)
        self.register_buffer(
            '_initial_geometry_unit_vectors', initial_geometry,
            persistent=False)
        self.register_buffer(
            '_training_token_counts',
            torch.zeros(self.vocab_size, dtype=torch.long), persistent=False)
        self.posterior_bucket_diagnostics = {}
        self.bias_logit_diagnostics = {}
        self.prototype_diagnostics = {}
        self.gradient_route_diagnostics = {}
        self.optimization_diagnostics = {}
        self._validation_posterior_rows = None
        self._validation_logit_stats = None
        self._route_candidate_weight_grad = None
        self._route_candidate_raw_snapshot = None
        self._route_physical_raw_snapshot = None
        self._route_logit_stats = None
        self._route_step = None
        self.last_sampling_nfe = 0
        self.embedding_diagnostics = {}
        self.gamma_bin_diagnostics = {}
        self._validation_gamma_bins = None
        self._task1_last_shapes = None
        self._task1_training_trace_written = False
        self.task1_training_time_diagnostics = {}
        self._task1_training_generators = {}
        self._task1_pending_training_rng_state = None
        self._task1_cached_global_time_plan = None
        for name in (
                '_task1_nonfinite_loss_events',
                '_task1_nonfinite_loss_values',
                '_task1_nonfinite_gradient_events',
                '_task1_nonfinite_gradient_tensors',
                '_task1_gradient_clip_events'):
            self.register_buffer(
                name, torch.zeros((), dtype=torch.long), persistent=False)
        self._validate_configuration()
        self._task1_lut_anchor_physical_t = None
        if self.state_space == 'vocab':
            anchor_tau = torch.tensor([0.0, 1.0 / 128.0, 0.1])
            self._task1_lut_anchor_physical_t = (
                self._task1_physical_time(anchor_tau).cpu().tolist())

    def _validate_configuration(self):
        if self.config.algo.name != 'langflow_flm_hybrid':
            raise ValueError('LangFlowFLMHybrid requires its dedicated config.')
        if self.config.algo.backbone != 'dit':
            raise ValueError('The hybrid baseline requires the DIT backbone.')
        if self.loss_type not in {
                'cross_entropy', 'softmax_probability_mse'}:
            raise ValueError(f'Unsupported hybrid loss {self.loss_type!r}.')
        if self.state_space not in {'embedding', 'vocab'}:
            raise ValueError(
                'state_space must be embedding or vocab, got '
                f'{self.state_space!r}.')
        if (self.state_space == 'vocab'
                and self.loss_type != 'softmax_probability_mse'):
            raise ValueError(
                'The Task1 vocabulary-state route requires '
                'softmax_probability_mse.')
        embedding_state = bool(self.config.algo.embedding_state)
        if embedding_state != (self.state_space == 'embedding'):
            raise ValueError(
                'embedding_state must be true exactly for state_space=embedding.')
        expected_corruption = (
            'langflow_vp_gaussian'
            if self.state_space == 'embedding'
            else 'flm_linear_gaussian')
        if self.corruption != expected_corruption:
            raise ValueError(
                f'{self.state_space} state requires corruption='
                f'{expected_corruption}, got {self.corruption!r}.')
        normalize_embeddings = bool(self.config.algo.normalize_embeddings)
        if self.state_space == 'embedding' and not normalize_embeddings:
            raise ValueError(
                'The hybrid baseline requires normalize_embeddings=true.')
        if self.state_space == 'vocab' and normalize_embeddings:
            raise ValueError(
                'Task1 requires an unconstrained learned V-to-hidden input '
                'projection, so normalize_embeddings must be false.')
        if not bool(self.config.algo.self_conditioning):
            raise ValueError(
                'The hybrid baseline requires self_conditioning=true.')
        if self.token_bias_schedule not in {'warmup', 'none', 'full'}:
            raise ValueError(
                'token_bias_schedule must be warmup, none, or full, got '
                f'{self.token_bias_schedule!r}.')
        tokenwise_bias = bool(self.config.algo.tokenwise_bias)
        if tokenwise_bias != (self.token_bias_schedule != 'none'):
            raise ValueError(
                'tokenwise_bias must be false exactly when '
                'token_bias_schedule=none.')
        if self.classification_prototype_mode not in {
                'shared', 'independent', 'direct_vocab_state'}:
            raise ValueError(
                'classification_prototype_mode must be shared, independent, '
                'or direct_vocab_state.')
        if (self.state_space == 'vocab'
                and self.classification_prototype_mode != 'direct_vocab_state'):
            raise ValueError(
                'Task1 Gaussian bias must use the direct vocabulary state.')
        if self.state_space == 'vocab':
            if self.model_time_condition not in {'tau', 'log_nsr'}:
                raise ValueError(
                    'Task1 model_time_condition must be tau or log_nsr.')
            if str(getattr(
                    self.config.algo, 'time_sampling', '')) != (
                        'flm_decoding_error_rate_tau_inverse'):
                raise ValueError(
                    'Task1 requires FLM decoding-error-rate tau inverse '
                    'time sampling.')
            if str(getattr(
                    self.config.algo, 'optimization_scale', '')) != 'one_half':
                raise ValueError(
                    'Task1 vocab-summed Brier loss requires one_half scale.')
            if bool(getattr(
                    self.config.model, 'tie_word_embeddings', False)):
                raise ValueError(
                    'Task1 input projection and output classifier must be '
                    'independent.')
            if self.training_time_sampling not in {
                    'uniform_tau', 'high_noise_quota32',
                    'v1_staged_quota32', 'v1_q30_frozen_global256',
                    'v1_m_tau25_global256'}:
                raise ValueError(
                    'Task1 training_time_sampling must be uniform_tau, '
                    'high_noise_quota32, v1_staged_quota32, '
                    'v1_q30_frozen_global256, or v1_m_tau25_global256.')
            if self.training_time_sampling in {
                    'high_noise_quota32', 'v1_staged_quota32',
                    'v1_q30_frozen_global256',
                    'v1_m_tau25_global256'}:
                global_batch_size = int(
                    self.config.loader.global_batch_size)
                micro_batch_size = int(self.config.loader.batch_size)
                accumulation = int(
                    self.config.trainer.accumulate_grad_batches)
                if global_batch_size != 256:
                    raise ValueError(
                        'Task1 quota sampling requires '
                        'loader.global_batch_size=256.')
                allowed_layouts = {32: 8, 64: 4, 128: 2}
                if (micro_batch_size not in allowed_layouts
                        or accumulation != allowed_layouts[micro_batch_size]):
                    raise ValueError(
                        'Task1 quota sampling requires H800 '
                        'micro/accumulation '
                        '32/8, 64/4, or 128/2.')
                if (int(self.config.trainer.devices) != 1
                        or int(self.config.trainer.num_nodes) != 1):
                    raise ValueError(
                        'Task1 quota sampling requires exactly one '
                        'training device and one node.')
        elif self.model_time_condition != 'gumbel_log_nsr':
            raise ValueError(
                'Embedding-state hybrids require gumbel_log_nsr conditioning.')
        if not 0.0 <= self.self_condition_probability <= 1.0:
            raise ValueError('self_condition_probability must lie in [0, 1].')
        if self.gumbel_scale <= 0.0:
            raise ValueError('The frozen Gumbel scale must be positive.')
        if not 0.0 < self.gumbel_cutoff < 0.5:
            raise ValueError('The Gumbel cutoff must lie in (0, 0.5).')
        if not 0.0 < self.flm_time_eps < 0.5:
            raise ValueError('flm_time_eps must lie in (0, 0.5).')
        if self.token_bias_warmup_steps < 0:
            raise ValueError('token_bias_warmup_steps cannot be negative.')
        if self.diagnostic_interval_steps <= 0:
            raise ValueError('diagnostic_interval_steps must be positive.')
        if self.gradient_route_interval_steps <= 0:
            raise ValueError(
                'gradient_route_interval_steps must be positive.')
        if self.optimization_diagnostic_interval_steps <= 0:
            raise ValueError(
                'optimization_diagnostic_interval_steps must be positive.')
        if self.codebook_gradient_mode not in {
                'all', 'detach_candidate', 'frozen'}:
            raise ValueError(
                'codebook_gradient_mode must be all, detach_candidate, or '
                f'frozen, got {self.codebook_gradient_mode!r}.')
        if (self.codebook_gradient_mode == 'detach_candidate'
                and self.loss_type != 'softmax_probability_mse'):
            raise ValueError('D-bias is registered only for probability MSE.')
        if bool(getattr(self.config.algo, 'learnable_loss_weighting', False)):
            raise ValueError('Scheduler/learnable loss weighting is not paired.')

    @property
    def optimization_scale(self):
        if self.loss_type == 'softmax_probability_mse':
            if self.state_space == 'vocab':
                # raw_brier already sums over V, so the requested
                # 1/(B*L) * 1/2 * sum_V reduction needs only this factor.
                return 0.5
            return self.vocab_size / 2.0
        return 1.0

    def embedding_weight(self):
        return self.backbone.vocab_embed.normalized_weight()

    def classification_prototype_weight(self):
        if self.classification_prototype_mode == 'shared':
            return self.embedding_weight()
        if self.classification_prototype_mode == 'direct_vocab_state':
            raise RuntimeError(
                'Direct Task1 vocabulary bias has no learned prototype.')
        raw = self.backbone.classification_prototype
        return (F.normalize(raw.float(), dim=-1)
                * math.sqrt(raw.shape[-1])).to(raw.dtype)

    def _classification_prototype_raw(self):
        if self.classification_prototype_mode == 'shared':
            return self.backbone.vocab_embed.embedding
        if self.classification_prototype_mode == 'direct_vocab_state':
            raise RuntimeError(
                'Direct Task1 vocabulary bias has no learned prototype.')
        return self.backbone.classification_prototype

    def embed_tokens(self, tokens):
        return self.backbone.vocab_embed(tokens)

    def embed_probabilities(self, probabilities):
        return self.backbone.vocab_embed(probabilities)

    def corrupt_embeddings(self, clean_tokens, gamma, noise=None):
        clean = self.embed_tokens(clean_tokens)
        if noise is None:
            noise = torch.randn_like(clean)
        return langflow_corrupt_embedding(clean, gamma, noise), noise

    def corrupt_vocab_state(self, clean_tokens, t, noise=None):
        preserve_noise = bool(
            self.training and self._task1_contract_smoke_enabled())
        state, _, sampled_noise = flm_corrupt_vocab(
            clean_tokens, t, self.vocab_size, noise=noise,
            return_target=False, return_noise=preserve_noise)
        return state, sampled_noise

    def _state_embedding(self, state):
        if self.state_space == 'vocab':
            return self.embed_probabilities(state)
        return state

    def _path_alpha_sigma(self, gamma):
        if self.state_space == 'vocab':
            return flm_linear_alpha_sigma(gamma)
        return langflow_alpha_sigma(gamma)

    def _task1_physical_time(self, u):
        if self.lut_a2g is None:
            raise RuntimeError('Task1 requires the official FLM inverse LUT.')
        return self._tau_to_t(u).float().clamp(0.0, 1.0)

    def task1_training_time_contract(self):
        """Return the runtime LUT anchors and configured training sampler."""
        if self.state_space != 'vocab':
            return None
        tau = torch.tensor(
            [0.0, 1.0 / 128.0, 0.1],
            dtype=torch.float32, device=self.device)
        physical_t = tau.new_tensor(self._task1_lut_anchor_physical_t)
        contract = {
            'mode': self.training_time_sampling,
            'returns': ['tau', 'physical_t', 'group_id'],
            'quota_block_size_sequences': 32,
            'lut_anchors': {
                'tau': tau.detach().cpu().tolist(),
                'physical_t': physical_t.detach().cpu().tolist(),
            },
            'importance_correction': False,
            'extra_group_loss_weighting': False,
        }
        if self.training_time_sampling != 'uniform_tau':
            spec = self._task1_training_time_spec()
            contract.update({
                'active_stage': spec['stage'],
                'active_group_counts': spec['counts'],
                'active_global_batch_group_counts': (
                    spec['counts'] if sum(spec['counts']) == 256 else [
                        count * 8 for count in spec['counts']]),
                'group_names': spec['group_names'],
                'quota_schedule': spec['schedule'],
            })
        return contract

    def _task1_training_generator(self, name):
        """Return a checkpointed RNG stream used only by Task1 training."""
        generator = self._task1_training_generators.get(name)
        if generator is not None:
            return generator
        generator = torch.Generator(device=self.device)
        experiment = getattr(self.config, 'experiment', {})
        base_seed = int(getattr(
            experiment, 'training_rng_seed', int(self.config.seed) + 100000))
        offsets = {'time_group': 11, 'time_value': 23,
                   'gaussian_noise': 37, 'self_conditioning': 53}
        generator.manual_seed(base_seed + offsets[name])
        pending = self._task1_pending_training_rng_state or {}
        state = pending.get(name)
        if state is not None:
            generator.set_state(state.cpu())
        self._task1_training_generators[name] = generator
        return generator

    def task1_training_rng_state(self):
        states = {
            name: generator.get_state().cpu()
            for name, generator in self._task1_training_generators.items()
        }
        pending = self._task1_pending_training_rng_state or {}
        for name, state in pending.items():
            states.setdefault(name, state.cpu())
        return states

    def _task1_training_time_spec(self):
        common_names = [
            'tau_zero', 'uniform_tau_first_interval',
            'uniform_physical_t_first_interval',
            'uniform_tau_high_noise_remainder']
        if self.training_time_sampling == 'high_noise_quota32':
            return {
                'stage': 'h0_15k_30k',
                'counts': [1, 4, 4, 17, 6],
                'group_names': common_names + ['uniform_tau_0p1_1p0'],
                'tau_ranges': {4: (0.1, 1.0)},
                'schedule': [{
                    'start_global_step': 15000,
                    'end_global_step': 30000,
                    'counts': [1, 4, 4, 17, 6],
                }],
            }
        if self.training_time_sampling == 'v1_q30_frozen_global256':
            return {
                'stage': 'v1_q30_frozen_30k_45k',
                'counts': global_group_counts(
                    'v1_q30_frozen_global256'),
                'group_names': common_names + [
                    'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                    'uniform_tau_0p7_1p0'],
                'tau_ranges': {
                    4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
                'schedule': [{
                    'start_global_step': 30000,
                    'end_global_step': 45000,
                    'counts': global_group_counts(
                        'v1_q30_frozen_global256'),
                    'count_scope': 'global_batch_256',
                }],
            }
        if self.training_time_sampling == 'v1_m_tau25_global256':
            return {
                'stage': 'v1_m_tau25_30k_45k',
                'counts': global_group_counts('v1_m_tau25_global256'),
                'group_names': common_names + [
                    'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                    'uniform_tau_0p7_1p0'],
                'tau_ranges': {
                    4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
                'schedule': [{
                    'start_global_step': 30000,
                    'end_global_step': 45000,
                    'counts': global_group_counts(
                        'v1_m_tau25_global256'),
                    'count_scope': 'global_batch_256',
                    'mixture': (
                        '0.75*q30+0.125*U_tau(0.1,0.4)'
                        '+0.125*U_tau(0.4,0.7)'),
                }],
            }
        schedule = [
            (15000, 18000, [2, 4, 6, 7, 7, 4, 2]),
            (18000, 24000, [1, 3, 5, 8, 8, 5, 2]),
            (24000, 30000, [1, 1, 4, 7, 10, 6, 3]),
        ]
        step = int(self.global_step)
        selected = schedule[0]
        for candidate in schedule:
            if candidate[0] <= step < candidate[1]:
                selected = candidate
                break
            if step >= candidate[1]:
                selected = candidate
        start, end, counts = selected
        return {
            'stage': f'v1_{start}_{end}',
            'counts': counts,
            'group_names': common_names + [
                'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                'uniform_tau_0p7_1p0'],
            'tau_ranges': {4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
            'schedule': [
                {'start_global_step': item_start,
                 'end_global_step': item_end,
                 'counts': item_counts}
                for item_start, item_end, item_counts in schedule
            ],
        }

    def _sample_task1_training_times(
            self, batch_size, current_accumulation_step=None):
        """Return paired tau, physical time, and group identity."""
        if self.training_time_sampling == 'uniform_tau':
            tau = self._sample_t_interval(
                batch_size, current_accumulation_step,
                t_min=0.0, t_max=1.0)
            physical_t = self._task1_physical_time(tau)
            group_id = torch.full(
                (batch_size,), -1, dtype=torch.long, device=self.device)
            return tau, physical_t, group_id

        global_plan_mode = self.training_time_sampling in {
            'v1_q30_frozen_global256', 'v1_m_tau25_global256'}
        if batch_size not in {32, 64, 128, 256}:
            raise ValueError(
                'Task1 quota sampling requires a batch of 32, 64, 128, or '
                '256 sequences.')
        spec = self._task1_training_time_spec()
        base_group_id = torch.tensor([
            group_index
            for group_index, count in enumerate(spec['counts'])
            for _ in range(count)
        ], dtype=torch.long, device=self.device)
        if global_plan_mode:
            group_generator = self._task1_training_generator('time_group')
            value_generator = self._task1_training_generator('time_value')
            if current_accumulation_step is None:
                if batch_size != 256:
                    raise ValueError(
                        'Global-256 quota sampling requires accumulation step '
                        'or a complete batch of 256 sequences.')
                current_accumulation_step = 0
            accumulation = 256 // batch_size
            if not 0 <= int(current_accumulation_step) < accumulation:
                raise ValueError('Invalid accumulation step for global plan.')
            cache_key = (int(self.global_step), self.training_time_sampling,
                         batch_size)
            if (self._task1_cached_global_time_plan is None
                    or self._task1_cached_global_time_plan['key'] != cache_key
                    or int(current_accumulation_step) == 0):
                permutation = torch.randperm(
                    256, device=self.device, generator=group_generator)
                global_group_id = base_group_id[permutation]
                global_random = torch.rand(
                    256, dtype=torch.float32, device=self.device,
                    generator=value_generator)
                self._task1_cached_global_time_plan = {
                    'key': cache_key,
                    'group_id': global_group_id,
                    'random': global_random,
                }
            start = int(current_accumulation_step) * batch_size
            stop = start + batch_size
            group_id = self._task1_cached_global_time_plan[
                'group_id'][start:stop]
            random = self._task1_cached_global_time_plan[
                'random'][start:stop]
        else:
            block_count = batch_size // 32
            group_blocks = base_group_id.repeat(block_count, 1)
            permutations = torch.rand(
                block_count, 32, device=self.device).argsort(dim=1)
            group_id = group_blocks.gather(1, permutations).flatten()
            random = torch.rand(
                batch_size, dtype=torch.float32, device=self.device)
        tau = torch.empty(
            batch_size, dtype=torch.float32, device=self.device)
        physical_t = torch.empty_like(tau)
        first_tau = 1.0 / 128.0
        first_physical_t = tau.new_tensor(
            self._task1_lut_anchor_physical_t[1])

        mask = group_id == 0
        tau[mask] = 0.0

        mask = group_id == 1
        tau[mask] = random[mask] * first_tau

        physical_mask = group_id == 2
        physical_t[physical_mask] = random[physical_mask] * first_physical_t
        tau[physical_mask] = self._t_to_tau(
            physical_t[physical_mask]).float().clamp(
            0.0, first_tau)

        mask = group_id == 3
        tau[mask] = first_tau + random[mask] * (0.1 - first_tau)

        for group_index, (lower, upper) in spec['tau_ranges'].items():
            mask = group_id == group_index
            tau[mask] = lower + random[mask] * (upper - lower)
        mapped_physical_t = self._task1_physical_time(tau)
        physical_t[~physical_mask] = mapped_physical_t[~physical_mask]
        physical_t[group_id == 0] = 0.0
        return tau, physical_t, group_id

    def _record_task1_training_times(
            self, tau, physical_t, group_id,
            current_accumulation_step=None):
        if self.training_time_sampling not in {
                'high_noise_quota32', 'v1_staged_quota32',
                'v1_q30_frozen_global256', 'v1_m_tau25_global256'}:
            return
        step = int(self.global_step) + 1
        should_record = (
            not self.task1_training_time_diagnostics
            or (current_accumulation_step in {None, 0}
                and step % self.diagnostic_interval_steps == 0))
        if not should_record:
            return
        spec = self._task1_training_time_spec()
        group_count = len(spec['counts'])
        counts = torch.bincount(group_id, minlength=group_count)
        proportions = counts.float() / group_id.numel()
        groups = {}
        for index in range(group_count):
            mask = group_id == index
            if not bool(mask.any()):
                groups[str(index)] = {
                    'count': 0,
                    'proportion': 0.0,
                    'tau_min': None, 'tau_max': None, 'tau_mean': None,
                    'physical_t_min': None, 'physical_t_max': None,
                    'physical_t_mean': None,
                }
                continue
            groups[str(index)] = {
                'count': int(counts[index].detach().cpu()),
                'proportion': float(proportions[index].detach().cpu()),
                'tau_min': float(tau[mask].detach().min().cpu()),
                'tau_max': float(tau[mask].detach().max().cpu()),
                'tau_mean': float(tau[mask].detach().mean().cpu()),
                'physical_t_min': float(
                    physical_t[mask].detach().min().cpu()),
                'physical_t_max': float(
                    physical_t[mask].detach().max().cpu()),
                'physical_t_mean': float(
                    physical_t[mask].detach().mean().cpu()),
            }
        self.task1_training_time_diagnostics = {
            'optimizer_step': step,
            'mode': self.training_time_sampling,
            'stage': spec['stage'],
            'micro_batch_sequences': int(group_id.numel()),
            'group_counts': counts.detach().cpu().tolist(),
            'group_proportions': proportions.detach().cpu().tolist(),
            'global_batch_group_counts': (
                spec['counts'] if sum(spec['counts']) == 256 else [
                    count * 8 for count in spec['counts']]),
            'groups': groups,
            'tau': {
                'min': float(tau.detach().min().cpu()),
                'max': float(tau.detach().max().cpu()),
                'mean': float(tau.detach().mean().cpu()),
            },
            'physical_t': {
                'min': float(physical_t.detach().min().cpu()),
                'max': float(physical_t.detach().max().cpu()),
                'mean': float(physical_t.detach().mean().cpu()),
            },
        }
        for index in range(group_count):
            self.log(
                f'train/task1_time_group_{index}_count', counts[index],
                on_step=True, on_epoch=False, sync_dist=True,
                batch_size=group_id.numel())
            self.log(
                f'train/task1_time_group_{index}_proportion',
                proportions[index], on_step=True, on_epoch=False,
                sync_dist=True, batch_size=group_id.numel())

    def _task1_condition(self, u, physical_t):
        return flm_model_time_condition(
            u, physical_t, self.model_time_condition, self.flm_time_eps)

    def _task1_contract_smoke_enabled(self):
        experiment = getattr(self.config, 'experiment', None)
        return bool(getattr(experiment, 'task1_contract_smoke', False))

    def _write_task1_training_trace(
            self, tokens, noise, state, u, physical_t, model_time,
            group_id, bias_weight, optimized_loss,
            self_conditioning_active):
        if (not self._task1_contract_smoke_enabled()
                or self._task1_training_trace_written
                or int(getattr(self, 'global_rank', 0)) != 0):
            return
        safe_t = physical_t.detach().float().clamp(
            0.0, 1.0 - self.flm_time_eps)
        coefficient = (
            float(bias_weight) * safe_t / (1.0 - safe_t).square())
        shapes = self._task1_last_shapes or {}
        trace = {
            'variant': self.model_time_condition,
            'clean_one_hot_shape': list(state.shape),
            'gaussian_source_shape': list(noise.shape),
            'noisy_state_shape': list(state.shape),
            'hidden_shape': shapes.get('hidden'),
            'logits_shape': shapes.get('logits'),
            'probability_shape': shapes.get('logits'),
            'tau_values': u.detach().float().cpu().tolist(),
            'physical_time_values': physical_t.detach().float().cpu().tolist(),
            'training_time_group_ids': group_id.detach().cpu().tolist(),
            'model_time_condition_values': (
                model_time.detach().float().cpu().tolist()),
            'bias_coefficient_values': coefficient.cpu().tolist(),
            'loss': float(optimized_loss.detach().float().mean().cpu()),
            'loss_reduction': '0.5 * vocab_sum_then_valid_token_mean',
            'self_conditioning_active': bool(self_conditioning_active),
            'clean_token_probe': tokens.detach().flatten()[:8].cpu().tolist(),
            'gaussian_source_probe': (
                noise.detach().float().flatten()[:8].cpu().tolist()),
            'noisy_state_probe': (
                state.detach().float().flatten()[:8].cpu().tolist()),
        }
        output_dir = Path(str(self.config.checkpointing.save_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'task1_training_trace.json').write_text(
            json.dumps(trace, indent=2), encoding='utf-8')
        self._task1_training_trace_written = True

    def _current_token_bias_weight(self):
        if self.token_bias_schedule == 'none':
            return 0.0
        if self.token_bias_schedule == 'full':
            return 1.0
        step = int(self.global_step)
        if self.config.mode in {'sample_eval', 'ppl_eval'}:
            step = int(getattr(
                self, '_loaded_checkpoint_global_step', step))
        return langflow_token_bias_weight(
            step, self.token_bias_warmup_steps)

    def _forward_logits(self, z_gamma, gamma, x_self_cond=None,
                        bias_weight=None, physical_t=None):
        gamma = self._process_sigma(gamma)
        state_embedding = self._state_embedding(z_gamma)
        if bias_weight is None:
            bias_weight = self._current_token_bias_weight()
        residual_logits = self.backbone(
            state_embedding,
            gamma,
            inputs_are_embeddings=True,
            x_self_cond=x_self_cond)
        if self.state_space == 'vocab' and self._task1_contract_smoke_enabled():
            self._task1_last_shapes = {
                'hidden': list(state_embedding.shape),
                'logits': list(residual_logits.shape),
            }
        # The no-bias arm exits before alpha/sigma, prototype selection, or the
        # vocabulary matmul. This is an absent branch, not a detached or
        # zero-multiplied bias.
        if not bias_weight:
            self._record_validation_logit_stats(
                residual_logits, None, residual_logits, bias_weight)
            return residual_logits

        if self.state_space == 'vocab':
            if physical_t is None:
                raise ValueError(
                    'Task1 direct Gaussian bias requires physical_t.')
            bias_logits = flm_vocab_gaussian_bias(
                z_gamma, physical_t, bias_weight, self.flm_time_eps)
            logits = residual_logits + bias_logits.to(residual_logits.dtype)
            self._record_validation_logit_stats(
                residual_logits, bias_logits, logits, bias_weight)
            return logits

        alpha, sigma = self._path_alpha_sigma(gamma)
        coefficient = (
            float(bias_weight) * alpha / sigma.square().clamp_min(1e-12))
        candidate_weight = self.classification_prototype_weight()
        if self.classification_prototype_mode == 'shared':
            candidate_weight = candidate_bias_codebook(
                candidate_weight, self.codebook_gradient_mode)
        route_step = int(self.global_step) + 1
        route_active = (
            self.training and torch.is_grad_enabled() and route_step > 0
            and route_step % self.gradient_route_interval_steps == 0)
        if route_active:
            self._route_step = route_step
            self._route_candidate_weight_grad = None
            self._route_candidate_raw_snapshot = (
                self._classification_prototype_raw().detach().clone())
            self._route_physical_raw_snapshot = (
                self.backbone.vocab_embed.embedding.detach().clone())
            if candidate_weight.requires_grad:
                candidate_weight.register_hook(
                    lambda grad: setattr(
                        self, '_route_candidate_weight_grad',
                        grad.detach().clone()))
        token_bias = torch.matmul(
            state_embedding, candidate_weight.transpose(0, 1))
        bias_logits = coefficient[:, None, None].to(
            token_bias.dtype) * token_bias
        logits = residual_logits + bias_logits
        self._record_validation_logit_stats(
            residual_logits, bias_logits, logits, bias_weight)
        return logits

    def forward(self, z_gamma, gamma, x_self_cond=None, bias_weight=None,
                physical_t=None):
        logits = self._forward_logits(
            z_gamma, gamma, x_self_cond=x_self_cond,
            bias_weight=bias_weight, physical_t=physical_t)
        return F.log_softmax(logits.float(), dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        self._active_valid_tokens = valid_tokens
        return x0, None, valid_tokens

    def validation_step(self, batch, batch_idx):
        """Keep validation gamma, corruption noise, and data trace repeatable."""
        self._active_validation_batch_idx = int(batch_idx)
        validation_seed = int(getattr(
            self.config.algo, 'validation_seed', 1729)) + int(batch_idx)
        cuda_devices = (
            [self.device.index]
            if self.device.type == 'cuda' and self.device.index is not None
            else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(validation_seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(validation_seed)
            return super().validation_step(batch, batch_idx)

    def _sample_validation_q(self, batch_size):
        """Optionally stratify validation tau while retaining random in-bin draws."""
        tau_stratified = bool(getattr(
            self.config.algo, 'validation_tau_stratified', False))
        physical_t_stratified = bool(getattr(
            self.config.algo, 'validation_physical_t_stratified', False))
        if tau_stratified and physical_t_stratified:
            raise ValueError(
                'Validation may stratify tau or physical t, not both.')
        if not tau_stratified and not physical_t_stratified:
            return self._sample_t_interval(batch_size, None, 0.0, 1.0)
        edges = self._validation_tau_edges()
        bin_count = edges.numel() - 1
        if bin_count <= 0:
            raise ValueError('validation_tau_bin_count must be positive')
        batch_idx = int(getattr(self, '_active_validation_batch_idx', 0))
        sample_indices = (
            torch.arange(batch_size, device=self.device)
            + batch_idx * batch_size)
        bin_indices = sample_indices.remainder(bin_count)
        in_bin_offsets = torch.rand(batch_size, device=self.device)
        starts = edges[bin_indices]
        ends = edges[bin_indices + 1]
        coordinate = starts + in_bin_offsets * (ends - starts)
        if physical_t_stratified:
            return self._t_to_tau(coordinate).float().clamp(0.0, 1.0)
        return coordinate

    def _validation_tau_edges(self):
        configured = getattr(
            self.config.algo, 'validation_tau_bin_edges', None)
        if configured is None:
            bin_count = int(getattr(
                self.config.algo, 'validation_tau_bin_count', 10))
            return torch.linspace(
                0.0, 1.0, bin_count + 1, device=self.device)
        edges = torch.tensor(
            list(configured), dtype=torch.float32, device=self.device)
        if (edges.ndim != 1 or edges.numel() < 2
                or float(edges[0]) != 0.0 or float(edges[-1]) != 1.0
                or not bool(torch.all(edges[1:] > edges[:-1]))):
            raise ValueError(
                'validation_tau_bin_edges must be strictly increasing from '
                '0 to 1.')
        return edges

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, xT, given_t, not_sampling_t
        batch_size = x0.shape[0]
        task1_group_id = None
        if self.state_space == 'vocab':
            if train_mode:
                q, t, task1_group_id = self._sample_task1_training_times(
                    batch_size, current_accumulation_step)
                self._record_task1_training_times(
                    q, t, task1_group_id, current_accumulation_step)
            else:
                q = self._sample_validation_q(batch_size)
                t = self._task1_physical_time(q)
            gamma = self._task1_condition(q, t).to(self.device)
            training_noise = None
            isolated_rng_mode = self.training_time_sampling in {
                'v1_q30_frozen_global256', 'v1_m_tau25_global256'}
            if train_mode and isolated_rng_mode:
                training_noise = torch.randn(
                    (batch_size, x0.shape[1], self.vocab_size),
                    dtype=torch.float32, device=self.device,
                    generator=self._task1_training_generator(
                        'gaussian_noise'))
            z_gamma, sampled_noise = self.corrupt_vocab_state(
                x0, t, noise=training_noise)
            alpha, sigma = t, 1.0 - t
        else:
            q = (self._sample_t_interval(
                batch_size, current_accumulation_step,
                t_min=0.0, t_max=1.0) if train_mode
                 else self._sample_validation_q(batch_size))
            gamma = langflow_gumbel_gamma(
                q, self.gumbel_loc, self.gumbel_scale,
                self.gumbel_cutoff).to(self.device)
            z_gamma, sampled_noise = self.corrupt_embeddings(x0, gamma)
            alpha, sigma = self._path_alpha_sigma(gamma)

        use_self_conditioning = False
        x_self_cond = None
        if train_mode:
            self._training_token_counts += torch.bincount(
                x0.detach().flatten(), minlength=self.vocab_size)
            if (self.state_space == 'vocab'
                    and self.training_time_sampling in {
                        'v1_q30_frozen_global256',
                        'v1_m_tau25_global256'}):
                sc_draw = torch.rand(
                    (), device=self.device,
                    generator=self._task1_training_generator(
                        'self_conditioning'))
            else:
                sc_draw = torch.rand((), device=self.device)
            use_self_conditioning = bool(
                sc_draw < self.self_condition_probability)

        prefix = 'train' if train_mode else 'val'
        self.log(
            f'{prefix}/hybrid_gamma_mean', gamma.detach().mean(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_gamma_std', gamma.detach().std(unbiased=False),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_gamma_min', gamma.detach().min(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_gamma_max', gamma.detach().max(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_alpha_mean', alpha.detach().mean(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_sigma_mean', sigma.detach().mean(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log(
            f'{prefix}/hybrid_token_bias_weight',
            float(self._current_token_bias_weight()),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        if self.state_space == 'vocab':
            bias_weight = float(self._current_token_bias_weight())
            safe_t = t.detach().float().clamp(
                0.0, 1.0 - self.flm_time_eps)
            bias_coefficient = (
                bias_weight * safe_t / (1.0 - safe_t).square())
            for name, value in (
                    ('task1_tau_mean', q.detach().mean()),
                    ('task1_physical_t_mean', t.detach().mean()),
                    ('task1_model_time_condition_mean', gamma.detach().mean()),
                    ('task1_bias_coefficient_mean', bias_coefficient.mean())):
                self.log(
                    f'{prefix}/{name}', value,
                    on_step=train_mode, on_epoch=not train_mode,
                    sync_dist=True, batch_size=batch_size)
        if train_mode:
            self.log(
                'train/hybrid_self_conditioning_active',
                float(use_self_conditioning), on_step=True, on_epoch=False,
                sync_dist=True, batch_size=batch_size)

        if use_self_conditioning:
            with torch.no_grad():
                initial_logits = self._forward_logits(
                    z_gamma, gamma, x_self_cond=None,
                    physical_t=t if self.state_space == 'vocab' else None)
                initial_probabilities = F.softmax(
                    initial_logits.float(), dim=-1)
                x_self_cond = detached_self_conditioning_embedding(
                    initial_probabilities, self.embed_probabilities)

        logits = self._forward_logits(
            z_gamma, gamma, x_self_cond=x_self_cond,
            physical_t=t if self.state_space == 'vocab' else None)
        diagnostics = probability_prediction_metrics(
            logits, x0, include_distribution_stats=not train_mode)
        if train_mode and self._route_step == int(self.global_step) + 1:
            probabilities_for_route = F.softmax(logits.detach().float(), dim=-1)
            targets_for_route = x0.detach()

            def capture_logit_route(gradient):
                non_target = torch.ones_like(gradient, dtype=torch.bool)
                non_target.scatter_(
                    -1, targets_for_route.unsqueeze(-1), False)
                negative_wrong = (gradient < 0) & non_target
                wrong_probability = probabilities_for_route.masked_fill(
                    ~negative_wrong, 0).sum()
                total_wrong_probability = probabilities_for_route.masked_fill(
                    ~non_target, 0).sum().clamp_min(1e-20)
                negative_energy = gradient.masked_fill(
                    ~negative_wrong, 0).square().sum()
                total_energy = gradient.square().sum().clamp_min(1e-20)
                self._route_logit_stats = {
                    'negative_wrong_probability_mass_fraction': float(
                        (wrong_probability / total_wrong_probability)
                        .detach().cpu()),
                    'negative_wrong_gradient_energy_fraction': float(
                        (negative_energy / total_energy).detach().cpu()),
                }
                return gradient

            logits.register_hook(capture_logit_route)
        valid_tokens = getattr(
            self, '_active_valid_tokens',
            torch.ones_like(x0, dtype=torch.float32))
        if not train_mode:
            self._accumulate_gamma_bins(
                q, t, diagnostics, valid_tokens, target_tokens=x0)
        self._log_probability_metrics(diagnostics, train_mode, valid_tokens)
        self.log(
            f'{"train" if train_mode else "val"}/hybrid_optimized_scale',
            float(self.optimization_scale),
            on_step=train_mode,
            on_epoch=not train_mode,
            sync_dist=True,
            batch_size=batch_size)

        if self.loss_type == 'cross_entropy':
            optimized = diagnostics['token_ce']
        else:
            optimized = diagnostics['raw_brier'] * self.optimization_scale
        if self.state_space == 'vocab':
            probability_mse = diagnostics['raw_brier'] * 0.5
            probability_mse_mean = (
                probability_mse.detach() * valid_tokens).sum()
            probability_mse_mean = (
                probability_mse_mean / valid_tokens.sum().clamp_min(1))
            self.log(
                f'{prefix}/hybrid_probability_mse', probability_mse_mean,
                on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
                batch_size=batch_size)
        if self.state_space == 'vocab' and train_mode:
            self._write_task1_training_trace(
                x0, sampled_noise, z_gamma, q, t, gamma,
                task1_group_id, self._current_token_bias_weight(), optimized,
                use_self_conditioning)
        del sampled_noise
        nonfinite_values = (~torch.isfinite(optimized.detach())).sum()
        if train_mode:
            self._task1_nonfinite_loss_values.add_(
                nonfinite_values.to(dtype=torch.long))
            self._task1_nonfinite_loss_events.add_(
                (nonfinite_values > 0).to(dtype=torch.long))
        self.log(
            f'{prefix}/hybrid_nonfinite_count',
            nonfinite_values.float(),
            on_step=train_mode, on_epoch=not train_mode, sync_dist=True,
            batch_size=batch_size)
        self.log('loss', optimized.detach().mean(), prog_bar=True)
        return optimized
