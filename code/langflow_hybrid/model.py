"""Model and Lightning core for the LangFlow-FLM hybrid."""

import math
import itertools

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
    gumbel_entropy_curve,
    langflow_alpha_sigma,
    langflow_corrupt_embedding,
    langflow_gumbel_gamma,
    langflow_token_bias_weight,
    learned_gumbel_gamma,
    probability_prediction_loss,
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
        self.learned_gumbel = (
            self.state_space == 'vocab'
            and self.training_time_sampling == 'learned_gumbel')
        if self.learned_gumbel:
            entropy_init = float(getattr(
                gumbel, 'entropy_init', math.log(self.vocab_size)))
            loc_init = float(getattr(gumbel, 'loc_init', 0.0))
            scale_init = float(getattr(gumbel, 'scale_init', 1.0))
            self.gumbel_min_scale = float(getattr(
                gumbel, 'min_scale', 1e-4))
            if entropy_init <= 0.0 or scale_init <= self.gumbel_min_scale:
                raise ValueError(
                    'Learned Gumbel entropy and scale initializers must be '
                    'positive.')
            self.gumbel_entropy_raw = torch.nn.Parameter(torch.tensor(
                math.log(math.expm1(entropy_init)), dtype=torch.float32))
            self.gumbel_loc_online = torch.nn.Parameter(torch.tensor(
                loc_init, dtype=torch.float32))
            self.gumbel_scale_raw = torch.nn.Parameter(torch.tensor(
                math.log(math.expm1(
                    scale_init - self.gumbel_min_scale)),
                dtype=torch.float32))
        self.self_condition_probability = float(
            config.algo.self_condition_probability)
        self.token_bias_warmup_steps = int(
            config.algo.token_bias_warmup_steps)
        self.token_bias_schedule = str(getattr(
            config.algo, 'token_bias_schedule', 'warmup'))
        self.classification_prototype_mode = str(getattr(
            config.algo, 'classification_prototype_mode', 'shared'))
        self.codebook_gradient_mode = str(getattr(
            config.algo, 'codebook_gradient_mode', 'all'))
        if (self.codebook_gradient_mode == 'frozen'
                and self.backbone.vocab_embed.embedding.requires_grad):
            raise RuntimeError(
                'Frozen physical codebook must be configured before EMA setup.')
        self.last_sampling_nfe = 0
        self._task1_training_generators = {}
        self._task1_pending_training_rng_state = None
        self._task1_cached_global_time_plan = None
        self._task1_lut_anchor_physical_t = None
        if self.state_space == 'vocab':
            anchor_tau = torch.tensor([0.0, 1.0 / 128.0, 0.1])
            self._task1_lut_anchor_physical_t = (
                self._task1_physical_time(anchor_tau).cpu().tolist())

    @property
    def optimization_scale(self):
        if self.loss_type == 'softmax_probability_mse':
            if self.state_space == 'vocab':
                # raw_brier already sums over V, so the requested
                # 1/(B*L) * 1/2 * sum_V reduction needs only this factor.
                return 0.5
            return self.vocab_size / 2.0
        return 1.0

    def _learned_gumbel_values(self):
        if not self.learned_gumbel:
            raise RuntimeError('Learned Gumbel scheduler is not enabled.')
        entropy = F.softplus(self.gumbel_entropy_raw)
        scale = F.softplus(self.gumbel_scale_raw) + self.gumbel_min_scale
        return entropy, self.gumbel_loc_online, scale

    def _learned_gumbel_parameters(self):
        if not self.learned_gumbel:
            return ()
        return (
            self.gumbel_entropy_raw,
            self.gumbel_loc_online,
            self.gumbel_scale_raw,
        )

    def _get_optimizer_parameters(self):
        return itertools.chain(
            super()._get_optimizer_parameters(),
            self._learned_gumbel_parameters())

    def configure_gradient_clipping(
            self, optimizer, gradient_clip_val=None,
            gradient_clip_algorithm=None):
        if not self.learned_gumbel:
            return super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm)
        del optimizer, gradient_clip_algorithm
        if gradient_clip_val is None:
            return
        # The CE and scheduler losses have disjoint parameter sets. Clip them
        # separately so a transient scheduler residual cannot rescale the
        # backbone gradient and violate LangFlow's non-interference contract.
        torch.nn.utils.clip_grad_norm_(
            tuple(self._get_parameters()), float(gradient_clip_val))
        torch.nn.utils.clip_grad_norm_(
            self._learned_gumbel_parameters(), float(gradient_clip_val))

    def learned_gumbel_state(self):
        if not self.learned_gumbel:
            return None
        entropy, loc, scale = self._learned_gumbel_values()
        return {
            'entropy': float(entropy.detach().cpu()),
            'loc': float(loc.detach().cpu()),
            'scale': float(scale.detach().cpu()),
            'cutoff': self.gumbel_cutoff,
            'inference_parameters': 'online',
        }

    def _learned_gumbel_icdf(self, q):
        _, loc, scale = self._learned_gumbel_values()
        return learned_gumbel_gamma(q, loc, scale, self.gumbel_cutoff)

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
        state, _, _ = flm_corrupt_vocab(
            clean_tokens, t, self.vocab_size, noise=noise,
            return_target=False, return_noise=False)
        return state

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
        if self.training_time_sampling == 'learned_gumbel':
            contract.update({
                'returns': ['gumbel_quantile', 'physical_t', 'group_id'],
                'distribution': 'learned_gumbel_log_nsr',
                'quantile_cutoff': self.gumbel_cutoff,
                'scheduler_loss': 'detached_sequence_ce_mse',
                'scheduler_updates_from_step': 0,
                'controls_sampling_from_step': 0,
                'scheduler_ema': False,
            })
        elif self.training_time_sampling != 'uniform_tau':
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
        if self.training_time_sampling == 'no_sc_high_noise_global256':
            return {
                'stage': 'no_sc_15k_high_noise_probe',
                'counts': global_group_counts(
                    'no_sc_high_noise_global256'),
                'group_names': common_names + [
                    'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                    'uniform_tau_0p7_1p0'],
                'tau_ranges': {
                    4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
                'schedule': [{
                    'start_global_step': 15000,
                    'end_global_step': 30000,
                    'counts': global_group_counts(
                        'no_sc_high_noise_global256'),
                    'count_scope': 'global_batch_256',
                }],
            }
        if (self.training_time_sampling
                == 'no_sc_legacy_uniform_physical_global256'):
            return {
                'stage': 'no_sc_15k_legacy_plus_uniform_physical_probe',
                'counts': global_group_counts(
                    'no_sc_legacy_uniform_physical_global256'),
                'group_names': common_names + [
                    'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                    'uniform_tau_0p7_1p0', 'uniform_physical_t_0p0_1p0'],
                'tau_ranges': {
                    4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
                'schedule': [{
                    'start_global_step': 15000,
                    'end_global_step': 30000,
                    'counts': global_group_counts(
                        'no_sc_legacy_uniform_physical_global256'),
                    'count_scope': 'global_batch_256',
                    'uniform_physical_fraction': 0.125,
                }],
            }
        if self.training_time_sampling == 'v1_staged_global256':
            schedule = [
                (15000, 18000, [16, 32, 48, 56, 56, 32, 16]),
                (18000, 24000, [8, 24, 40, 64, 64, 40, 16]),
                (24000, 30000, [8, 8, 32, 56, 80, 48, 24]),
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
                'stage': f'v1_global_{start}_{end}',
                'counts': counts,
                'group_names': common_names + [
                    'uniform_tau_0p1_0p4', 'uniform_tau_0p4_0p7',
                    'uniform_tau_0p7_1p0'],
                'tau_ranges': {
                    4: (0.1, 0.4), 5: (0.4, 0.7), 6: (0.7, 1.0)},
                'schedule': [
                    {'start_global_step': item_start,
                     'end_global_step': item_end,
                     'counts': item_counts,
                     'count_scope': 'global_batch_256'}
                    for item_start, item_end, item_counts in schedule
                ],
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
        if self.training_time_sampling == 'learned_gumbel':
            q = self._sample_t_interval(
                batch_size, current_accumulation_step,
                t_min=0.0, t_max=1.0).float().clamp(
                    self.gumbel_cutoff, 1.0 - self.gumbel_cutoff)
            gamma = self._learned_gumbel_icdf(q).detach()
            physical_t = torch.sigmoid(-0.5 * gamma)
            group_id = torch.full(
                (batch_size,), -2, dtype=torch.long, device=self.device)
            return q, physical_t, group_id
        if self.training_time_sampling == 'uniform_tau':
            tau = self._sample_t_interval(
                batch_size, current_accumulation_step,
                t_min=0.0, t_max=1.0)
            physical_t = self._task1_physical_time(tau)
            group_id = torch.full(
                (batch_size,), -1, dtype=torch.long, device=self.device)
            return tau, physical_t, group_id

        global_plan_mode = self.training_time_sampling in {
            'v1_q30_frozen_global256', 'v1_m_tau25_global256',
            'v1_staged_global256',
            'no_sc_high_noise_global256',
            'no_sc_legacy_uniform_physical_global256'}
        if batch_size not in {4, 8, 16, 32, 64, 128, 256}:
            raise ValueError(
                'Task1 quota sampling requires a batch of 4, 8, 16, 32, '
                '64, 128, or 256 sequences.')
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

        full_physical_mask = group_id == 7
        physical_t[full_physical_mask] = random[full_physical_mask]
        tau[full_physical_mask] = self._t_to_tau(
            physical_t[full_physical_mask]).float().clamp(0.0, 1.0)
        physical_mask = physical_mask | full_physical_mask

        mask = group_id == 3
        tau[mask] = first_tau + random[mask] * (0.1 - first_tau)

        for group_index, (lower, upper) in spec['tau_ranges'].items():
            mask = group_id == group_index
            tau[mask] = lower + random[mask] * (upper - lower)
        mapped_physical_t = self._task1_physical_time(tau)
        physical_t[~physical_mask] = mapped_physical_t[~physical_mask]
        physical_t[group_id == 0] = 0.0
        return tau, physical_t, group_id

    def _task1_condition(self, u, physical_t):
        return flm_model_time_condition(
            u, physical_t, self.model_time_condition, self.flm_time_eps)

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
        # The no-bias arm exits before alpha/sigma, prototype selection, or the
        # vocabulary matmul. This is an absent branch, not a detached or
        # zero-multiplied bias.
        if not bias_weight:
            return residual_logits

        if self.state_space == 'vocab':
            if physical_t is None:
                raise ValueError(
                    'Task1 direct Gaussian bias requires physical_t.')
            bias_logits = flm_vocab_gaussian_bias(
                z_gamma, physical_t, bias_weight, self.flm_time_eps)
            return residual_logits + bias_logits.to(residual_logits.dtype)

        alpha, sigma = self._path_alpha_sigma(gamma)
        coefficient = (
            float(bias_weight) * alpha / sigma.square().clamp_min(1e-12))
        candidate_weight = self.classification_prototype_weight()
        if self.classification_prototype_mode == 'shared':
            candidate_weight = candidate_bias_codebook(
                candidate_weight, self.codebook_gradient_mode)
        token_bias = torch.matmul(
            state_embedding, candidate_weight.transpose(0, 1))
        bias_logits = coefficient[:, None, None].to(
            token_bias.dtype) * token_bias
        return residual_logits + bias_logits

    def forward(self, z_gamma, gamma, x_self_cond=None, bias_weight=None,
                physical_t=None):
        logits = self._forward_logits(
            z_gamma, gamma, x_self_cond=x_self_cond,
            bias_weight=bias_weight, physical_t=physical_t)
        return F.log_softmax(logits.float(), dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _sample_validation_q(self, batch_size):
        return self._sample_t_interval(batch_size, None, 0.0, 1.0)

    def _loss(self, x0, valid_tokens, current_accumulation_step=None,
              train_mode=False, xT=None, given_t=None,
              not_sampling_t=False):
        self._task1_scheduler_batch = None
        result = super()._loss(
            x0, valid_tokens, current_accumulation_step, train_mode,
            xT=xT, given_t=given_t, not_sampling_t=not_sampling_t)
        if not train_mode or self._task1_scheduler_batch is None:
            return result
        per_token_ce, gamma = self._task1_scheduler_batch
        mask = valid_tokens.to(per_token_ce.dtype)
        per_sequence_ce = (
            (per_token_ce * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0))
        entropy, loc, scale = self._learned_gumbel_values()
        predicted_entropy = gumbel_entropy_curve(
            gamma.detach(), entropy, loc, scale)
        scheduler_loss = (
            per_sequence_ce.detach() - predicted_entropy).square().mean()
        result.loss = result.loss + scheduler_loss
        self.log(
            'scheduler/loss', scheduler_loss.detach(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'scheduler/entropy', entropy.detach(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'scheduler/loc', loc.detach(),
            on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            'scheduler/scale', scale.detach(),
            on_step=True, on_epoch=False, sync_dist=True)
        return result

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, xT, given_t, not_sampling_t
        batch_size = x0.shape[0]
        if self.state_space == 'vocab':
            if train_mode:
                q, t, _ = self._sample_task1_training_times(
                    batch_size, current_accumulation_step)
            else:
                q = self._sample_validation_q(batch_size)
                t = self._task1_physical_time(q)
            if train_mode and self.learned_gumbel:
                gamma = self._learned_gumbel_icdf(q).detach().to(self.device)
            else:
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
            z_gamma = self.corrupt_vocab_state(x0, t, noise=training_noise)
        else:
            q = (self._sample_t_interval(
                batch_size, current_accumulation_step,
                t_min=0.0, t_max=1.0) if train_mode
                 else self._sample_validation_q(batch_size))
            gamma = langflow_gumbel_gamma(
                q, self.gumbel_loc, self.gumbel_scale,
                self.gumbel_cutoff).to(self.device)
            z_gamma, _ = self.corrupt_embeddings(x0, gamma)

        use_self_conditioning = False
        x_self_cond = None
        if train_mode:
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
        optimized = probability_prediction_loss(
            logits, x0, self.loss_type) * self.optimization_scale
        if train_mode and self.learned_gumbel:
            self._task1_scheduler_batch = (
                optimized.detach(), gamma.detach())
        self.log('loss', optimized.detach().mean(), prog_bar=True)
        return optimized
