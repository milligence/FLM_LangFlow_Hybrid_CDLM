"""Closed no-self-conditioning probability-MSE teacher for Task1."""

import collections

import torch
import torch.nn.functional as F

from algo import LangFlowFLMHybrid
from langflow_hybrid.ops import (
    flm_linear_euler_update,
    flm_model_time_condition,
    flm_vocab_gaussian_bias,
    probability_prediction_loss,
)


class Task1NoSCMSETeacher(LangFlowFLMHybrid):
    """A closed (x, t) -> posterior field with no hidden SC state."""

    def __init__(self, config, tokenizer):
        if bool(getattr(config.algo, 'self_conditioning', False)):
            raise ValueError('The no-SC teacher requires self_conditioning=false.')
        if float(getattr(config.algo, 'self_condition_probability', 0.0)) != 0.0:
            raise ValueError('The no-SC teacher requires SC probability zero.')
        if str(config.algo.backbone) != 'no_sc_dit':
            raise ValueError('The no-SC teacher requires the NoSCDIT backbone.')
        if str(config.algo.loss_type) != 'softmax_probability_mse':
            raise ValueError('The no-SC teacher requires probability MSE.')
        super().__init__(config, tokenizer)
        forbidden = [
            name for name, _ in self.named_parameters()
            if 'self_cond' in name or 'self_condition' in name]
        if forbidden:
            raise RuntimeError(f'No-SC model contains forbidden parameters: {forbidden}')

    def _forward_logits(self, z_gamma, gamma, bias_weight=None,
                        physical_t=None):
        gamma = self._process_sigma(gamma)
        state_embedding = self._state_embedding(z_gamma)
        if bias_weight is None:
            bias_weight = self._current_token_bias_weight()
        residual_logits = self.backbone(
            state_embedding, gamma, inputs_are_embeddings=True)
        if not bias_weight:
            return residual_logits
        if physical_t is None:
            raise ValueError('Task1 direct Gaussian bias requires physical_t.')
        bias_logits = flm_vocab_gaussian_bias(
            z_gamma, physical_t, bias_weight, self.flm_time_eps)
        return residual_logits + bias_logits.to(residual_logits.dtype)

    def forward(self, z_gamma, gamma, bias_weight=None, physical_t=None):
        logits = self._forward_logits(
            z_gamma, gamma, bias_weight=bias_weight,
            physical_t=physical_t)
        return F.log_softmax(logits.float(), dim=-1)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        del output_tokens, xT, given_t, not_sampling_t
        batch_size = x0.shape[0]
        if train_mode:
            q, physical_t, _ = self._sample_task1_training_times(
                batch_size, current_accumulation_step)
        else:
            q = self._sample_validation_q(batch_size)
            physical_t = self._task1_physical_time(q)
        gamma = self._task1_condition(q, physical_t).to(self.device)
        training_noise = None
        if train_mode and self.training_time_sampling in {
                'v1_q30_frozen_global256', 'v1_m_tau25_global256',
                'v1_staged_global256',
                'no_sc_high_noise_global256',
                'no_sc_legacy_uniform_physical_global256'}:
            training_noise = torch.randn(
                (batch_size, x0.shape[1], self.vocab_size),
                dtype=torch.float32, device=self.device,
                generator=self._task1_training_generator('gaussian_noise'))
        state = self.corrupt_vocab_state(
            x0, physical_t, noise=training_noise)
        logits = self._forward_logits(
            state, gamma, physical_t=physical_t)
        optimized = probability_prediction_loss(
            logits, x0, self.loss_type) * self.optimization_scale
        self.log('loss', optimized.detach().mean(), prog_bar=True)
        return optimized

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if isinstance(num_steps, collections.abc.Sequence):
            if len(num_steps) != 1:
                raise ValueError('generate_samples expects one NFE value.')
            num_steps = int(num_steps[0])
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError('Sampling NFE must be positive.')

        u_grid, t_grid, metadata = self._task1_vocab_sampling_grid(num_steps)
        seed = getattr(self.config.sampling, 'task1_initial_noise_seed', None)
        schedule = str(getattr(
            self.config.sampling, 'task1_initial_noise_schedule',
            'base_seed_plus_batch_index'))
        shape = (num_samples, self.num_tokens, self.vocab_size)
        if seed is not None and schedule == 'base_seed_plus_sample_index':
            offset = int(getattr(self, '_task1_sampling_sample_offset', 0))
            state = torch.empty(shape, device=self.device, dtype=torch.float32)
            generator = torch.Generator(device=self.device)
            for index in range(num_samples):
                generator.manual_seed(int(seed) + offset + index)
                state[index].normal_(generator=generator)
            self._task1_sampling_sample_offset = offset + num_samples
        elif schedule == 'base_seed_plus_batch_index':
            batch_index = int(getattr(self, '_task1_sampling_batch_index', 0))
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(seed) + batch_index)
            state = torch.randn(
                shape, device=self.device, dtype=torch.float32,
                generator=generator)
            self._task1_sampling_batch_index = batch_index + 1
        else:
            raise ValueError(f'Unsupported initial-noise schedule: {schedule!r}.')
        state.mul_(float(metadata.get('initial_noise_scale', 1.0)))

        probabilities = None
        self.last_sampling_nfe = 0
        bias_weight = self._current_token_bias_weight()
        for index in range(num_steps):
            u = u_grid[index].expand(num_samples)
            physical_t = t_grid[index].expand(num_samples)
            model_time = flm_model_time_condition(
                u, physical_t, self.model_time_condition,
                self.flm_time_eps).to(self.device)
            logits = self._forward_logits(
                state, model_time, bias_weight=bias_weight,
                physical_t=physical_t)
            self.last_sampling_nfe += 1
            probabilities = F.softmax(logits.float(), dim=-1)
            if index + 1 < num_steps or bool(
                    metadata.get('update_after_last_query', False)):
                state = flm_linear_euler_update(
                    state, probabilities, t_grid[index],
                    t_grid[index + 1], self.flm_time_eps)
        return probabilities.argmax(dim=-1)
