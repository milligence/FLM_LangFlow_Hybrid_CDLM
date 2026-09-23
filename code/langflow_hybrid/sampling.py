"""Fixed-NFE Euler-EDM generation for the LangFlow-FLM hybrid."""

import collections
import json

import torch
import torch.nn.functional as F

from .ops import (
    finite_sc_gate,
    flm_linear_euler_update,
    flm_linear_gamma,
    flm_model_time_condition,
    langflow_euler_edm_update,
    langflow_gumbel_gamma,
)


class LangFlowSamplingMixin:
    """Sampling methods kept separate from the training and diagnostic paths."""

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

        if getattr(self, 'state_space', 'embedding') == 'vocab':
            return self._generate_vocab_state_samples(num_samples, num_steps)

        q_grid = torch.linspace(
            1.0 - self.gumbel_cutoff,
            self.gumbel_cutoff,
            num_steps,
            device=self.device)
        gamma_grid = langflow_gumbel_gamma(
            q_grid, self.gumbel_loc, self.gumbel_scale,
            self.gumbel_cutoff).to(self.device)
        z_gamma = torch.randn(
            (num_samples, self.num_tokens,
             self.config.model.hidden_size),
            device=self.device,
            dtype=self.embedding_weight().dtype)
        x_self_cond = None
        predicted_clean = None
        self.last_sampling_nfe = 0
        bias_weight = self._current_token_bias_weight()

        for index, gamma_scalar in enumerate(gamma_grid):
            gamma = gamma_scalar.expand(num_samples)
            logits = self._forward_logits(
                z_gamma, gamma,
                x_self_cond=x_self_cond,
                bias_weight=bias_weight)
            self.last_sampling_nfe += 1
            probabilities = F.softmax(logits.float(), dim=-1)
            predicted_clean = self.embed_probabilities(
                probabilities).detach()
            x_self_cond = predicted_clean
            if index + 1 < num_steps:
                next_gamma = gamma_grid[index + 1]
                z_gamma = langflow_euler_edm_update(
                    z_gamma,
                    predicted_clean,
                    gamma_scalar,
                    next_gamma)

        if self.last_sampling_nfe != num_steps:
            raise RuntimeError(
                f'Backbone NFE mismatch: {self.last_sampling_nfe} != '
                f'{num_steps}.')
        return probabilities.argmax(dim=-1)

    def _generate_vocab_state_samples(self, num_samples, num_steps):
        """Generate along the FLM vocabulary-state path with LangFlow blocks."""
        eps = float(getattr(self, 'flm_time_eps', 1e-5))
        u_grid, t_grid, grid_metadata = self._task1_vocab_sampling_grid(
            num_steps)
        batch_index = int(getattr(self, '_task1_sampling_batch_index', 0))
        initial_noise_seed = getattr(
            self.config.sampling, 'task1_initial_noise_seed', None)
        initial_noise_schedule = str(getattr(
            self.config.sampling, 'task1_initial_noise_schedule',
            'base_seed_plus_batch_index'))
        state_shape = (num_samples, self.num_tokens, self.vocab_size)
        if (initial_noise_seed is not None
                and initial_noise_schedule == 'base_seed_plus_sample_index'):
            sample_offset = int(getattr(
                self, '_task1_sampling_sample_offset', 0))
            state = torch.empty(
                state_shape, device=self.device, dtype=torch.float32)
            generator = torch.Generator(device=self.device)
            for local_index in range(num_samples):
                generator.manual_seed(
                    int(initial_noise_seed) + sample_offset + local_index)
                state[local_index].normal_(generator=generator)
            self._task1_sampling_sample_offset = sample_offset + num_samples
        elif initial_noise_schedule == 'base_seed_plus_batch_index':
            generator = None
            if initial_noise_seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(initial_noise_seed) + batch_index)
            state = torch.randn(
                state_shape, device=self.device, dtype=torch.float32,
                generator=generator)
        else:
            raise ValueError(
                'Unsupported Task1 initial-noise schedule: '
                f'{initial_noise_schedule!r}.')
        state.mul_(float(grid_metadata.get('initial_noise_scale', 1.0)))
        self._task1_sampling_batch_index = batch_index + 1
        x_self_cond = None
        probabilities = None
        self.last_sampling_nfe = 0
        bias_weight = self._current_token_bias_weight()
        disable_self_conditioning = bool(getattr(
            self.config.sampling, 'task1_disable_self_conditioning', False))
        closed_same_time_self_conditioning = bool(getattr(
            self.config.sampling,
            'task1_closed_same_time_self_conditioning', False))
        same_time_self_conditioning_iterations = int(getattr(
            self.config.sampling,
            'task1_same_time_self_conditioning_iterations', 2))
        zero_self_conditioning_start_index = getattr(
            self.config.sampling,
            'task1_zero_self_conditioning_start_index', None)
        zero_self_conditioning_length = int(getattr(
            self.config.sampling,
            'task1_zero_self_conditioning_length', 0))
        compare_same_state_self_conditioning = bool(getattr(
            self.config.sampling,
            'task1_compare_same_state_self_conditioning', False))
        compare_same_state_self_conditioning_start_index = getattr(
            self.config.sampling,
            'task1_compare_same_state_self_conditioning_start_index', None)
        if compare_same_state_self_conditioning_start_index is not None:
            compare_same_state_self_conditioning_start_index = int(
                compare_same_state_self_conditioning_start_index)
        clear_self_conditioning_once_index = getattr(
            self.config.sampling,
            'task1_clear_self_conditioning_once_index', None)
        reinitialize_self_conditioning_once_index = getattr(
            self.config.sampling,
            'task1_reinitialize_self_conditioning_once_index', None)
        reinitialize_self_conditioning_iterations = int(getattr(
            self.config.sampling,
            'task1_reinitialize_self_conditioning_iterations', 0))
        sc_gate_d50 = getattr(
            self.config.sampling, 'task1_sc_gate_d50', None)
        if sc_gate_d50 is not None:
            sc_gate_d50 = float(sc_gate_d50)
        if clear_self_conditioning_once_index is not None:
            clear_self_conditioning_once_index = int(
                clear_self_conditioning_once_index)
        if reinitialize_self_conditioning_once_index is not None:
            reinitialize_self_conditioning_once_index = int(
                reinitialize_self_conditioning_once_index)
        if zero_self_conditioning_start_index is not None:
            zero_self_conditioning_start_index = int(
                zero_self_conditioning_start_index)
        if disable_self_conditioning and closed_same_time_self_conditioning:
            raise ValueError(
                'Task1 sampling cannot disable self-conditioning and use '
                'closed same-time self-conditioning simultaneously.')
        if sc_gate_d50 is not None:
            if (disable_self_conditioning
                    or closed_same_time_self_conditioning
                    or zero_self_conditioning_start_index is not None
                    or compare_same_state_self_conditioning
                    or clear_self_conditioning_once_index is not None
                    or reinitialize_self_conditioning_once_index is not None):
                raise ValueError(
                    'Continuousized reference SC cannot be combined with '
                    'other self-conditioning interventions.')
            # Validate without adding any floor to the gate itself.
            finite_sc_gate(torch.zeros((), device=self.device), sc_gate_d50)
        if (closed_same_time_self_conditioning
                and same_time_self_conditioning_iterations < 2):
            raise ValueError(
                'Closed same-time self-conditioning requires at least two '
                'iterations per time point.')
        if zero_self_conditioning_start_index is not None:
            if disable_self_conditioning or closed_same_time_self_conditioning:
                raise ValueError(
                    'A finite self-conditioning reset window requires normal '
                    'cross-time self-conditioning.')
            if (zero_self_conditioning_start_index < 0
                    or zero_self_conditioning_length <= 0
                    or zero_self_conditioning_start_index
                    + zero_self_conditioning_length > num_steps):
                raise ValueError(
                    'Invalid self-conditioning reset window: '
                    f'start={zero_self_conditioning_start_index}, '
                    f'length={zero_self_conditioning_length}, '
                    f'num_steps={num_steps}.')
        one_step_modes = [
            clear_self_conditioning_once_index is not None,
            reinitialize_self_conditioning_once_index is not None]
        if sum(one_step_modes) > 1:
            raise ValueError(
                'Choose either one-step h clearing or one-step local h '
                'reinitialization, not both.')
        if any(one_step_modes):
            selected_index = (
                clear_self_conditioning_once_index
                if clear_self_conditioning_once_index is not None
                else reinitialize_self_conditioning_once_index)
            if (disable_self_conditioning
                    or closed_same_time_self_conditioning
                    or zero_self_conditioning_start_index is not None
                    or compare_same_state_self_conditioning):
                raise ValueError(
                    'One-step self-conditioning interventions require an '
                    'otherwise normal cross-time SC trajectory.')
            if not (0 <= selected_index < num_steps):
                raise ValueError(
                    f'Invalid one-step SC intervention index: '
                    f'{selected_index}.')
        if (reinitialize_self_conditioning_once_index is not None
                and reinitialize_self_conditioning_iterations <= 0):
            raise ValueError(
                'Local self-conditioning reinitialization requires a '
                'positive iteration count.')
        if compare_same_state_self_conditioning:
            if disable_self_conditioning or closed_same_time_self_conditioning:
                raise ValueError(
                    'Same-state self-conditioning comparison requires '
                    'cross-time self-conditioning.')
            if compare_same_state_self_conditioning_start_index is None:
                compare_same_state_self_conditioning_start_index = 0
            if not (0 <= compare_same_state_self_conditioning_start_index
                    < num_steps):
                raise ValueError(
                    'Invalid same-state self-conditioning comparison start: '
                    f'{compare_same_state_self_conditioning_start_index}.')
            if (zero_self_conditioning_start_index is not None
                    and compare_same_state_self_conditioning_start_index
                    != zero_self_conditioning_start_index
                    + zero_self_conditioning_length):
                raise ValueError(
                    'A reset-trajectory comparison must start at the first '
                    'post-window recovery query.')
            self._task1_prepare_same_state_sc_comparison(
                num_steps, t_grid,
                compare_same_state_self_conditioning_start_index,
                zero_self_conditioning_start_index,
                zero_self_conditioning_length)

        update_after_last_query = bool(
            grid_metadata.get('update_after_last_query', False))
        comparison_rows = []
        for index in range(num_steps):
            u_scalar = u_grid[index]
            t_scalar = t_grid[index]
            u = u_scalar.expand(num_samples)
            physical_t = t_scalar.expand(num_samples)
            model_time = flm_model_time_condition(
                u, physical_t, self.model_time_condition, eps).to(self.device)
            if closed_same_time_self_conditioning:
                same_time_self_cond = None
                for _ in range(same_time_self_conditioning_iterations):
                    logits = self._forward_logits(
                        state, model_time, x_self_cond=same_time_self_cond,
                        bias_weight=bias_weight, physical_t=physical_t)
                    self.last_sampling_nfe += 1
                    probabilities = F.softmax(logits.float(), dim=-1)
                    same_time_self_cond = self.embed_probabilities(
                        probabilities).detach()
            else:
                in_zero_window = (
                    zero_self_conditioning_start_index is not None
                    and zero_self_conditioning_start_index <= index
                    < zero_self_conditioning_start_index
                    + zero_self_conditioning_length)
                if in_zero_window:
                    x_self_cond = None
                if index == clear_self_conditioning_once_index:
                    x_self_cond = None
                if index == reinitialize_self_conditioning_once_index:
                    local_self_cond = None
                    for _ in range(
                            reinitialize_self_conditioning_iterations):
                        local_logits = self._forward_logits(
                            state, model_time,
                            x_self_cond=local_self_cond,
                            bias_weight=bias_weight,
                            physical_t=physical_t)
                        self.last_sampling_nfe += 1
                        local_probabilities = F.softmax(
                            local_logits.float(), dim=-1)
                        local_self_cond = self.embed_probabilities(
                            local_probabilities).detach()
                    x_self_cond = local_self_cond
                logits = self._forward_logits(
                    state, model_time, x_self_cond=x_self_cond,
                    bias_weight=bias_weight, physical_t=physical_t)
                self.last_sampling_nfe += 1
            compare_at_index = (
                compare_same_state_self_conditioning
                and index
                >= compare_same_state_self_conditioning_start_index)
            if compare_at_index:
                with_log_probabilities = F.log_softmax(
                    logits.float(), dim=-1)
                probabilities = with_log_probabilities.exp()
                without_logits = self._forward_logits(
                    state, model_time, x_self_cond=None,
                    bias_weight=bias_weight, physical_t=physical_t)
                self.last_sampling_nfe += 1
                without_log_probabilities = F.log_softmax(
                    without_logits.float(), dim=-1)
                without_probabilities = without_log_probabilities.exp()
                kl_with_without = (
                    probabilities * (
                        with_log_probabilities
                        - without_log_probabilities)).sum(dim=-1)
                kl_without_with = (
                    without_probabilities * (
                        without_log_probabilities
                        - with_log_probabilities)).sum(dim=-1)
                total_variation = 0.5 * (
                    probabilities - without_probabilities).abs().sum(dim=-1)
                top1_disagreement = (
                    probabilities.argmax(dim=-1)
                    != without_probabilities.argmax(dim=-1)).float()
                comparison_rows.append((index, torch.stack((
                    kl_with_without.sum(dtype=torch.float64),
                    kl_without_with.sum(dtype=torch.float64),
                    total_variation.sum(dtype=torch.float64),
                    top1_disagreement.sum(dtype=torch.float64)))))
            else:
                probabilities = F.softmax(logits.float(), dim=-1)
            if (not disable_self_conditioning
                    and not closed_same_time_self_conditioning):
                if in_zero_window:
                    x_self_cond = None
                elif sc_gate_d50 is not None:
                    proposal = self.embed_probabilities(
                        probabilities).detach().float()
                    if x_self_cond is None:
                        x_self_cond = torch.zeros_like(
                            proposal, dtype=torch.float32)
                    delta = (t_grid[index + 1] - t_scalar).float()
                    gate = finite_sc_gate(delta, sc_gate_d50).to(
                        proposal.device)
                    update = gate * (proposal - x_self_cond.float())
                    next_self_cond = x_self_cond.float() + update
                    self._task1_accumulate_sc_gate_statistics(
                        x_self_cond, proposal, next_self_cond, update,
                        gate, num_samples, sc_gate_d50)
                    x_self_cond = next_self_cond
                else:
                    x_self_cond = self.embed_probabilities(
                        probabilities).detach()
            if (index + 1 < num_steps or update_after_last_query
                    or sc_gate_d50 is not None):
                state = flm_linear_euler_update(
                    state, probabilities, t_scalar, t_grid[index + 1], eps)
        if compare_same_state_self_conditioning:
            self._task1_accumulate_same_state_sc_comparison(
                comparison_rows, num_samples)
        return probabilities.argmax(dim=-1)

    def _task1_accumulate_sc_gate_statistics(
            self, carried, proposal, output, update, gate,
            num_samples, d50):
        if not hasattr(self, '_task1_sc_gate_statistics_sums'):
            self._task1_sc_gate_statistics_sums = {
                'input_rms': 0.0,
                'proposal_rms': 0.0,
                'proposal_minus_input_rms': 0.0,
                'update_rms': 0.0,
                'output_rms': 0.0,
                'gate': 0.0,
            }
            self._task1_sc_gate_statistics_max = {
                'input_abs_max': 0.0,
                'proposal_abs_max': 0.0,
                'update_abs_max': 0.0,
                'output_abs_max': 0.0,
            }
            self._task1_sc_gate_statistics_count = 0
            self._task1_sc_gate_statistics_sample_count = 0
            self._task1_sc_gate_statistics_d50 = float(d50)
        if self._task1_sc_gate_statistics_d50 != float(d50):
            raise ValueError('SC gate d50 changed between sampling batches.')

        delta = proposal.float() - carried.float()
        values = {
            'input_rms': carried.float().square().mean().sqrt(),
            'proposal_rms': proposal.float().square().mean().sqrt(),
            'proposal_minus_input_rms': delta.square().mean().sqrt(),
            'update_rms': update.float().square().mean().sqrt(),
            'output_rms': output.float().square().mean().sqrt(),
            'gate': gate.float().mean(),
        }
        for name, value in values.items():
            self._task1_sc_gate_statistics_sums[name] += float(value.cpu())
        maxima = {
            'input_abs_max': carried.float().abs().max(),
            'proposal_abs_max': proposal.float().abs().max(),
            'update_abs_max': update.float().abs().max(),
            'output_abs_max': output.float().abs().max(),
        }
        for name, value in maxima.items():
            self._task1_sc_gate_statistics_max[name] = max(
                self._task1_sc_gate_statistics_max[name],
                float(value.cpu()))
        self._task1_sc_gate_statistics_count += 1
        self._task1_sc_gate_statistics_sample_count += int(num_samples)

    def task1_sc_gate_statistics(self):
        if not hasattr(self, '_task1_sc_gate_statistics_sums'):
            return None
        count = self._task1_sc_gate_statistics_count
        d50 = self._task1_sc_gate_statistics_d50
        return {
            'rule': 'h_next=h+b(delta)*(proposal-h)',
            'd50': d50,
            'alpha': float(torch.log(torch.tensor(2.0)) / d50),
            'transition_batch_count': count,
            'transition_sample_count': (
                self._task1_sc_gate_statistics_sample_count),
            'mean_over_transition_batches': {
                name: value / count
                for name, value in
                self._task1_sc_gate_statistics_sums.items()
            },
            'max_over_transition_batches': dict(
                self._task1_sc_gate_statistics_max),
        }

    def _task1_prepare_same_state_sc_comparison(
            self, num_steps, t_grid, comparison_start_index,
            reset_start_index, reset_length):
        physical_times = t_grid[:num_steps].detach().float().cpu()
        if not hasattr(self, '_task1_same_state_sc_comparison_sums'):
            self._task1_same_state_sc_comparison_sums = torch.zeros(
                (num_steps, 4), dtype=torch.float64)
            self._task1_same_state_sc_comparison_counts = torch.zeros(
                num_steps, dtype=torch.int64)
            self._task1_same_state_sc_comparison_sample_count = 0
            self._task1_same_state_sc_comparison_physical_times = (
                physical_times)
            self._task1_same_state_sc_comparison_start_index = int(
                comparison_start_index)
            self._task1_same_state_sc_comparison_reset_start_index = (
                None if reset_start_index is None else int(reset_start_index))
            self._task1_same_state_sc_comparison_reset_length = int(
                reset_length)
            return
        if (len(self._task1_same_state_sc_comparison_physical_times)
                != num_steps
                or not torch.equal(
                    self._task1_same_state_sc_comparison_physical_times,
                    physical_times)
                or self._task1_same_state_sc_comparison_start_index
                != int(comparison_start_index)):
            raise ValueError(
                'Same-state self-conditioning comparison grid changed '
                'between sampling batches.')

    def _task1_accumulate_same_state_sc_comparison(
            self, comparison_rows, num_samples):
        indices = torch.tensor(
            [index for index, _ in comparison_rows], dtype=torch.int64)
        rows = torch.stack(
            [row for _, row in comparison_rows]).detach().cpu()
        self._task1_same_state_sc_comparison_sums[indices] += rows
        posterior_sites = int(num_samples) * int(self.num_tokens)
        self._task1_same_state_sc_comparison_counts[indices] += posterior_sites
        self._task1_same_state_sc_comparison_sample_count += int(num_samples)

    def task1_same_state_sc_posterior_comparison(self):
        if not hasattr(self, '_task1_same_state_sc_comparison_sums'):
            return None
        metric_names = (
            'kl_with_h_to_h_zero_nats',
            'kl_h_zero_to_with_h_nats',
            'total_variation',
            'top1_disagreement_rate')
        sums = self._task1_same_state_sc_comparison_sums
        counts = self._task1_same_state_sc_comparison_counts
        rows = []
        for index in range(len(counts)):
            count = int(counts[index])
            if count == 0:
                continue
            row = {
                'query_index_zero_based': index,
                'physical_t': float(
                    self._task1_same_state_sc_comparison_physical_times[
                        index]),
                'posterior_site_count': count,
            }
            row.update({
                name: float(sums[index, metric_index] / count)
                for metric_index, name in enumerate(metric_names)})
            rows.append(row)
        total_count = int(counts.sum())
        total_sums = sums.sum(dim=0)
        return {
            'definition': (
                'After the configured h=0 reset window, follow that arm\'s '
                'recovered cross-time self-conditioning trajectory and, at '
                'each fixed x_i,t_i, compare T(x_i,t_i,h_i) with '
                'T(x_i,t_i,0); only the with-h posterior advances h and x.'),
            'sample_count': int(
                self._task1_same_state_sc_comparison_sample_count),
            'time_point_count': len(rows),
            'comparison_start_query_index_zero_based': int(
                self._task1_same_state_sc_comparison_start_index),
            'reset_start_query_index_zero_based': (
                self._task1_same_state_sc_comparison_reset_start_index),
            'reset_length': int(
                self._task1_same_state_sc_comparison_reset_length),
            'sequence_length': int(self.num_tokens),
            'posterior_site_count': total_count,
            'trajectory_backbone_nfe_per_sample': len(counts),
            'diagnostic_backbone_nfe_per_sample': len(rows),
            'total_backbone_nfe_per_sample': len(counts) + len(rows),
            'aggregate': {
                name: float(total_sums[metric_index] / total_count)
                for metric_index, name in enumerate(metric_names)},
            'per_time_point': rows,
        }


    def _task1_vocab_sampling_grid(self, num_steps):
        mode = str(getattr(
            self.config.sampling, 'task1_time_grid', 'uniform_tau'))
        if mode == 'learned_gumbel_equal_cumulative_progress':
            if not getattr(self, 'learned_gumbel', False):
                raise ValueError(
                    'Pure-Gumbel inference requires a learned-Gumbel '
                    'checkpoint configuration.')
            if num_steps < 2:
                raise ValueError(
                    'Pure-Gumbel inference requires at least two queries.')
            t_end = torch.tensor(
                511.0 / 512.0, device=self.device, dtype=torch.float32)
            gamma_end = flm_linear_gamma(
                t_end[None], self.flm_time_eps)[0]
            _, loc, scale = self._learned_gumbel_values()
            loc = loc.detach().to(device=self.device, dtype=torch.float32)
            scale = scale.detach().to(
                device=self.device, dtype=torch.float32)
            cdf_end = torch.exp(
                -torch.exp(-(gamma_end - loc) / scale))
            progress = torch.arange(
                num_steps, device=self.device, dtype=torch.float32)
            progress.div_(float(num_steps - 1))
            q_query = 1.0 - progress * (1.0 - cdf_end)
            gamma_query = self._learned_gumbel_icdf(q_query).detach()
            t_query = torch.sigmoid(-0.5 * gamma_query)
            t_query[0] = 0.0
            t_query[-1] = t_end
            if not bool(torch.all(t_query[1:] > t_query[:-1])):
                raise RuntimeError(
                    'Pure-Gumbel physical-time queries must increase.')
            t_grid = torch.cat((t_query, t_query.new_ones(1)))
            # The model consumes physical t through log-NSR in this mode.
            # Keep a same-shaped placeholder without invoking the tau LUT.
            u_grid = t_grid.clone()
            return u_grid, t_grid, {
                'update_after_last_query': False,
                'time_grid_schedule': (
                    'learned_gumbel_equal_cumulative_progress'),
                'gumbel_cumulative_progress_uniform': True,
                'tau_lut_used': False,
                'query_count': int(num_steps),
                'update_count': int(num_steps - 1),
                'final_query_physical_time': float(t_end),
                'endpoint_update_performed': False,
                'initial_noise_scale': 1.0,
                'scheduler_parameters': self.learned_gumbel_state(),
            }
        if mode == 'learned_gumbel_quantiles':
            if not getattr(self, 'learned_gumbel', False):
                raise ValueError(
                    'Learned-Gumbel inference requires a learned-Gumbel '
                    'checkpoint configuration.')
            q_query = torch.linspace(
                1.0 - self.gumbel_cutoff,
                self.gumbel_cutoff,
                num_steps,
                device=self.device,
                dtype=torch.float32)
            gamma_query = self._learned_gumbel_icdf(q_query).detach()
            t_query = torch.sigmoid(-0.5 * gamma_query)
            t_grid = torch.cat((t_query, t_query.new_ones(1)))
            u_grid = t_grid.clone()
            return u_grid, t_grid, {
                'update_after_last_query': False,
                'time_grid_schedule': 'learned_gumbel_quantiles',
                'initial_noise_scale': float(1.0 - t_query[0]),
                'scheduler_parameters': self.learned_gumbel_state(),
            }
        if mode == 'custom_physical_time_nodes':
            path = str(getattr(
                self.config.sampling, 'task1_custom_time_grid_path', ''))
            schedule = str(getattr(
                self.config.sampling, 'task1_custom_time_grid_schedule', ''))
            if not path or path == 'None' or not schedule or schedule == 'None':
                raise ValueError(
                    'Custom Task1 time grid requires a path and schedule.')
            with open(path, encoding='utf-8') as handle:
                payload = json.load(handle)
            try:
                t_query = payload['schedules'][schedule]['t_query']
            except KeyError as error:
                raise ValueError(
                    f'Unknown custom Task1 schedule: {schedule!r}.') from error
            if len(t_query) != num_steps:
                raise ValueError(
                    'Custom Task1 query count does not match requested NFE: '
                    f'{len(t_query)} != {num_steps}.')
            t_grid = torch.tensor(
                [*t_query, float(payload.get('endpoint_physical_t', 1.0))],
                device=self.device, dtype=torch.float32)
            if float(t_grid[0]) != 0.0 or float(t_grid[-2]) >= 1.0:
                raise ValueError(
                    'Custom Task1 grid must start at zero and query below one.')
            if not bool(torch.all(t_grid[1:] > t_grid[:-1])):
                raise ValueError(
                    'Custom Task1 physical-time nodes must be strictly increasing.')
            u_grid = self._t_to_tau(t_grid).float().clamp(0.0, 1.0)
            u_grid[0] = 0.0
            u_grid[-1] = 1.0
            return u_grid, t_grid, {
                'update_after_last_query': False,
                'time_grid_schedule': schedule,
                'tau_pairing': payload.get('tau_pairing'),
            }
        if mode == 'physical_time_uniform_official_inverse_lut':
            t_grid = torch.linspace(
                0.0, 1.0, num_steps + 1,
                device=self.device, dtype=torch.float32)
            u_grid = self._t_to_tau(t_grid).float().clamp(0.0, 1.0)
            u_grid[0] = 0.0
            u_grid[-1] = 1.0
            return u_grid, t_grid, {'update_after_last_query': False}
        u_grid = (
            torch.arange(
                num_steps + 1, device=self.device, dtype=torch.float32)
            / float(num_steps))
        return (
            u_grid,
            self._task1_physical_time(u_grid),
            {'update_after_last_query': False})
