"""Fixed-NFE Euler-EDM generation for the LangFlow-FLM hybrid."""

import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .ops import (
    flm_linear_euler_update,
    flm_model_time_condition,
    langflow_euler_edm_update,
    langflow_gumbel_gamma,
)
from .task1_trajectory import Task1TrajectoryDiagnostics


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
        self._task1_sampling_batch_index = batch_index + 1
        x_self_cond = None
        probabilities = None
        self.last_sampling_nfe = 0
        bias_weight = self._current_token_bias_weight()
        experiment = getattr(self.config, 'experiment', None)
        trace_enabled = bool(getattr(
            experiment, 'task1_contract_smoke', False))
        initial_state_probe = (
            state.detach().float().flatten()[:8].cpu().tolist()
            if trace_enabled else None)
        self._task1_initialize_sampling_diagnostics(
            grid_metadata, state, batch_index, initial_noise_seed,
            initial_noise_schedule)
        trajectory_diagnostics = self._task1_initialize_trajectory_diagnostics(
            grid_metadata, state, initial_noise_seed,
            initial_noise_schedule)
        trajectory = []

        update_after_last_query = bool(
            grid_metadata.get('update_after_last_query', False))
        for index in range(num_steps):
            if trajectory_diagnostics is not None:
                trajectory_diagnostics.arrive_at_fine_state(index, state)
            u_scalar = u_grid[index]
            t_scalar = t_grid[index]
            u = u_scalar.expand(num_samples)
            physical_t = t_scalar.expand(num_samples)
            model_time = flm_model_time_condition(
                u, physical_t, self.model_time_condition, eps).to(self.device)
            self_conditioning_input = x_self_cond
            logits = self._forward_logits(
                state, model_time, x_self_cond=x_self_cond,
                bias_weight=bias_weight, physical_t=physical_t)
            self.last_sampling_nfe += 1
            probabilities = F.softmax(logits.float(), dim=-1)
            x_self_cond = self.embed_probabilities(probabilities).detach()
            if trajectory_diagnostics is not None:
                fp32_probabilities = None
                if trajectory_diagnostics.should_run_fp32(index):
                    fp32_count = (
                        trajectory_diagnostics.fp32_local_sample_count())
                    with torch.autocast(
                            device_type=self.device.type, enabled=False):
                        fp32_logits = self._forward_logits(
                            state[:fp32_count].float(),
                            model_time[:fp32_count].float(),
                            x_self_cond=(
                                None if self_conditioning_input is None else
                                self_conditioning_input[:fp32_count].float()),
                            bias_weight=bias_weight,
                            physical_t=physical_t[:fp32_count].float())
                        fp32_probabilities = F.softmax(
                            fp32_logits.float(), dim=-1)
                trajectory_diagnostics.observe_prediction(
                    index, state, probabilities, t_scalar, eps,
                    flm_linear_euler_update,
                    self_conditioning=self_conditioning_input,
                    fp32_probabilities=fp32_probabilities)
            if trace_enabled:
                safe_t = t_scalar.detach().float().clamp(
                    0.0, 1.0 - eps)
                trajectory.append({
                    'index': index,
                    'tau': float(u_scalar.detach().cpu()),
                    'physical_time': float(t_scalar.detach().cpu()),
                    'model_time_condition': float(
                        model_time[0].detach().cpu()),
                    'bias_coefficient': float(
                        (float(bias_weight) * safe_t
                         / (1.0 - safe_t).square()).cpu()),
                    'state_finite': bool(torch.isfinite(state).all().cpu()),
                    'posterior_finite': bool(
                        torch.isfinite(probabilities).all().cpu()),
                })
            if (index + 1 < num_steps or update_after_last_query
                    or trajectory_diagnostics is not None):
                state = flm_linear_euler_update(
                    state, probabilities, t_scalar, t_grid[index + 1], eps)
            self._task1_record_sampling_diagnostic(
                index + 1, state, probabilities)

        if trajectory_diagnostics is not None:
            trajectory_diagnostics.finish_batch(state)

        if self.last_sampling_nfe != num_steps:
            raise RuntimeError(
                f'Backbone NFE mismatch: {self.last_sampling_nfe} != '
                f'{num_steps}.')
        if trace_enabled and int(getattr(self, 'global_rank', 0)) == 0:
            samples_path = Path(str(self.config.eval.generated_samples_path))
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            trace = {
                'variant': self.model_time_condition,
                'nfe': self.last_sampling_nfe,
                'state_shape': [
                    num_samples, self.num_tokens, self.vocab_size],
                'initial_state_probe': initial_state_probe,
                'trajectory': trajectory,
                'endpoint_used_last_clean_posterior': True,
                'endpoint_velocity_division_skipped': True,
            }
            (samples_path.parent / 'task1_sampling_trace.json').write_text(
                json.dumps(trace, indent=2), encoding='utf-8')
        return probabilities.argmax(dim=-1)

    def _task1_vocab_sampling_grid(self, num_steps):
        """Build the official grid or a fixed-NFE first-interval split."""
        base_u = (
            torch.arange(
                num_steps + 1, device=self.device, dtype=torch.float32)
            / float(num_steps))
        base_t = self._task1_physical_time(base_u)
        mode = str(getattr(
            self.config.sampling, 'task1_time_grid', 'uniform_tau'))
        if mode == 'uniform_tau':
            return base_u, base_t, {
                'name': mode,
                'nfe': num_steps,
                'first_interval_substeps': 1,
                'removed_uniform_tau_indices': [],
            }
        if mode == 'tau_box_physical_equal_0p6_jump':
            return self._task1_tau_box_physical_equal_grid(num_steps)
        if mode == 'first_interval32_tail_bins_balanced':
            return self._task1_first_interval_tail_balanced_grid(num_steps)
        if mode == 'first_interval64_tail_keep_4_3_2_1':
            return self._task1_first_interval64_tail_keep_grid(num_steps)
        if mode == 'physical_time_uniform_official_inverse_lut':
            return self._task1_physical_time_uniform_grid(num_steps)
        if mode != 'first_interval_physical_split':
            raise ValueError(f'Unsupported Task1 sampling grid: {mode!r}.')

        substeps = int(getattr(
            self.config.sampling, 'task1_first_interval_substeps', 1))
        if substeps < 2 or num_steps <= substeps + 2:
            raise ValueError(
                'The first-interval split requires at least 2 substeps and '
                'enough later nodes to preserve the final prediction point.')
        early_t = torch.linspace(
            0.0, float(base_t[1].detach().cpu()), substeps + 1,
            device=self.device, dtype=torch.float32)
        early_u = self._t_to_tau(early_t).float().clamp(0.0, 1.0)
        early_u[0] = base_u[0]
        early_u[-1] = base_u[1]
        u_grid = torch.cat((
            early_u,
            base_u[2:num_steps - substeps],
            base_u[-2:]))
        t_grid = torch.cat((
            early_t,
            base_t[2:num_steps - substeps],
            base_t[-2:]))
        if u_grid.numel() != num_steps + 1:
            raise RuntimeError(
                f'Task1 grid has {u_grid.numel() - 1} steps, expected '
                f'{num_steps}.')
        if not bool(torch.all(t_grid[1:] > t_grid[:-1]).cpu()):
            raise RuntimeError('Task1 sampling grid must increase in physical time.')
        removed = list(range(num_steps - substeps, num_steps - 1))
        return u_grid, t_grid, {
            'name': mode,
            'nfe': num_steps,
            'first_interval_substeps': substeps,
            'first_uniform_tau': float(base_u[1].detach().cpu()),
            'first_uniform_physical_time': float(base_t[1].detach().cpu()),
            'removed_uniform_tau_indices': removed,
            'last_prediction_tau': float(u_grid[-2].detach().cpu()),
            'last_prediction_physical_time': float(t_grid[-2].detach().cpu()),
            'tau_points': u_grid.detach().cpu().tolist(),
            'physical_time_points': t_grid.detach().cpu().tolist(),
        }

    def _task1_tau_box_physical_equal_grid(self, num_steps):
        """Build the frozen 127-plus-one tau-box diagnostic grid.

        The first six tau deciles own 127 query starts.  Points inside each
        decile are equally spaced in physical time and mapped back through the
        official LUT.  Query 128 starts exactly at tau=0.6 and updates directly
        to the endpoint, so this diagnostic has 128 queries and 128 updates.
        """
        configured = list(getattr(
            self.config.sampling, 'task1_tau_box_query_counts', []))
        query_counts = [int(value) for value in configured]
        expected_counts = [32, 23, 23, 24, 12, 13, 1]
        if query_counts != expected_counts:
            raise ValueError(
                'tau_box_physical_equal_0p6_jump requires query counts '
                f'{expected_counts}, got {query_counts}.')
        if num_steps != sum(query_counts):
            raise ValueError(
                'tau_box_physical_equal_0p6_jump requires exactly 128 NFE, '
                f'got {num_steps}.')

        tau_edges = torch.arange(
            7, device=self.device, dtype=torch.float32) / 10.0
        physical_edges = self._task1_physical_time(tau_edges)
        query_tau_parts = []
        query_t_parts = []
        query_box_indices = []
        for box_index, count in enumerate(query_counts[:6]):
            fractions = (
                torch.arange(
                    count, device=self.device, dtype=torch.float32)
                / float(count))
            physical = (
                physical_edges[box_index]
                + fractions * (
                    physical_edges[box_index + 1]
                    - physical_edges[box_index]))
            tau = self._t_to_tau(physical).float().clamp(0.0, 1.0)
            tau[0] = tau_edges[box_index]
            query_tau_parts.append(tau)
            query_t_parts.append(physical)
            query_box_indices.extend([box_index] * count)

        final_query_tau = tau_edges[-1:].clone()
        final_query_t = physical_edges[-1:].clone()
        query_tau_parts.append(final_query_tau)
        query_t_parts.append(final_query_t)
        query_box_indices.append(6)
        query_tau = torch.cat(query_tau_parts)
        query_t = torch.cat(query_t_parts)
        endpoint_tau = query_tau.new_tensor([1.0])
        endpoint_t = query_t.new_tensor([1.0])
        u_grid = torch.cat((query_tau, endpoint_tau))
        t_grid = torch.cat((query_t, endpoint_t))

        if query_tau.numel() != num_steps:
            raise RuntimeError(
                f'Task1 tau-box grid has {query_tau.numel()} queries, '
                f'expected {num_steps}.')
        if not bool(torch.all(t_grid[1:] > t_grid[:-1]).cpu()):
            raise RuntimeError(
                'Task1 tau-box physical-time points must strictly increase.')
        return u_grid, t_grid, {
            'name': 'tau_box_physical_equal_0p6_jump',
            'nfe': num_steps,
            'update_count': num_steps,
            'update_after_last_query': True,
            'tau_box_edges': [index / 10.0 for index in range(11)],
            'query_count_by_tau_decile': [
                32, 23, 23, 24, 12, 13, 1, 0, 0, 0],
            'query_counts_first_seven_boxes': query_counts,
            'query_box_indices': query_box_indices,
            'query_tau_points': query_tau.detach().cpu().tolist(),
            'query_physical_time_points': query_t.detach().cpu().tolist(),
            'update_target_tau_points': u_grid[1:].detach().cpu().tolist(),
            'update_target_physical_time_points': (
                t_grid[1:].detach().cpu().tolist()),
            'tau_points': u_grid.detach().cpu().tolist(),
            'physical_time_points': t_grid.detach().cpu().tolist(),
            'final_query_index_zero_based': num_steps - 1,
            'final_query_tau': float(final_query_tau.item()),
            'final_query_physical_time': float(final_query_t.item()),
            'final_update_target_tau': 1.0,
            'final_update_target_physical_time': 1.0,
        }

    def _task1_first_interval_tail_balanced_grid(self, num_steps):
        """Build G32 at 128 NFE, then refine every physical-time step."""
        removal_counts = [int(value) for value in getattr(
            self.config.sampling, 'task1_tail_bin_removal_counts', [])]
        expected_removals = [8, 8, 8, 7]
        if num_steps not in (128, 256, 512):
            raise ValueError(
                'first_interval32_tail_bins_balanced requires '
                '128, 256, or 512 NFE.')
        if removal_counts != expected_removals:
            raise ValueError(
                'first_interval32_tail_bins_balanced requires tail removal '
                f'counts {expected_removals}, got {removal_counts}.')

        base_steps = 128
        refinement_factor = num_steps // base_steps
        base_u = (
            torch.arange(
                base_steps + 1, device=self.device, dtype=torch.float32)
            / float(base_steps))
        base_t = self._task1_physical_time(base_u)
        early_t = torch.linspace(
            0.0, float(base_t[1].detach().cpu()), 33,
            device=self.device, dtype=torch.float32)
        early_u = self._t_to_tau(early_t).float().clamp(0.0, 1.0)
        early_u[0] = base_u[0]
        early_u[-1] = base_u[1]

        removed_by_box = []
        removed = set()
        for offset, count in enumerate(removal_counts):
            decile = 6 + offset
            start = (decile * base_steps + 9) // 10
            stop = ((decile + 1) * base_steps + 9) // 10
            candidates = list(range(start, min(stop, base_steps)))
            if decile == 9:
                candidates.remove(base_steps - 1)
            positions = [
                ((2 * index + 1) * len(candidates)) // (2 * count)
                for index in range(count)]
            selected = [candidates[position] for position in positions]
            if len(set(selected)) != count:
                raise RuntimeError('Balanced tail removal produced duplicates.')
            removed.update(selected)
            removed_by_box.append({
                'tau_decile': decile + 1,
                'removal_count': count,
                'candidate_uniform_tau_indices': candidates,
                'removed_uniform_tau_indices': selected,
            })

        tail_indices = [
            index for index in range(2, base_steps + 1)
            if index not in removed]
        base_g32_u = torch.cat((early_u, base_u[tail_indices]))
        base_g32_t = torch.cat((early_t, base_t[tail_indices]))
        if base_g32_u.numel() != base_steps + 1:
            raise RuntimeError(
                'Balanced G32 base grid does not have 128 intervals.')

        u_grid = base_g32_u
        t_grid = base_g32_t
        refinement_levels = [base_steps]
        immediate_parent_u = None
        immediate_parent_t = None
        current_steps = base_steps
        while current_steps < num_steps:
            immediate_parent_u = u_grid
            immediate_parent_t = t_grid
            midpoint_t = (t_grid[:-1] + t_grid[1:]) / 2.0
            refined_t = torch.empty(
                current_steps * 2 + 1, device=self.device,
                dtype=torch.float32)
            refined_t[0::2] = t_grid
            refined_t[1::2] = midpoint_t
            refined_u = self._t_to_tau(
                refined_t).float().clamp(0.0, 1.0)
            refined_u[0::2] = u_grid
            u_grid = refined_u
            t_grid = refined_t
            current_steps *= 2
            refinement_levels.append(current_steps)

        if u_grid.numel() != num_steps + 1:
            raise RuntimeError(
                f'Refined G32 grid has {u_grid.numel() - 1} intervals, '
                f'expected {num_steps}.')
        if not bool(torch.all(t_grid[1:] > t_grid[:-1]).cpu()):
            raise RuntimeError(
                'Balanced first-interval physical-time points must increase.')
        query_tau = u_grid[:-1]
        query_counts_by_decile = [int((
            (query_tau >= decile / 10.0)
            & (query_tau < (decile + 1) / 10.0)
        ).sum().cpu()) for decile in range(10)]
        return u_grid, t_grid, {
            'name': 'first_interval32_tail_bins_balanced',
            'nfe': num_steps,
            'update_count': num_steps - 1,
            'update_after_last_query': False,
            'endpoint_update_performed': False,
            'base_grid_nfe': base_steps,
            'refinement_factor': refinement_factor,
            'refinement_levels_nfe': refinement_levels,
            'refinement_coordinate': 'physical_time',
            'base_grid_points_preserved': True,
            'base_grid_tau_points': base_g32_u.detach().cpu().tolist(),
            'base_grid_physical_time_points': (
                base_g32_t.detach().cpu().tolist()),
            'immediate_parent_nfe': (
                None if immediate_parent_u is None else num_steps // 2),
            'immediate_parent_tau_points': (
                None if immediate_parent_u is None
                else immediate_parent_u.detach().cpu().tolist()),
            'immediate_parent_physical_time_points': (
                None if immediate_parent_t is None
                else immediate_parent_t.detach().cpu().tolist()),
            'first_interval_substeps': 32 * refinement_factor,
            'first_uniform_tau': 1.0 / base_steps,
            'first_uniform_physical_time': float(base_t[1].detach().cpu()),
            'tail_tau_deciles': [7, 8, 9, 10],
            'tail_removal_counts': removal_counts,
            'tail_removal_selection': (
                'floor((j+0.5)*candidate_count/removal_count)'),
            'removed_by_tau_decile': removed_by_box,
            'removed_uniform_tau_indices': sorted(removed),
            'preserved_last_uniform_query_index': base_steps - 1,
            'preserved_last_uniform_query_refined_index': (
                (base_steps - 1) * refinement_factor),
            'last_prediction_tau': float(u_grid[-2].detach().cpu()),
            'last_prediction_physical_time': float(t_grid[-2].detach().cpu()),
            'query_count_by_tau_decile': query_counts_by_decile,
            'query_tau_points': query_tau.detach().cpu().tolist(),
            'query_physical_time_points': t_grid[:-1].detach().cpu().tolist(),
            'update_target_tau_points': u_grid[1:num_steps].detach().cpu().tolist(),
            'update_target_physical_time_points': (
                t_grid[1:num_steps].detach().cpu().tolist()),
            'tau_points': u_grid.detach().cpu().tolist(),
            'physical_time_points': t_grid.detach().cpu().tolist(),
        }

    def _task1_first_interval64_tail_keep_grid(self, num_steps):
        """Split interval one 64 ways and freeze later decile budgets.

        The original uniform-tau queries outside the first interval are
        deterministically subsampled.  Deciles two through six share the
        remaining 42-query budget as 9/9/8/8/8, while deciles seven through
        ten retain 4/3/2/1 queries.  The final uniform query at 127/128 is
        always retained.  As in the existing official and first-interval
        evaluators, the final query produces the returned tokens and is not
        followed by an endpoint update.
        """
        if num_steps != 128:
            raise ValueError(
                'first_interval64_tail_keep_4_3_2_1 requires 128 NFE.')

        target_counts = [76, 9, 9, 8, 8, 8, 4, 3, 2, 1]
        base_u = (
            torch.arange(
                num_steps + 1, device=self.device, dtype=torch.float32)
            / float(num_steps))
        base_t = self._task1_physical_time(base_u)
        early_t = torch.linspace(
            0.0, float(base_t[1].detach().cpu()), 65,
            device=self.device, dtype=torch.float32)
        early_u = self._t_to_tau(early_t).float().clamp(0.0, 1.0)
        early_u[0] = base_u[0]
        early_u[-1] = base_u[1]

        kept_by_box = []
        kept = list(range(2, (num_steps + 9) // 10))
        removed = []
        for decile in range(1, 10):
            start = (decile * num_steps + 9) // 10
            stop = ((decile + 1) * num_steps + 9) // 10
            candidates = list(range(start, min(stop, num_steps)))
            keep_count = target_counts[decile]
            if decile == 9:
                selected = [num_steps - 1]
                selection = 'preserve_final_uniform_query'
            else:
                positions = [
                    ((2 * index + 1) * len(candidates))
                    // (2 * keep_count)
                    for index in range(keep_count)]
                selected = [candidates[position] for position in positions]
                selection = (
                    'floor((j+0.5)*candidate_count/keep_count)')
            if len(set(selected)) != keep_count:
                raise RuntimeError(
                    'Deterministic decile query selection produced duplicates.')
            selected_set = set(selected)
            deleted = [
                index for index in candidates if index not in selected_set]
            kept.extend(selected)
            removed.extend(deleted)
            kept_by_box.append({
                'tau_decile': decile + 1,
                'target_query_count': keep_count,
                'candidate_uniform_tau_indices': candidates,
                'kept_uniform_tau_indices': selected,
                'removed_uniform_tau_indices': deleted,
                'selection': selection,
            })

        kept = sorted(kept)
        removed = sorted(removed)
        u_grid = torch.cat((early_u, base_u[kept], base_u[-1:]))
        t_grid = torch.cat((early_t, base_t[kept], base_t[-1:]))
        if u_grid.numel() != num_steps + 1:
            raise RuntimeError(
                f'First-interval-64 grid has {u_grid.numel() - 1} '
                f'queries, expected {num_steps}.')
        if len(removed) != 63:
            raise RuntimeError(
                f'First-interval-64 grid removed {len(removed)} uniform '
                'queries, expected 63.')
        if not bool(torch.all(t_grid[1:] > t_grid[:-1]).cpu()):
            raise RuntimeError(
                'First-interval-64 physical-time points must increase.')

        query_tau = u_grid[:-1]
        query_t = t_grid[:-1]
        query_counts = [int((
            (query_tau >= decile / 10.0)
            & (query_tau < (decile + 1) / 10.0)
        ).sum().cpu()) for decile in range(10)]
        if query_counts != target_counts:
            raise RuntimeError(
                f'First-interval-64 decile counts {query_counts} do not '
                f'match {target_counts}.')
        return u_grid, t_grid, {
            'name': 'first_interval64_tail_keep_4_3_2_1',
            'nfe': num_steps,
            'update_count': num_steps - 1,
            'update_after_last_query': False,
            'endpoint_update_performed': False,
            'first_interval_substeps': 64,
            'query_count_by_tau_decile': query_counts,
            'target_query_count_by_tau_decile': target_counts,
            'later_decile_selection': (
                'deterministic_midpoint; final decile preserves index 127'),
            'kept_by_tau_decile': kept_by_box,
            'kept_uniform_tau_indices_after_first_interval': kept,
            'removed_uniform_tau_indices': removed,
            'preserved_last_uniform_query_index': num_steps - 1,
            'query_tau_points': query_tau.detach().cpu().tolist(),
            'query_physical_time_points': query_t.detach().cpu().tolist(),
            'update_target_tau_points': (
                u_grid[1:num_steps].detach().cpu().tolist()),
            'update_target_physical_time_points': (
                t_grid[1:num_steps].detach().cpu().tolist()),
            'final_query_index_zero_based': num_steps - 1,
            'final_query_tau': float(query_tau[-1].detach().cpu()),
            'final_query_physical_time': float(query_t[-1].detach().cpu()),
            'last_update_source_tau': float(query_tau[-2].detach().cpu()),
            'last_update_source_physical_time': float(
                query_t[-2].detach().cpu()),
            'last_update_target_tau': float(query_tau[-1].detach().cpu()),
            'last_update_target_physical_time': float(
                query_t[-1].detach().cpu()),
            'endpoint_tau': 1.0,
            'endpoint_physical_time': 1.0,
            'tau_points': u_grid.detach().cpu().tolist(),
            'physical_time_points': t_grid.detach().cpu().tolist(),
        }

    def _task1_physical_time_uniform_grid(self, num_steps):
        """Build uniform physical-time intervals with LUT-paired tau."""
        if num_steps not in (128, 256, 512):
            raise ValueError(
                'physical_time_uniform_official_inverse_lut requires '
                '128, 256, or 512 NFE.')
        t_grid = torch.linspace(
            0.0, 1.0, num_steps + 1,
            device=self.device, dtype=torch.float32)
        u_grid = self._t_to_tau(t_grid).float().clamp(0.0, 1.0)
        u_grid[0] = 0.0
        u_grid[-1] = 1.0
        midpoint_t = (t_grid[:-1] + t_grid[1:]) * 0.5
        midpoint_u = self._t_to_tau(midpoint_t).float().clamp(0.0, 1.0)

        query_tau = u_grid[:-1]
        query_t = t_grid[:-1]
        query_counts = [int((
            (query_tau >= decile / 10.0)
            & (query_tau < (decile + 1) / 10.0)
        ).sum().cpu()) for decile in range(10)]
        return u_grid, t_grid, {
            'name': 'physical_time_uniform_official_inverse_lut',
            'nfe': num_steps,
            'update_count': num_steps - 1,
            'update_after_last_query': False,
            'endpoint_update_performed': False,
            'physical_time_grid': f'linspace(0,1,{num_steps + 1})',
            'physical_time_step': 1.0 / num_steps,
            'tau_pairing': 'official_inverse_lut_gamma_to_alpha',
            'query_count_by_tau_decile': query_counts,
            'query_tau_points': query_tau.detach().cpu().tolist(),
            'query_physical_time_points': query_t.detach().cpu().tolist(),
            'update_target_tau_points': (
                u_grid[1:num_steps].detach().cpu().tolist()),
            'update_target_physical_time_points': (
                t_grid[1:num_steps].detach().cpu().tolist()),
            'final_query_index_zero_based': num_steps - 1,
            'final_query_tau': float(query_tau[-1].detach().cpu()),
            'final_query_physical_time': float(query_t[-1].detach().cpu()),
            'last_update_source_tau': float(query_tau[-2].detach().cpu()),
            'last_update_source_physical_time': float(
                query_t[-2].detach().cpu()),
            'last_update_target_tau': float(query_tau[-1].detach().cpu()),
            'last_update_target_physical_time': float(
                query_t[-1].detach().cpu()),
            'endpoint_tau': 1.0,
            'endpoint_physical_time': 1.0,
            'tau_points': u_grid.detach().cpu().tolist(),
            'physical_time_points': t_grid.detach().cpu().tolist(),
            'interval_mid_physical_time_points': (
                midpoint_t.detach().cpu().tolist()),
            'interval_mid_tau_points': midpoint_u.detach().cpu().tolist(),
        }

    def _task1_initialize_sampling_diagnostics(
            self, grid_metadata, initial_state, batch_index,
            initial_noise_seed, initial_noise_schedule):
        steps = tuple(sorted({int(step) for step in getattr(
            self.config.sampling, 'task1_diagnostic_steps', [])}))
        if not steps:
            return
        if not hasattr(self, '_task1_sampling_diagnostic_accumulators'):
            token_pieces = self.tokenizer.convert_ids_to_tokens(
                list(range(self.vocab_size)))
            self._task1_sampling_category_masks = {
                'slash': torch.tensor(
                    ['/' in piece for piece in token_pieces],
                    device=self.device, dtype=torch.bool),
                'digit': torch.tensor(
                    [any('0' <= char <= '9' for char in piece)
                     for piece in token_pieces],
                    device=self.device, dtype=torch.bool),
            }
            self._task1_sampling_diagnostic_accumulators = {
                step: {
                    'state_token_counts': torch.zeros(
                        self.vocab_size, dtype=torch.long),
                    'posterior_token_counts': torch.zeros(
                        self.vocab_size, dtype=torch.long),
                    'token_count': 0,
                    'state_slash_count': 0,
                    'state_digit_count': 0,
                    'posterior_slash_count': 0,
                    'posterior_digit_count': 0,
                    'state_l2_sum': 0.0,
                    'predictive_entropy_sum': 0.0,
                    'posterior_max_probability_sum': 0.0,
                }
                for step in steps
            }
            self._task1_sampling_diagnostic_metadata = {
                'grid': grid_metadata,
                'diagnostic_steps': list(steps),
                'initial_noise_seed': (
                    None if initial_noise_seed is None
                    else int(initial_noise_seed)),
                'initial_noise_schedule': initial_noise_schedule,
                'initial_state_probe': initial_state.detach().flatten()[
                    :8].float().cpu().tolist(),
                'initial_state_shape_per_batch': list(initial_state.shape),
                'batches': 0,
            }
        self._task1_sampling_diagnostic_metadata['batches'] = batch_index + 1

    def _task1_initialize_trajectory_diagnostics(
            self, grid_metadata, initial_state, initial_noise_seed,
            initial_noise_schedule):
        enabled = bool(getattr(
            self.config.sampling, 'task1_trajectory_diagnostics', False))
        if not enabled:
            return None
        if initial_noise_seed is None:
            raise ValueError(
                'Trajectory diagnostics require an explicit initial-noise seed.')
        diagnostics = getattr(
            self, '_task1_trajectory_diagnostic_accumulator', None)
        if diagnostics is None:
            diagnostics = Task1TrajectoryDiagnostics(
                grid_metadata=grid_metadata,
                initial_state=initial_state,
                initial_noise_seed=initial_noise_seed,
                initial_noise_schedule=initial_noise_schedule,
                checkpoint_label=getattr(
                    self.config.sampling,
                    'task1_trajectory_checkpoint_label', ''),
                top_k=getattr(
                    self.config.sampling, 'task1_trajectory_top_k', 32),
                fp32_sample_count=getattr(
                    self.config.sampling,
                    'task1_trajectory_fp32_sample_count', 8),
                fp32_node_count=getattr(
                    self.config.sampling,
                    'task1_trajectory_fp32_node_count', 17))
            self._task1_trajectory_diagnostic_accumulator = diagnostics
        diagnostics.start_batch(initial_state)
        return diagnostics

    def task1_finalize_trajectory_diagnostics(self):
        diagnostics = getattr(
            self, '_task1_trajectory_diagnostic_accumulator', None)
        return None if diagnostics is None else diagnostics.finalize()

    def _task1_record_sampling_diagnostic(
            self, step, state, probabilities):
        accumulators = getattr(
            self, '_task1_sampling_diagnostic_accumulators', None)
        if not accumulators or step not in accumulators:
            return
        accumulator = accumulators[step]
        state_tokens = state.argmax(dim=-1)
        posterior_max, posterior_tokens = probabilities.max(dim=-1)
        token_count = state_tokens.numel()
        masks = self._task1_sampling_category_masks
        accumulator['state_token_counts'] += torch.bincount(
            state_tokens.flatten(), minlength=self.vocab_size).cpu()
        accumulator['posterior_token_counts'] += torch.bincount(
            posterior_tokens.flatten(), minlength=self.vocab_size).cpu()
        accumulator['token_count'] += token_count
        accumulator['state_slash_count'] += int(
            masks['slash'][state_tokens].sum().cpu())
        accumulator['state_digit_count'] += int(
            masks['digit'][state_tokens].sum().cpu())
        accumulator['posterior_slash_count'] += int(
            masks['slash'][posterior_tokens].sum().cpu())
        accumulator['posterior_digit_count'] += int(
            masks['digit'][posterior_tokens].sum().cpu())
        accumulator['state_l2_sum'] += float(
            torch.linalg.vector_norm(state.float(), dim=-1).sum().cpu())
        predictive_entropy = -(
            probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
        accumulator['predictive_entropy_sum'] += float(
            predictive_entropy.sum().cpu())
        accumulator['posterior_max_probability_sum'] += float(
            posterior_max.sum().cpu())

    def task1_finalize_sampling_diagnostics(self):
        """Return exact corpus reductions for the configured early steps."""
        accumulators = getattr(
            self, '_task1_sampling_diagnostic_accumulators', None)
        if not accumulators:
            return None

        def distribution_summary(counts, token_count):
            positive = counts[counts > 0].double()
            frequencies = positive / float(token_count)
            entropy = float(-(frequencies * frequencies.log()).sum())
            top_count = min(10, counts.numel())
            top_values, top_ids = torch.topk(counts, k=top_count)
            top_tokens = []
            for count, token_id in zip(top_values.tolist(), top_ids.tolist()):
                top_tokens.append({
                    'token_id': token_id,
                    'token_text': self.tokenizer.decode([token_id]),
                    'count': count,
                    'fraction': count / float(token_count),
                })
            return {
                'token_entropy_nats': entropy,
                'top1_token_fraction': top_values[0].item() / float(token_count),
                'unique_token_count': int((counts > 0).sum()),
                'top_tokens': top_tokens,
            }

        rows = []
        for step, accumulator in sorted(accumulators.items()):
            token_count = accumulator['token_count']
            state_summary = distribution_summary(
                accumulator['state_token_counts'], token_count)
            posterior_summary = distribution_summary(
                accumulator['posterior_token_counts'], token_count)
            state_summary.update({
                'slash_fraction': (
                    accumulator['state_slash_count'] / token_count),
                'digit_fraction': (
                    accumulator['state_digit_count'] / token_count),
            })
            posterior_summary.update({
                'slash_fraction': (
                    accumulator['posterior_slash_count'] / token_count),
                'digit_fraction': (
                    accumulator['posterior_digit_count'] / token_count),
                'mean_predictive_entropy_nats': (
                    accumulator['predictive_entropy_sum'] / token_count),
                'mean_top1_probability': (
                    accumulator['posterior_max_probability_sum'] / token_count),
            })
            rows.append({
                'step': step,
                'token_count': token_count,
                'mean_state_l2': accumulator['state_l2_sum'] / token_count,
                'mean_state_rms': (
                    accumulator['state_l2_sum'] / token_count
                    / (self.vocab_size ** 0.5)),
                'state_top_choice': state_summary,
                'posterior_top_choice': posterior_summary,
            })
        return {
            **self._task1_sampling_diagnostic_metadata,
            'rows': rows,
        }
