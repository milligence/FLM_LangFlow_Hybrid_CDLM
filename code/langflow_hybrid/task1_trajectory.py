"""Streaming diagnostics for a uniform-physical-time Task1 trajectory."""

import math

import torch


class Task1TrajectoryDiagnostics:
    """Aggregate p/v, confidence, SC, and fp32 probes without full traces."""

    SUPPORTED_REFERENCE_NFE = (512, 1024, 2048)
    COARSE_NFES = (128, 256, 512, 1024)

    def __init__(self, grid_metadata, initial_state, initial_noise_seed,
                 initial_noise_schedule, checkpoint_label, top_k,
                 fp32_sample_count=8, fp32_node_count=17):
        self.reference_nfe = int(grid_metadata.get('nfe', -1))
        if self.reference_nfe not in self.SUPPORTED_REFERENCE_NFE:
            raise ValueError(
                'Trajectory diagnostics require 512, 1024, or 2048 NFE.')
        if grid_metadata.get('name') != (
                'physical_time_uniform_official_inverse_lut'):
            raise ValueError(
                'Trajectory diagnostics require strict uniform physical time.')
        self.tau_points = tuple(float(value) for value in
                                grid_metadata['tau_points'])
        self.physical_time_points = tuple(float(value) for value in
                                          grid_metadata['physical_time_points'])
        self.interval_mid_physical_time_points = tuple(
            float(value) for value in
            grid_metadata['interval_mid_physical_time_points'])
        self.interval_mid_tau_points = tuple(
            float(value) for value in grid_metadata['interval_mid_tau_points'])
        if len(self.physical_time_points) != self.reference_nfe + 1:
            raise ValueError('Reference grid endpoint count mismatch.')
        self.device = initial_state.device
        self.checkpoint_label = str(checkpoint_label)
        self.initial_noise_seed = int(initial_noise_seed)
        self.initial_noise_schedule = str(initial_noise_schedule)
        self.top_k = int(top_k)
        self.fp32_sample_count = int(fp32_sample_count)
        self.fp32_node_count = int(fp32_node_count)
        if self.top_k <= 0 or self.fp32_sample_count <= 0:
            raise ValueError('Trajectory top-k and fp32 sample count must be positive.')
        if not 2 <= self.fp32_node_count <= self.reference_nfe:
            raise ValueError('fp32 node count must be in [2, reference_nfe].')
        self.fp32_query_indices = tuple(sorted({
            round(index * (self.reference_nfe - 1)
                  / (self.fp32_node_count - 1))
            for index in range(self.fp32_node_count)}))
        self.batch_count = 0
        self.sample_count = 0
        self._current_fp32_count = 0
        self._vector_field = self._new_difference_accumulator(
            self.reference_nfe)
        self._velocity_change = self._new_difference_accumulator(
            self.reference_nfe)
        self._probability_change = self._new_difference_accumulator(
            self.reference_nfe)
        self._probability_reconstruction_error = (
            self._new_difference_accumulator(self.reference_nfe))
        self._self_conditioning = self._new_difference_accumulator(
            self.reference_nfe)
        self._self_conditioning_change = self._new_difference_accumulator(
            self.reference_nfe)
        self._self_conditioning_nonfinite = torch.zeros(
            self.reference_nfe, device=self.device, dtype=torch.int64)
        self._confidence = self._new_confidence_accumulator(
            self.reference_nfe)
        self._fp32 = self._new_fp32_accumulator(self.reference_nfe)
        self.coarse_nfes = tuple(
            nfe for nfe in self.COARSE_NFES if nfe < self.reference_nfe)
        self._coarse = {
            nfe: self._new_difference_accumulator(nfe)
            for nfe in self.coarse_nfes}
        self._previous_vector_field = None
        self._previous_probabilities = None
        self._previous_top1 = None
        self._previous_top1_probability = None
        self._previous_margin = None
        self._previous_self_conditioning = None
        self._pending = {}

    def _new_difference_accumulator(self, size):
        return {
            'sum_absolute': torch.zeros(size, device=self.device,
                                        dtype=torch.float64),
            'sum_squared': torch.zeros(size, device=self.device,
                                       dtype=torch.float64),
            'max_absolute': torch.zeros(size, device=self.device,
                                        dtype=torch.float32),
            'element_count': torch.zeros(size, device=self.device,
                                         dtype=torch.int64),
        }

    def _new_confidence_accumulator(self, size):
        fields = (
            'top1_probability_sum', 'top2_probability_sum', 'margin_sum',
            'total_variation_sum', 'flipped_previous_top1_probability_sum',
            'flipped_current_top1_probability_sum',
            'flipped_previous_margin_sum', 'flipped_current_margin_sum')
        result = {
            field: torch.zeros(size, device=self.device, dtype=torch.float64)
            for field in fields}
        result.update({
            'token_count': torch.zeros(size, device=self.device,
                                       dtype=torch.int64),
            'flip_count': torch.zeros(size, device=self.device,
                                      dtype=torch.int64),
        })
        return result

    def _new_fp32_accumulator(self, size):
        result = self._new_difference_accumulator(size)
        for field in (
                'token_count', 'top1_changed_count', 'top2_set_changed_count',
                'near_tie_count', 'near_tie_top1_changed_count'):
            result[field] = torch.zeros(
                size, device=self.device, dtype=torch.int64)
        for field in ('total_variation_sum', 'main_margin_sum',
                      'fp32_margin_sum'):
            result[field] = torch.zeros(
                size, device=self.device, dtype=torch.float64)
        return result

    @staticmethod
    def _accumulate(accumulator, index, difference):
        flat = difference.float()
        accumulator['sum_absolute'][index] += flat.abs().sum(
            dtype=torch.float64)
        accumulator['sum_squared'][index] += flat.square().sum(
            dtype=torch.float64)
        accumulator['max_absolute'][index] = torch.maximum(
            accumulator['max_absolute'][index], flat.abs().amax())
        accumulator['element_count'][index] += flat.numel()

    def start_batch(self, initial_state):
        if initial_state.device != self.device:
            raise ValueError('All trajectory batches must use one device.')
        already_selected = min(self.sample_count, self.fp32_sample_count)
        self._current_fp32_count = min(
            int(initial_state.shape[0]),
            self.fp32_sample_count - already_selected)
        self.batch_count += 1
        self.sample_count += int(initial_state.shape[0])
        self._previous_vector_field = None
        self._previous_probabilities = None
        self._previous_top1 = None
        self._previous_top1_probability = None
        self._previous_margin = None
        self._previous_self_conditioning = None
        self._pending = {}

    def fp32_local_sample_count(self):
        return self._current_fp32_count

    def should_run_fp32(self, fine_index):
        return (self._current_fp32_count > 0
                and int(fine_index) in self.fp32_query_indices)

    def arrive_at_fine_state(self, fine_index, state):
        completed = [
            nfe for nfe, pending in self._pending.items()
            if pending['target_fine_index'] == fine_index]
        for nfe in completed:
            pending = self._pending.pop(nfe)
            self._accumulate(
                self._coarse[nfe], pending['coarse_interval_index'],
                pending['coarse_endpoint'] - state)

    def _record_confidence(self, index, probabilities):
        values, indices = probabilities.float().topk(k=2, dim=-1)
        top1_probability = values[..., 0]
        top2_probability = values[..., 1]
        margin = top1_probability - top2_probability
        top1 = indices[..., 0]
        count = top1.numel()
        self._confidence['top1_probability_sum'][index] += (
            top1_probability.sum(dtype=torch.float64))
        self._confidence['top2_probability_sum'][index] += (
            top2_probability.sum(dtype=torch.float64))
        self._confidence['margin_sum'][index] += margin.sum(dtype=torch.float64)
        self._confidence['token_count'][index] += count
        return top1, top1_probability, margin

    def _record_fp32(self, index, probabilities, fp32_probabilities):
        if fp32_probabilities is None:
            return
        main = probabilities[:self._current_fp32_count].float()
        precise = fp32_probabilities.float()
        if main.shape != precise.shape:
            raise ValueError('fp32 probe probabilities must match its main subset.')
        difference = precise - main
        self._accumulate(self._fp32, index, difference)
        main_values, main_indices = main.topk(k=2, dim=-1)
        fp32_values, fp32_indices = precise.topk(k=2, dim=-1)
        main_margin = main_values[..., 0] - main_values[..., 1]
        fp32_margin = fp32_values[..., 0] - fp32_values[..., 1]
        near_tie = main_margin <= 1e-3
        top1_changed = main_indices[..., 0] != fp32_indices[..., 0]
        top2_set_changed = (
            main_indices.sort(dim=-1).values
            != fp32_indices.sort(dim=-1).values).any(dim=-1)
        token_count = top1_changed.numel()
        self._fp32['token_count'][index] += token_count
        self._fp32['top1_changed_count'][index] += top1_changed.sum()
        self._fp32['top2_set_changed_count'][index] += top2_set_changed.sum()
        self._fp32['near_tie_count'][index] += near_tie.sum()
        self._fp32['near_tie_top1_changed_count'][index] += (
            near_tie & top1_changed).sum()
        self._fp32['total_variation_sum'][index] += (
            0.5 * difference.abs().sum(dim=-1).sum(dtype=torch.float64))
        self._fp32['main_margin_sum'][index] += main_margin.sum(
            dtype=torch.float64)
        self._fp32['fp32_margin_sum'][index] += fp32_margin.sum(
            dtype=torch.float64)

    def observe_prediction(self, fine_index, state, probabilities,
                           physical_t, eps, euler_update,
                           self_conditioning=None,
                           fp32_probabilities=None):
        denominator = (1.0 - physical_t.float()).clamp_min(float(eps))
        while denominator.ndim < state.ndim:
            denominator = denominator.unsqueeze(-1)
        vector_field = (
            (probabilities.float() - state.float()) / denominator)
        self._accumulate(self._vector_field, fine_index, vector_field)
        top1, top1_probability, margin = self._record_confidence(
            fine_index, probabilities)

        if self_conditioning is not None:
            self._self_conditioning_nonfinite[fine_index] += (
                ~torch.isfinite(self_conditioning)).sum()
            self._accumulate(
                self._self_conditioning, fine_index, self_conditioning)
            if self._previous_self_conditioning is not None:
                self._accumulate(
                    self._self_conditioning_change, fine_index,
                    self_conditioning.float()
                    - self._previous_self_conditioning.float())
            self._previous_self_conditioning = self_conditioning.detach()

        if self._previous_vector_field is not None:
            delta_v = vector_field - self._previous_vector_field
            delta_p = probabilities.float() - self._previous_probabilities
            reconstruction = (1.0 - physical_t.float()) * delta_v
            self._accumulate(self._velocity_change, fine_index, delta_v)
            self._accumulate(self._probability_change, fine_index, delta_p)
            self._accumulate(
                self._probability_reconstruction_error, fine_index,
                delta_p - reconstruction)
            tv = 0.5 * delta_p.abs().sum(dim=-1)
            self._confidence['total_variation_sum'][fine_index] += tv.sum(
                dtype=torch.float64)
            changed = top1 != self._previous_top1
            flip_count = changed.sum()
            self._confidence['flip_count'][fine_index] += flip_count
            if bool(changed.any()):
                self._confidence[
                    'flipped_previous_top1_probability_sum'][fine_index] += (
                    self._previous_top1_probability[changed].sum(
                        dtype=torch.float64))
                self._confidence[
                    'flipped_current_top1_probability_sum'][fine_index] += (
                    top1_probability[changed].sum(dtype=torch.float64))
                self._confidence['flipped_previous_margin_sum'][fine_index] += (
                    self._previous_margin[changed].sum(dtype=torch.float64))
                self._confidence['flipped_current_margin_sum'][fine_index] += (
                    margin[changed].sum(dtype=torch.float64))

        self._record_fp32(fine_index, probabilities, fp32_probabilities)
        self._previous_vector_field = vector_field.detach()
        self._previous_probabilities = probabilities.detach().float()
        self._previous_top1 = top1.detach()
        self._previous_top1_probability = top1_probability.detach()
        self._previous_margin = margin.detach()

        for coarse_nfe in self.coarse_nfes:
            ratio = self.reference_nfe // coarse_nfe
            if fine_index % ratio != 0:
                continue
            if coarse_nfe in self._pending:
                raise RuntimeError(
                    f'Unclosed pending endpoint for {coarse_nfe} NFE.')
            target_index = fine_index + ratio
            self._pending[coarse_nfe] = {
                'coarse_interval_index': fine_index // ratio,
                'target_fine_index': target_index,
                'coarse_endpoint': euler_update(
                    state, probabilities,
                    self.physical_time_points[fine_index],
                    self.physical_time_points[target_index], eps).detach(),
            }

    def finish_batch(self, final_state):
        self.arrive_at_fine_state(self.reference_nfe, final_state)
        if self._pending:
            raise RuntimeError(
                f'Unclosed trajectory endpoints: {sorted(self._pending)}.')
        self._previous_vector_field = None
        self._previous_probabilities = None
        self._previous_top1 = None
        self._previous_self_conditioning = None

    @staticmethod
    def _metrics(accumulator, index, allow_empty=False):
        count = int(accumulator['element_count'][index].item())
        if count <= 0:
            if allow_empty:
                return None
            raise RuntimeError('Trajectory interval has no aggregated elements.')
        return {
            'element_count': count,
            'mean_absolute': float(
                accumulator['sum_absolute'][index].item() / count),
            'rms': math.sqrt(
                accumulator['sum_squared'][index].item() / count),
            'max_absolute': float(
                accumulator['max_absolute'][index].item()),
        }

    def _physical_interval_rows(self):
        rows = []
        for index in range(self.reference_nfe):
            vector = self._metrics(self._vector_field, index)
            count = int(self._confidence['token_count'][index].item())
            adjacent = index > 0
            flip_count = int(self._confidence['flip_count'][index].item())
            velocity = self._metrics(
                self._velocity_change, index, allow_empty=True)
            probability = self._metrics(
                self._probability_change, index, allow_empty=True)
            reconstruction = self._metrics(
                self._probability_reconstruction_error, index,
                allow_empty=True)
            sc = self._metrics(
                self._self_conditioning, index, allow_empty=True)
            sc_change = self._metrics(
                self._self_conditioning_change, index, allow_empty=True)
            row = {
                'interval_index': index,
                'start_physical_t': self.physical_time_points[index],
                'mid_physical_t': self.interval_mid_physical_time_points[index],
                'end_physical_t': self.physical_time_points[index + 1],
                'start_tau': self.tau_points[index],
                'mid_tau': self.interval_mid_tau_points[index],
                'end_tau': self.tau_points[index + 1],
                'vector_field': vector,
                'top1_probability_mean': float(
                    self._confidence['top1_probability_sum'][index].item()
                    / count),
                'top2_probability_mean': float(
                    self._confidence['top2_probability_sum'][index].item()
                    / count),
                'top1_top2_margin_mean': float(
                    self._confidence['margin_sum'][index].item() / count),
                'adjacent_change_available': adjacent,
                'velocity_change': velocity,
                'probability_change_direct': probability,
                'probability_delta_reconstruction_error': reconstruction,
                'probability_total_variation_mean': (
                    None if not adjacent else float(
                        self._confidence['total_variation_sum'][index].item()
                        / count)),
                'top1_flip_count': (None if not adjacent else flip_count),
                'top1_flip_rate': (
                    None if not adjacent else flip_count / count),
                'flip_confidence': (
                    None if flip_count == 0 else {
                        'previous_top1_probability_mean': float(
                            self._confidence[
                                'flipped_previous_top1_probability_sum'][
                                    index].item() / flip_count),
                        'current_top1_probability_mean': float(
                            self._confidence[
                                'flipped_current_top1_probability_sum'][
                                    index].item() / flip_count),
                        'previous_margin_mean': float(
                            self._confidence[
                                'flipped_previous_margin_sum'][index].item()
                            / flip_count),
                        'current_margin_mean': float(
                            self._confidence[
                                'flipped_current_margin_sum'][index].item()
                            / flip_count),
                    }),
                'self_conditioning': {
                    'input_available': sc is not None,
                    'input_finite': (
                        None if sc is None else
                        int(self._self_conditioning_nonfinite[index].item())
                        == 0),
                    'nonfinite_count': (
                        None if sc is None else int(
                            self._self_conditioning_nonfinite[index].item())),
                    'input': sc,
                    'change_from_previous_sc_input': sc_change,
                },
            }
            rows.append(row)
        return rows

    def _coarse_rows(self, coarse_nfe):
        ratio = self.reference_nfe // coarse_nfe
        rows = []
        for index in range(coarse_nfe):
            start = index * ratio
            end = (index + 1) * ratio
            row = {
                'coarse_interval_index': index,
                'start_physical_t': self.physical_time_points[start],
                'end_physical_t': self.physical_time_points[end],
                'start_tau': self.tau_points[start],
                'end_tau': self.tau_points[end],
                'fine_step_ratio': ratio,
            }
            row.update(self._metrics(self._coarse[coarse_nfe], index))
            rows.append(row)
        return rows

    def _fp32_rows(self):
        rows = []
        for index in self.fp32_query_indices:
            count = int(self._fp32['token_count'][index].item())
            if count <= 0:
                continue
            near = int(self._fp32['near_tie_count'][index].item())
            rows.append({
                'query_index': index,
                'physical_t': self.physical_time_points[index],
                'tau': self.tau_points[index],
                'sample_count': self.fp32_sample_count,
                'token_count': count,
                'probability_difference': self._metrics(self._fp32, index),
                'probability_total_variation_mean': float(
                    self._fp32['total_variation_sum'][index].item() / count),
                'top1_changed_count': int(
                    self._fp32['top1_changed_count'][index].item()),
                'top1_changed_rate': float(
                    self._fp32['top1_changed_count'][index].item() / count),
                'top2_set_changed_rate': float(
                    self._fp32['top2_set_changed_count'][index].item()
                    / count),
                'near_tie_threshold': 1e-3,
                'near_tie_count': near,
                'near_tie_top1_changed_rate': (
                    None if near == 0 else float(
                        self._fp32['near_tie_top1_changed_count'][index].item()
                        / near)),
                'main_margin_mean': float(
                    self._fp32['main_margin_sum'][index].item() / count),
                'fp32_margin_mean': float(
                    self._fp32['fp32_margin_sum'][index].item() / count),
            })
        return rows

    @staticmethod
    def _cpu(accumulator):
        return {key: value.detach().cpu()
                for key, value in accumulator.items()}

    def finalize(self):
        self._vector_field = self._cpu(self._vector_field)
        self._velocity_change = self._cpu(self._velocity_change)
        self._probability_change = self._cpu(self._probability_change)
        self._probability_reconstruction_error = self._cpu(
            self._probability_reconstruction_error)
        self._self_conditioning = self._cpu(self._self_conditioning)
        self._self_conditioning_change = self._cpu(
            self._self_conditioning_change)
        self._self_conditioning_nonfinite = (
            self._self_conditioning_nonfinite.detach().cpu())
        self._confidence = self._cpu(self._confidence)
        self._fp32 = self._cpu(self._fp32)
        self._coarse = {
            nfe: self._cpu(accumulator)
            for nfe, accumulator in self._coarse.items()}
        rows = self._physical_interval_rows()
        scales = []
        for coarse_nfe in self.coarse_nfes:
            coarse_rows = self._coarse_rows(coarse_nfe)
            scales.append({
                'coarse_nfe': coarse_nfe,
                'fine_step_ratio': self.reference_nfe // coarse_nfe,
                'interval_count': len(coarse_rows),
                'rows': coarse_rows,
            })
        return {
            'schema': 'task1-uniform-t-trajectory-diagnostics-v2',
            'artifact_semantics': (
                'unlabeled free-generation trajectory diagnostics; these '
                'metrics are not perplexity and generated tokens are never '
                'treated as ground-truth labels'),
            'identity': {
                'checkpoint_label': self.checkpoint_label,
                'reference_nfe': self.reference_nfe,
                'sample_count': self.sample_count,
                'batch_count': self.batch_count,
                'initial_noise_seed': self.initial_noise_seed,
                'common_noise_identity': (
                    f'base_seed_{self.initial_noise_seed}_'
                    f'{self.initial_noise_schedule.removeprefix("base_seed_")}'),
                'grid_identity': (
                    f'physical_time=linspace(0,1,{self.reference_nfe + 1}); '
                    'tau=official_inverse_lut(physical_time)'),
                'reference_endpoint_update_performed_for_diagnostics': True,
                'returned_sample_uses_last_query_prediction': True,
            },
            'definitions': {
                'p': 'softmax probability prediction at each query node',
                'v': '(p - state) / max(1 - physical_t, eps)',
                'probability_change_direct': 'p_next - p_current',
                'probability_change_reconstruction': (
                    '(1 - t_next) * (v_next - v_current)'),
                'probability_delta_precedence': (
                    'direct p subtraction is authoritative when clamp or '
                    'rounding makes the reconstruction differ'),
                'probability_total_variation': (
                    '0.5 * sum_vocabulary(abs(p_next - p_current))'),
                'labels': 'none for free generation',
            },
            'physical_intervals': {
                'interval_count': len(rows),
                'rows': rows,
                'top_intervals_by_velocity_change_rms': sorted(
                    rows[1:],
                    key=lambda row: row['velocity_change']['rms'],
                    reverse=True)[:self.top_k],
                'top_intervals_by_probability_tv': sorted(
                    rows[1:],
                    key=lambda row: row['probability_total_variation_mean'],
                    reverse=True)[:self.top_k],
                'top_intervals_by_flip_rate': sorted(
                    rows[1:], key=lambda row: row['top1_flip_rate'],
                    reverse=True)[:self.top_k],
            },
            'local_coarse_vs_fine': {
                'reference_nfe': self.reference_nfe,
                'scales': scales,
            },
            'fp32_forward_sensitivity': {
                'sample_count': self.fp32_sample_count,
                'query_indices': list(self.fp32_query_indices),
                'common_physical_nodes': True,
                'same_checkpoint_noise_state_and_sc_input': True,
                'rows': self._fp32_rows(),
            },
        }
