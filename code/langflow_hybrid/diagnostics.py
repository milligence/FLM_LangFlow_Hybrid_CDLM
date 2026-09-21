"""Diagnostics and diagnostic checkpoint state for the LangFlow hybrid."""

import collections
import copy
import math

import torch
import torch.nn.functional as F


class LangFlowDiagnosticsMixin:
    """Gamma/frequency, gradient-route, logit, and geometry diagnostics."""

    @staticmethod
    def _l2_norm(parameters, use_grad=False):
        squared = None
        for parameter in parameters:
            value = parameter.grad if use_grad else parameter
            if value is None:
                continue
            term = value.detach().float().square().sum()
            squared = term if squared is None else squared + term
        return None if squared is None else squared.sqrt()

    @staticmethod
    def _optimization_group(parameter_name):
        if parameter_name.startswith('backbone.vocab_embed.'):
            return 'input_projection'
        if parameter_name.startswith('backbone.output_layer.linear.'):
            return 'output_classifier'
        if (parameter_name.startswith('backbone.sigma_map.')
                or parameter_name.startswith('backbone.sigma_map_prime.')
                or '.adaLN_modulation.' in parameter_name):
            return 'time_conditioning'
        return 'backbone_core'

    def _gradient_statistics(self):
        squared = {
            name: torch.zeros((), device=self.device, dtype=torch.float32)
            for name in (
                'global', 'backbone_core', 'input_projection',
                'output_classifier', 'time_conditioning')}
        nonfinite_tensors = torch.zeros(
            (), device=self.device, dtype=torch.long)
        found = False
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            found = True
            gradient = parameter.grad.detach().float()
            all_finite = torch.isfinite(gradient).all()
            nonfinite_tensors.add_((~all_finite).to(dtype=torch.long))
            term = gradient.square().sum()
            squared['global'].add_(term)
            squared[self._optimization_group(name)].add_(term)
        if not found:
            return None, nonfinite_tensors
        return ({name: value.sqrt() for name, value in squared.items()},
                nonfinite_tensors)

    def configure_gradient_clipping(
            self, optimizer, gradient_clip_val=None,
            gradient_clip_algorithm=None):
        if self.state_space != 'vocab':
            should_log = int(self.global_step) % 10 == 0
            parameters = list(self._get_parameters())
            grad_before = self._l2_norm(parameters, use_grad=True)
            parameter_norm = self._l2_norm(parameters)
            super().configure_gradient_clipping(
                optimizer, gradient_clip_val, gradient_clip_algorithm)
            grad_after = self._l2_norm(parameters, use_grad=True)
            if (should_log and grad_before is not None
                    and grad_after is not None):
                self.log(
                    'diagnostics/grad_norm_before_clip', grad_before,
                    on_step=True, on_epoch=False, sync_dist=True)
                self.log(
                    'diagnostics/grad_norm_after_clip', grad_after,
                    on_step=True, on_epoch=False, sync_dist=True)
                self.log(
                    'diagnostics/gradient_was_clipped',
                    (grad_after < grad_before).float(),
                    on_step=True, on_epoch=False, sync_dist=True)
            if should_log and parameter_norm is not None:
                self.log(
                    'diagnostics/parameter_norm', parameter_norm,
                    on_step=True, on_epoch=False, sync_dist=True)
            return
        optimizer_step = int(self.global_step) + 1
        should_log = (
            optimizer_step == 1
            or optimizer_step % self.optimization_diagnostic_interval_steps == 0)
        parameters = list(self._get_parameters())
        gradient_norms, nonfinite_gradient_tensors = (
            self._gradient_statistics())
        grad_before = (
            None if gradient_norms is None else gradient_norms['global'])
        self._task1_nonfinite_gradient_tensors.add_(
            nonfinite_gradient_tensors)
        self._task1_nonfinite_gradient_events.add_(
            (nonfinite_gradient_tensors > 0).to(dtype=torch.long))
        clip_value = float(gradient_clip_val or 0.0)
        clip_triggered = (
            torch.zeros((), device=self.device, dtype=torch.bool)
            if grad_before is None or clip_value <= 0.0
            else grad_before > clip_value)
        self._task1_gradient_clip_events.add_(
            clip_triggered.to(dtype=torch.long))
        parameter_norm = self._l2_norm(parameters) if should_log else None
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm)
        grad_after = (
            self._l2_norm(parameters, use_grad=True) if should_log else None)
        if should_log and grad_before is not None and grad_after is not None:
            self.log(
                'diagnostics/grad_norm_before_clip', grad_before,
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'diagnostics/grad_norm_after_clip', grad_after,
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'diagnostics/gradient_was_clipped',
                clip_triggered.float(),
                on_step=True, on_epoch=False, sync_dist=True)
            self.log(
                'diagnostics/grad_global_norm_preclip', grad_before,
                on_step=True, on_epoch=False, sync_dist=True)
            for group_name in (
                    'backbone_core', 'input_projection',
                    'output_classifier', 'time_conditioning'):
                self.log(
                    f'diagnostics/grad_{group_name}_norm_preclip',
                    gradient_norms[group_name], on_step=True,
                    on_epoch=False, sync_dist=True)
        if should_log and parameter_norm is not None:
            self.log(
                'diagnostics/parameter_norm', parameter_norm,
                on_step=True, on_epoch=False, sync_dist=True)
            learning_rate = float(optimizer.param_groups[0]['lr'])
            update_ratio_proxy = (
                learning_rate * grad_before
                / parameter_norm.clamp_min(1e-20))
            record = {
                'optimizer_step': optimizer_step,
                'actual_lr': learning_rate,
                'global_gradient_norm_preclip': float(
                    grad_before.detach().cpu()),
                'gradient_norm_after_clip': float(
                    grad_after.detach().cpu()),
                'gradient_clip_threshold': clip_value,
                'gradient_clip_triggered': bool(
                    clip_triggered.detach().cpu()),
                'gradient_clip_event_count': int(
                    self._task1_gradient_clip_events.detach().cpu()),
                'nonfinite_loss_event_count': int(
                    self._task1_nonfinite_loss_events.detach().cpu()),
                'nonfinite_loss_value_count': int(
                    self._task1_nonfinite_loss_values.detach().cpu()),
                'nonfinite_gradient_event_count': int(
                    self._task1_nonfinite_gradient_events.detach().cpu()),
                'nonfinite_gradient_tensor_count': int(
                    self._task1_nonfinite_gradient_tensors.detach().cpu()),
                'parameter_norm': float(parameter_norm.detach().cpu()),
                'lr_scaled_grad_to_weight_ratio_proxy': float(
                    update_ratio_proxy.detach().cpu()),
                'update_scale_definition': (
                    'actual_lr * global_preclip_gradient_l2 / parameter_l2; '
                    'low-overhead AdamW update-scale proxy'),
            }
            for group_name in (
                    'backbone_core', 'input_projection',
                    'output_classifier', 'time_conditioning'):
                record[f'{group_name}_gradient_norm_preclip'] = float(
                    gradient_norms[group_name].detach().cpu())
            self.optimization_diagnostics[str(optimizer_step)] = record
            for name, value in (
                    ('actual_lr', learning_rate),
                    ('lr_scaled_grad_to_weight_ratio_proxy',
                     update_ratio_proxy),
                    ('nonfinite_loss_event_count',
                     self._task1_nonfinite_loss_events),
                    ('nonfinite_gradient_event_count',
                     self._task1_nonfinite_gradient_events),
                    ('gradient_clip_event_count',
                     self._task1_gradient_clip_events)):
                self.log(
                    f'diagnostics/{name}', value, on_step=True,
                    on_epoch=False, sync_dist=True)

    @staticmethod
    def _update_tensor_moments(accumulator, name, value, count=None):
        if value is None:
            if count is None:
                raise ValueError('count is required for an absent tensor.')
            accumulator[name]['count'] += int(count)
            return
        detached = value.detach().float()
        accumulator[name]['sum'] += float(detached.sum().cpu())
        accumulator[name]['sum_square'] += float(
            detached.square().sum().cpu())
        accumulator[name]['count'] += detached.numel()

    def _record_validation_logit_stats(
            self, residual_logits, bias_logits, final_logits, bias_weight):
        if self._validation_logit_stats is None:
            return
        count = residual_logits.numel()
        self._update_tensor_moments(
            self._validation_logit_stats, 'residual', residual_logits)
        self._update_tensor_moments(
            self._validation_logit_stats, 'bias', bias_logits, count=count)
        self._update_tensor_moments(
            self._validation_logit_stats, 'final', final_logits)
        self._validation_logit_stats['bias_weight'] = float(bias_weight)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        metric_count = len(self._posterior_diagnostic_names())
        bin_count = self._validation_tau_edges().numel() - 1
        self._validation_gamma_bins = torch.zeros(
            bin_count, metric_count + 2,
            device=self.device, dtype=torch.float64)
        self._validation_posterior_rows = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list)))
        self._validation_logit_stats = {
            name: {'sum': 0.0, 'sum_square': 0.0, 'count': 0}
            for name in ('residual', 'bias', 'final')}
        self._validation_logit_stats['bias_weight'] = float(
            self._current_token_bias_weight())

    @staticmethod
    def _posterior_diagnostic_names():
        return (
            'token_ce', 'raw_brier', 'target_probability',
            'target_margin', 'top1_accuracy', 'top10_true_inclusion',
            'top100_true_inclusion', 'posterior_entropy',
            'max_probability', 'probability_square_sum',
            'abs_mse_target_logit_gradient',
            'ce_target_gradient_magnitude', 'target_gradient_ratio',
            'target_probability_lt_1e_6',
            'target_probability_lt_1e_5',
            'target_probability_lt_1e_4',
            'target_probability_lt_uniform', 'high_confidence_wrong')

    def _frequency_group_ids(self):
        counts = self._training_token_counts
        result = torch.zeros_like(counts)
        seen = counts[counts > 0].float()
        if seen.numel() == 0:
            return result
        low_cut = torch.quantile(seen, 1 / 3)
        high_cut = torch.quantile(seen, 2 / 3)
        result[(counts > 0) & (counts <= low_cut)] = 1
        result[(counts > low_cut) & (counts <= high_cut)] = 2
        result[counts > high_cut] = 3
        return result

    def _accumulate_gamma_bins(
            self, q, physical_t, diagnostics, valid_tokens,
            target_tokens=None):
        if self._validation_gamma_bins is None:
            return
        names = self._posterior_diagnostic_names()
        diagnostic_matrix = torch.stack(
            [diagnostics[name].detach().float() for name in names], dim=-1)
        edges = self._validation_tau_edges()
        custom_edges = getattr(
            self.config.algo, 'validation_tau_bin_edges', None) is not None
        coordinate = (
            physical_t if bool(getattr(
                self.config.algo,
                'validation_physical_t_stratified', False)) else q)
        bin_indices = torch.bucketize(
            coordinate.detach(), edges[1:-1], right=False)
        for bin_index in range(edges.numel() - 1):
            token_mask = (
                (bin_indices == bin_index)[:, None] * valid_tokens).bool()
            count = token_mask.sum()
            self._validation_gamma_bins[bin_index, 0] += (
                bin_indices == bin_index).sum()
            self._validation_gamma_bins[bin_index, 1] += count
            for metric_index, name in enumerate(names, start=2):
                self._validation_gamma_bins[bin_index, metric_index] += (
                    diagnostics[name].detach()[token_mask].double().sum())
            if target_tokens is None or self._validation_posterior_rows is None:
                continue
            frequency_groups = self._frequency_group_ids()[target_tokens]
            for group_index, group_name in enumerate(
                    ('unseen', 'low', 'mid', 'high')):
                combined_mask = token_mask & (frequency_groups == group_index)
                if not combined_mask.any():
                    continue
                label = (
                    f'{float(edges[bin_index]):.8g}-'
                    f'{float(edges[bin_index + 1]):.8g}'
                    if custom_edges else
                    f'{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}')
                bucket = self._validation_posterior_rows[
                    label][group_name]
                bucket['matrix'].append(
                    diagnostic_matrix[combined_mask].cpu())

    def _finish_gamma_bin_diagnostics(self):
        if self._validation_gamma_bins is None:
            return
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            torch.distributed.all_reduce(self._validation_gamma_bins)
        step = int(self.global_step)
        if step <= 0:
            step = int(getattr(self, '_loaded_checkpoint_global_step', step))
        if step <= 0 or step % self.diagnostic_interval_steps != 0:
            return
        names = self._posterior_diagnostic_names()
        edges = self._validation_tau_edges()
        custom_edges = getattr(
            self.config.algo, 'validation_tau_bin_edges', None) is not None
        result = {}
        for bin_index in range(edges.numel() - 1):
            count = self._validation_gamma_bins[bin_index, 1].clamp_min(1)
            prefix = (
                f'val/gamma_bin_{bin_index:02d}' if custom_edges else
                f'val/gamma_bin_{bin_index:02d}_{bin_index + 1:02d}')
            values = {'sample_count': float(
                self._validation_gamma_bins[bin_index, 0].detach().cpu())}
            self.log(
                f'{prefix}_sample_count',
                self._validation_gamma_bins[bin_index, 0],
                on_step=False, on_epoch=True, sync_dist=False)
            self.log(
                f'{prefix}_token_count',
                self._validation_gamma_bins[bin_index, 1],
                on_step=False, on_epoch=True, sync_dist=False)
            for metric_index, name in enumerate(names, start=2):
                value = (
                    self._validation_gamma_bins[bin_index, metric_index]
                    / count)
                self.log(
                    f'{prefix}_{name}', value,
                    on_step=False, on_epoch=True, sync_dist=False)
                values[name] = float(value.detach().cpu())
            label = (
                f'{float(edges[bin_index]):.8g}-'
                f'{float(edges[bin_index + 1]):.8g}'
                if custom_edges else
                f'{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}')
            result[label] = values
        self.gamma_bin_diagnostics[str(step)] = result
        bucket_result = {}
        for gamma_name, frequency_rows in self._validation_posterior_rows.items():
            bucket_result[gamma_name] = {}
            for frequency_name, rows in frequency_rows.items():
                values = {}
                matrix = torch.cat(rows['matrix'])
                for metric_index, name in enumerate(names):
                    vector = matrix[:, metric_index]
                    values[name] = float(vector.mean())
                    if name == 'target_probability':
                        values['target_probability_p10'] = float(
                            torch.quantile(vector, 0.10))
                        values['target_probability_median'] = float(
                            torch.quantile(vector, 0.50))
                        values['target_probability_p90'] = float(
                            torch.quantile(vector, 0.90))
                values['token_count'] = int(matrix.shape[0])
                bucket_result[gamma_name][frequency_name] = values
        self.posterior_bucket_diagnostics[str(step)] = bucket_result

    def _finish_validation_logit_stats(self):
        if self._validation_logit_stats is None:
            return
        result = {}
        for name in ('residual', 'bias', 'final'):
            values = self._validation_logit_stats[name]
            count = max(values['count'], 1)
            mean = values['sum'] / count
            mean_square = values['sum_square'] / count
            result[f'{name}_logit_rms'] = math.sqrt(
                max(mean_square, 0.0))
            result[f'{name}_logit_std'] = math.sqrt(
                max(mean_square - mean * mean, 0.0))
        result['bias_residual_rms_ratio'] = (
            result['bias_logit_rms']
            / max(result['residual_logit_rms'], 1e-20))
        result['bias_weight'] = self._validation_logit_stats['bias_weight']
        step = str(int(self.global_step))
        self.bias_logit_diagnostics[step] = result
        for name, value in result.items():
            self.log(
                f'val/hybrid_{name}', value, on_step=False, on_epoch=True,
                sync_dist=False)

    def on_validation_epoch_end(self):
        self._finish_gamma_bin_diagnostics()
        self._finish_validation_logit_stats()
        step = int(self.global_step)
        if step > 0 and step % self.diagnostic_interval_steps == 0:
            diagnostic = self.embedding_nearest_neighbor_diagnostic(4096)
            self.embedding_diagnostics[str(step)] = {
                key: float(value.detach().cpu())
                for key, value in diagnostic.items()}
            for key, value in diagnostic.items():
                self.log(
                    f'val/embedding_{key}', value,
                    on_step=False, on_epoch=True, sync_dist=True)
        if self.classification_prototype_mode == 'independent':
            diagnostic = self.classification_prototype_diagnostic(4096)
            self.prototype_diagnostics[str(step)] = {
                key: float(value.detach().cpu())
                for key, value in diagnostic.items()}
            for key, value in diagnostic.items():
                self.log(
                    f'val/prototype_{key}', value,
                    on_step=False, on_epoch=True, sync_dist=True)
        super().on_validation_epoch_end()

    def _log_probability_metrics(self, values, train_mode, valid_tokens):
        prefix = 'train' if train_mode else 'val'
        denominator = valid_tokens.sum().clamp_min(1)
        for name, value in values.items():
            masked_mean = (value.detach() * valid_tokens).sum() / denominator
            self.log(
                f'{prefix}/hybrid_{name}', masked_mean,
                on_step=train_mode,
                on_epoch=not train_mode,
                sync_dist=True,
                batch_size=value.shape[0])

    @staticmethod
    def _raw_gradient_from_normalized(raw_weight, normalized_gradient):
        unit = F.normalize(raw_weight.float(), dim=-1)
        radial = (normalized_gradient.float() * unit).sum(
            dim=-1, keepdim=True)
        return (math.sqrt(raw_weight.shape[-1])
                / raw_weight.float().norm(
                    dim=-1, keepdim=True).clamp_min(1e-20)
                * (normalized_gradient.float() - radial * unit))

    def _geometry_separation_proxy(self, raw_weight):
        indices = self._geometry_sample_indices[:512]
        unit = F.normalize(raw_weight[indices].float(), dim=-1)
        return (1.0 - (unit[0::2] * unit[1::2]).sum(dim=-1)).mean()

    def on_after_backward(self):
        super().on_after_backward()
        if self._route_step != int(self.global_step) + 1:
            return
        physical_parameter = self.backbone.vocab_embed.embedding
        candidate_parameter = self._classification_prototype_raw()
        physical_gradient = (
            physical_parameter.grad.detach().float().clone()
            if physical_parameter.grad is not None
            else torch.zeros_like(physical_parameter, dtype=torch.float32))
        candidate_total = (
            candidate_parameter.grad.detach().float().clone()
            if candidate_parameter.grad is not None
            else torch.zeros_like(candidate_parameter, dtype=torch.float32))
        if self._route_candidate_weight_grad is None:
            candidate = torch.zeros_like(candidate_total)
        else:
            candidate = self._raw_gradient_from_normalized(
                self._route_candidate_raw_snapshot,
                self._route_candidate_weight_grad)
        if self.classification_prototype_mode == 'shared':
            input_gradient = physical_gradient - candidate
            total = physical_gradient
        else:
            input_gradient = physical_gradient
            total = physical_gradient
        input_norm = input_gradient.norm()
        candidate_norm = candidate.norm()
        total_norm = total.norm()
        cosine = torch.tensor(0.0, device=total.device)
        if input_norm > 0 and candidate_norm > 0:
            cosine = F.cosine_similarity(
                input_gradient.flatten(), candidate.flatten(), dim=0)
        candidate_raw = self._route_candidate_raw_snapshot.float()
        physical_raw = self._route_physical_raw_snapshot.float()
        candidate_base_proxy = self._geometry_separation_proxy(candidate_raw)
        physical_base_proxy = self._geometry_separation_proxy(physical_raw)

        def directional_delta(raw, base_proxy, gradient):
            norm = gradient.norm().clamp_min(1e-20)
            perturbed = raw - 1e-3 * gradient / norm
            return self._geometry_separation_proxy(perturbed) - base_proxy

        record = {
            'input_gradient_norm': float(input_norm.cpu()),
            'candidate_gradient_norm': float(candidate_norm.cpu()),
            'total_codebook_gradient_norm': float(total_norm.cpu()),
            'physical_codebook_gradient_norm': float(
                physical_gradient.norm().cpu()),
            'classification_prototype_gradient_norm': float(
                candidate_total.norm().cpu()),
            'input_candidate_cosine': float(cosine.cpu()),
            'input_separation_proxy_directional_delta': float(
                directional_delta(
                    physical_raw, physical_base_proxy,
                    input_gradient).cpu()),
            'candidate_separation_proxy_directional_delta': float(
                directional_delta(
                    candidate_raw, candidate_base_proxy,
                    candidate).cpu()),
            'physical_separation_proxy_before_update': float(
                physical_base_proxy.cpu()),
            'classification_prototype_separation_proxy_before_update': float(
                candidate_base_proxy.cpu()),
            # Backward-compatible alias: this has always described the
            # candidate matrix for independent-prototype runs.
            'separation_proxy_before_update': float(
                candidate_base_proxy.cpu()),
            'approximation': (
                'fixed 512-token adjacent-pair cosine-distance proxy; '
                'unit-norm gradient perturbation 1e-3'),
        }
        if self._route_logit_stats:
            record.update(self._route_logit_stats)
        self.gradient_route_diagnostics[str(self._route_step)] = record

    def optimizer_step(self, *args, **kwargs):
        route_step = self._route_step
        route_record = self.gradient_route_diagnostics.get(
            str(route_step), {})
        prototype_before = None
        if (route_record
                and self.classification_prototype_mode == 'independent'):
            prototype_before = (
                self.backbone.classification_prototype.detach().clone())
        physical_before = None
        if route_record:
            physical_before = (
                self.backbone.vocab_embed.embedding.detach().clone())
        super().optimizer_step(*args, **kwargs)
        if route_record:
            route_raw = self._classification_prototype_raw().detach()
            after = self._geometry_separation_proxy(route_raw)
            route_record['separation_proxy_after_adam_update'] = float(
                after.detach().cpu())
            route_record['actual_adam_separation_proxy_delta'] = (
                route_record['separation_proxy_after_adam_update']
                - route_record['separation_proxy_before_update'])
            if prototype_before is not None:
                route_record[
                    'classification_prototype_update_norm'
                ] = float(
                    (route_raw - prototype_before).float().norm().cpu())
            route_record['physical_codebook_update_norm'] = float(
                (self.backbone.vocab_embed.embedding.detach()
                 - physical_before).float().norm().cpu())
        self._route_candidate_weight_grad = None
        self._route_candidate_raw_snapshot = None
        self._route_physical_raw_snapshot = None
        self._route_logit_stats = None

    @torch.no_grad()
    def embedding_nearest_neighbor_diagnostic(self, sample_size=8192):
        count = min(int(sample_size), len(self._geometry_sample_indices))
        indices = self._geometry_sample_indices[:count]
        raw_weights = self.backbone.vocab_embed.embedding[indices].float()
        raw_norms = raw_weights.norm(dim=-1)
        weights = F.normalize(raw_weights, dim=-1)
        similarities = weights @ weights.transpose(0, 1)
        similarities.fill_diagonal_(-torch.inf)
        nearest_distances = 1.0 - similarities.max(dim=-1).values
        centered = weights - weights.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(count - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        normalized_spectrum = eigenvalues / eigenvalues.sum().clamp_min(1e-20)
        positive = normalized_spectrum[normalized_spectrum > 0]
        effective_rank = torch.exp(-(positive * positive.log()).sum())
        initial = self._initial_geometry_unit_vectors[:count].to(weights.device)
        angular_drift = torch.acos(
            (weights * initial).sum(dim=-1).clamp(-1.0, 1.0))
        result = {
            'raw_norm_mean': raw_norms.mean(),
            'raw_norm_std': raw_norms.std(unbiased=False),
            'raw_norm_p05': torch.quantile(raw_norms, 0.05),
            'raw_norm_p50': torch.quantile(raw_norms, 0.50),
            'raw_norm_p95': torch.quantile(raw_norms, 0.95),
            'nearest_cosine_distance_mean': nearest_distances.mean(),
            'nearest_cosine_distance_p05': torch.quantile(
                nearest_distances, 0.05),
            'nearest_cosine_distance_p50': torch.quantile(
                nearest_distances, 0.50),
            'nearest_cosine_distance_p95': torch.quantile(
                nearest_distances, 0.95),
            'centroid_norm': weights.mean(dim=0).norm(),
            'physical_codebook_mean_norm': self.embedding_weight().float().mean(
                dim=0).norm(),
            'centered_covariance_trace': eigenvalues.sum(),
            'centered_effective_rank': effective_rank,
            'angular_drift_mean_radians': angular_drift.mean(),
        }
        for rank, value in enumerate(eigenvalues.flip(0)[:16], start=1):
            result[f'centered_covariance_eigenvalue_top{rank:02d}'] = value
        frequency_groups = self._frequency_group_ids()[indices]
        for group_index, group_name in enumerate(
                ('unseen', 'low', 'mid', 'high')):
            mask = frequency_groups == group_index
            if mask.any():
                result[f'{group_name}_nn_distance_mean'] = (
                    nearest_distances[mask].mean())
                result[f'{group_name}_angular_drift_mean_radians'] = (
                    angular_drift[mask].mean())
                result[f'{group_name}_sample_count'] = mask.sum().float()
        return result

    @torch.no_grad()
    def classification_prototype_diagnostic(self, sample_size=8192):
        if self.classification_prototype_mode != 'independent':
            return {}
        count = min(int(sample_size), len(self._geometry_sample_indices))
        indices = self._geometry_sample_indices[:count]
        raw = self.backbone.classification_prototype[indices].float()
        effective = F.normalize(raw, dim=-1)
        physical = F.normalize(
            self.backbone.vocab_embed.embedding[indices].float(), dim=-1)
        raw_norms = raw.norm(dim=-1)
        effective_norms = self.classification_prototype_weight()[
            indices].float().norm(dim=-1)
        similarities = effective @ effective.transpose(0, 1)
        similarities.fill_diagonal_(-torch.inf)
        nearest = 1.0 - similarities.max(dim=-1).values
        angular = torch.acos(
            (effective * physical).sum(dim=-1).clamp(-1.0, 1.0))
        initial = self._initial_geometry_unit_vectors[:count].to(
            effective.device)
        initial_angular = torch.acos(
            (effective * initial).sum(dim=-1).clamp(-1.0, 1.0))
        return {
            'bc_mean_angular_distance_radians': angular.mean(),
            'b_b0_mean_angular_drift_radians': initial_angular.mean(),
            'nearest_neighbor_cosine_distance_mean': nearest.mean(),
            'raw_norm_mean': raw_norms.mean(),
            'raw_norm_std': raw_norms.std(unbiased=False),
            'raw_norm_min': raw_norms.min(),
            'raw_norm_max': raw_norms.max(),
            'effective_norm_mean': effective_norms.mean(),
            'effective_norm_std': effective_norms.std(unbiased=False),
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hybrid_token_bias_step'] = int(self.global_step)
        optimizer_lrs = [
            float(group['lr'])
            for optimizer in self.trainer.optimizers
            for group in optimizer.param_groups]
        transition_target = self._task1_resume_value(
            'transition_target_learning_rate')
        transition_schedule = self._task1_resume_value(
            'transition_schedule')
        checkpoint['task1_optimizer_contract'] = {
            'target_learning_rate': float(
                transition_target if transition_target is not None
                else self.config.optim.lr),
            'actual_learning_rates': optimizer_lrs,
            'scheduler': (
                f'{transition_schedule or "linear"}_transition_then_constant'
                if transition_target is not None
                else 'constant_after_warmup'),
            'lr_warmup_optimizer_steps': int(
                self.config.lr_scheduler.num_warmup_steps),
            'gaussian_bias_warmup_optimizer_steps': int(
                self.token_bias_warmup_steps),
            'global_optimizer_step': int(self.global_step),
        }
        checkpoint['task1_lr_graft_history'] = copy.deepcopy(
            self._task1_lr_graft_history)
        checkpoint['task1_training_time_diagnostics'] = copy.deepcopy(
            self.task1_training_time_diagnostics)
        if hasattr(self, 'task1_training_rng_state'):
            checkpoint['task1_training_rng_state'] = (
                self.task1_training_rng_state())
        checkpoint['hybrid_codebook_gradient_mode'] = (
            self.codebook_gradient_mode)
        checkpoint['hybrid_token_bias_schedule'] = self.token_bias_schedule
        checkpoint['hybrid_classification_prototype_mode'] = (
            self.classification_prototype_mode)
        checkpoint['hybrid_embedding_diagnostics'] = copy.deepcopy(
            self.embedding_diagnostics)
        checkpoint['hybrid_gamma_bin_diagnostics'] = copy.deepcopy(
            self.gamma_bin_diagnostics)
        checkpoint['hybrid_posterior_bucket_diagnostics'] = copy.deepcopy(
            self.posterior_bucket_diagnostics)
        checkpoint['hybrid_gradient_route_diagnostics'] = copy.deepcopy(
            self.gradient_route_diagnostics)
        checkpoint['task1_optimization_diagnostics'] = copy.deepcopy(
            self.optimization_diagnostics)
        checkpoint['task1_nonfinite_counts'] = {
            'loss_events': int(
                self._task1_nonfinite_loss_events.detach().cpu()),
            'loss_values': int(
                self._task1_nonfinite_loss_values.detach().cpu()),
            'gradient_events': int(
                self._task1_nonfinite_gradient_events.detach().cpu()),
            'gradient_tensors': int(
                self._task1_nonfinite_gradient_tensors.detach().cpu()),
            'gradient_clip_events': int(
                self._task1_gradient_clip_events.detach().cpu()),
        }
        checkpoint['hybrid_bias_logit_diagnostics'] = copy.deepcopy(
            self.bias_logit_diagnostics)
        checkpoint['hybrid_prototype_diagnostics'] = copy.deepcopy(
            self.prototype_diagnostics)
        checkpoint['hybrid_training_token_counts'] = (
            self._training_token_counts.detach().cpu())
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self._loaded_checkpoint_global_step = int(checkpoint.get(
            'hybrid_token_bias_step', checkpoint.get('global_step', 0)))
        self.embedding_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_embedding_diagnostics', {}))
        self.posterior_bucket_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_posterior_bucket_diagnostics', {}))
        self.gradient_route_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_gradient_route_diagnostics', {}))
        self.optimization_diagnostics = copy.deepcopy(
            checkpoint.get('task1_optimization_diagnostics', {}))
        self.task1_training_time_diagnostics = copy.deepcopy(
            checkpoint.get('task1_training_time_diagnostics', {}))
        self._task1_pending_training_rng_state = copy.deepcopy(
            checkpoint.get('task1_training_rng_state'))
        nonfinite_counts = checkpoint.get('task1_nonfinite_counts', {})
        for field, buffer_name in (
                ('loss_events', '_task1_nonfinite_loss_events'),
                ('loss_values', '_task1_nonfinite_loss_values'),
                ('gradient_events', '_task1_nonfinite_gradient_events'),
                ('gradient_tensors', '_task1_nonfinite_gradient_tensors'),
                ('gradient_clip_events', '_task1_gradient_clip_events')):
            getattr(self, buffer_name).fill_(int(nonfinite_counts.get(field, 0)))
        self.bias_logit_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_bias_logit_diagnostics', {}))
        self.prototype_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_prototype_diagnostics', {}))
        saved_counts = checkpoint.get('hybrid_training_token_counts')
        if saved_counts is not None:
            self._training_token_counts.copy_(saved_counts.to(
                self._training_token_counts.device))
        self.gamma_bin_diagnostics = copy.deepcopy(
            checkpoint.get('hybrid_gamma_bin_diagnostics', {}))
        super().on_load_checkpoint(checkpoint)

    def on_train_start(self):
        super().on_train_start()
        if 'initial' in self.embedding_diagnostics:
            return
        diagnostic = self.embedding_nearest_neighbor_diagnostic()
        self.embedding_diagnostics['initial'] = {
            key: float(value.detach().cpu())
            for key, value in diagnostic.items()}

    def on_train_end(self):
        diagnostic = self.embedding_nearest_neighbor_diagnostic()
        self.embedding_diagnostics['final'] = {
            key: float(value.detach().cpu())
            for key, value in diagnostic.items()}
        if self.classification_prototype_mode == 'independent':
            diagnostic = self.classification_prototype_diagnostic()
            self.prototype_diagnostics['final'] = {
                key: float(value.detach().cpu())
                for key, value in diagnostic.items()}
