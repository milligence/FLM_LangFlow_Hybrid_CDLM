"""Final Task1 F/P finite-map contract on the legacy vocabulary FLM.

The scientific constants live in the delivered YAML contract.  This module is
only the repository binding: global logical plans, legacy-local anchoring,
explicit layer-JVP finite queries, canonical target EMA, and F/P objectives.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import yaml

from algo import LangFlowFLMHybrid
from langflow_hybrid.ops import (
    detached_self_conditioning_embedding,
    flm_vocab_gaussian_bias,
    langflow_token_bias_weight,
)
from trainer_base import Loss


CONTRACT_VERSION = 'task1-tvm-50k-final-v1.0.0'


def _linear_knots(step, knots):
    if step <= knots[0][0]:
        return float(knots[0][1])
    for (left, left_value), (right, right_value) in zip(knots, knots[1:]):
        if left <= step < right:
            fraction = (step - left) / (right - left)
            return float(left_value + fraction * (right_value - left_value))
    return float(knots[-1][1])


def _stage(stages, step):
    for value in stages:
        if int(value['start_step']) <= step < int(value['end_step']):
            return value
    raise ValueError(f'No contracted stage for completed_updates={step}.')


def _rho_map(sampler, step):
    value = _stage(sampler['map']['stages'], step)
    width = value['end_step'] - value['start_step']
    fraction = (step - value['start_step']) / width
    return float(
        value['rho_map_start']
        + fraction * (value['rho_map_end'] - value['rho_map_start']))


def _log_softplus(value):
    output = torch.empty_like(value)
    low, high = value < -40, value > 40
    middle = ~(low | high)
    output[low] = value[low]
    output[middle] = F.softplus(value[middle], threshold=50).log()
    output[high] = (
        value[high].log()
        + torch.log1p(torch.log1p(torch.exp(-value[high])) / value[high]))
    return output


def _sequence_mean(token_loss, weights=None):
    per_sequence = token_loss.mean(dim=1)
    if weights is not None:
        per_sequence = per_sequence * weights
    return per_sequence.mean()


def _softmax_jvp(logits, tangent):
    probability = F.softmax(logits.float(), dim=-1)
    probability_tangent = probability * (
        tangent.float() - (probability * tangent.float()).sum(
            dim=-1, keepdim=True))
    return probability, probability_tangent


def _f_map_tokens(logits, tangent, b_raw, b_tangent, eta, teacher):
    eta = eta[:, None, None]
    probability, probability_tangent = _softmax_jvp(logits, tangent)
    correction = b_raw.float() - b_raw.float().mean(dim=-1, keepdim=True)
    correction_tangent = (
        b_tangent.float() - b_tangent.float().mean(dim=-1, keepdim=True))
    value = probability + eta * correction
    value_tangent = (
        probability_tangent + correction + eta * correction_tangent)
    residual = (
        (1.0 - eta) * value
        + eta * (1.0 - eta) * value_tangent
        + eta * value.detach()
        - teacher.detach())
    return 0.5 * residual.square().sum(dim=-1), value


def _student_log_calibration(log_reference, gate, kappa):
    student_numerator = (
        log_reference + math.log(kappa) + _log_softplus(gate / kappa))
    return student_numerator - student_numerator.logsumexp(
        dim=-1, keepdim=True)


def _teacher_log_calibration_chunk(log_reference, log_teacher, kappa):
    ratio = log_teacher - log_reference - math.log(kappa)
    high = ratio > math.log(40.0)
    transform = torch.where(
        high,
        math.log(kappa) + ratio,
        math.log(kappa)
        + F.softplus(ratio.clamp_max(math.log(40.0)).exp(),
                     threshold=50).log())
    numerator = log_reference + transform
    return numerator - numerator.logsumexp(dim=-1, keepdim=True)


def _teacher_log_calibration(
        log_reference, log_teacher, kappa, batch_chunk_size=4,
        chunk_transform=None):
    if log_reference.shape != log_teacher.shape:
        raise ValueError('Teacher and reference log-probabilities must match.')
    transform = chunk_transform or _teacher_log_calibration_chunk
    output = torch.empty_like(log_reference, dtype=torch.float32)
    chunk_size = max(1, int(batch_chunk_size))
    with torch.no_grad():
        for start in range(0, log_reference.shape[0], chunk_size):
            stop = min(start + chunk_size, log_reference.shape[0])
            chunk = transform(
                log_reference[start:stop].detach(),
                log_teacher[start:stop].detach(), kappa)
            output[start:stop].copy_(chunk)
    return output


def _p_map_tokens(logits, tangent, eta, teacher_log, kappa, negative_weight):
    eta_view = eta[:, None, None]
    log_reference = F.log_softmax(logits.float(), dim=-1)
    reference = log_reference.exp()
    velocity = eta_view * (1.0 - eta_view) * tangent.float()
    mean = (reference * velocity).sum(dim=-1, keepdim=True)
    gate = 1.0 + velocity - mean
    raw = reference * gate
    student_log = _student_log_calibration(log_reference, gate, kappa)
    teacher_calibrated = _teacher_log_calibration(
        log_reference, teacher_log, kappa).exp()
    cross_entropy = -(teacher_calibrated * student_log).sum(dim=-1)
    negative_mass = (-raw).clamp_min(0.0).sum(dim=-1)
    return cross_entropy + negative_weight * negative_mass.square(), reference


@dataclass(frozen=True)
class TargetPosterior:
    prob: torch.Tensor
    log_prob: torch.Tensor
    actual_queries: int
    provenance: dict


class Task1TVM50KFinal(LangFlowFLMHybrid):
    """One code snapshot implementing the contracted F and P lines."""

    def __init__(self, config, tokenizer):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(config, tokenizer)
        if self.ema is not None:
            # The repository EMA has a different built-in warmup.  Disable it
            # so optimizer_step can apply the contract's exact beta(j).
            self.ema.num_updates = None
        self.contract_dir = Path(str(
            config.algo.task1_tvm_contract_dir)).expanduser().resolve()
        self.line = str(config.algo.task1_tvm_final_line).upper()
        if self.line not in {'F', 'P'}:
            raise ValueError('task1_tvm_final_line must be F or P.')
        self.train_contract = self._read_yaml(
            f'train_{self.line.lower()}.yaml')
        self.sampler_contract = self._read_yaml(
            f'sampler_{self.line.lower()}.yaml')
        self.eval_contract = self._read_yaml('eval.yaml')
        self._validate_contract()

        self.plan_seed = int(self.train_contract['seed'])
        self.map_size = int(self.train_contract['objective']['map_batch'])
        self.map_microbatch = int(getattr(
            config.algo, 'task1_map_microbatch_override',
            self.train_contract['map_microbatch']))
        if self.map_microbatch not in {
                int(value) for value in self.train_contract[
                    'oom']['map_candidates']}:
            raise ValueError('Map microbatch override is outside OOM contract.')
        self.target_decay = float(self.train_contract['ema']['target_decay'])
        self.eval_decay = float(self.train_contract['ema']['eval_decay'])
        self.kappa = self.train_contract['objective']['kappa']
        self.negative_mass_weight = self.train_contract[
            'objective']['negative_mass_weight']
        self.training_target_ema = copy.deepcopy(self.backbone)
        self.training_target_ema.requires_grad_(False)
        self.training_target_ema.eval()
        self.register_buffer(
            'training_target_ema_updates', torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'map_calibration', torch.ones((), dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'closure_calibration', torch.ones((), dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            'canonical_k', torch.ones((), dtype=torch.long), persistent=True)
        self.register_buffer(
            'canonical_previous_k', torch.ones((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'canonical_transition_start',
            torch.full((), -1, dtype=torch.long), persistent=True)
        self.register_buffer(
            'controller_revision', torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'last_calibration_step', torch.full((), -1, dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'completed_updates_counter', torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'cumulative_sequences', torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'cumulative_tokens', torch.zeros((), dtype=torch.long),
            persistent=True)
        self.register_buffer(
            'gpu_wall_seconds', torch.zeros((), dtype=torch.float64),
            persistent=True)
        self._compiled_finite_jvp = None
        self._compiled_student_calibration = None
        self._compiled_teacher_calibration_chunk = None
        self._last_plan_audit = None
        self._canonical_gate_tokens_cpu = None
        self._calibration_tokens_cpu = None
        self._canonical_gate_events = []
        self._calibration_events = []
        self._gradient_audit_events = []
        self._generation_sample_offset = 0
        self._evaluation_online_backbone = None
        self._last_optimizer_wall_time = None
        self._validation_batch_index = 0

    def _read_yaml(self, name):
        path = self.contract_dir / name
        with path.open('r', encoding='utf-8') as handle:
            value = yaml.safe_load(handle)
        if value.get('contract_version') != CONTRACT_VERSION:
            raise ValueError(f'{path} has the wrong contract_version.')
        return value

    def _validate_contract(self):
        train, sampler = self.train_contract, self.sampler_contract
        if train['line'] != self.line or sampler['line'] != self.line:
            raise ValueError('Train/sampler line does not match the Hydra line.')
        exact = {
            'global_batch': 256, 'seq_len': 128, 'vocab_size': 50257,
            'total_optimizer_steps': 50000,
        }
        for name, expected in exact.items():
            if int(train[name]) != expected:
                raise ValueError(f'Contracted {name} must be {expected}.')
        if int(self.config.loader.global_batch_size) != 256:
            raise ValueError('Hydra global batch must be the contracted 256.')
        if int(self.config.model.length) != 128 or self.vocab_size != 50257:
            raise ValueError('Hydra model shape does not match the contract.')
        if not math.isclose(float(self.config.model.dropout), 0.1):
            raise ValueError('Legacy dropout must remain 0.1.')
        if bool(getattr(self.config.model, 'qk_norm', False)):
            raise ValueError('The final contract forbids new QK RMSNorm.')
        if self.line == 'F' and self.backbone.finite_correction_head is None:
            raise ValueError('F requires the centered free correction head.')
        if self.line == 'P' and self.backbone.finite_correction_head is not None:
            raise ValueError('P must not construct an F correction head.')

    def task1_training_time_contract(self):
        return {
            'workflow': 'TASK1-TVM-50K-FINAL',
            'contract_version': CONTRACT_VERSION,
            'line': self.line,
            'contract_dir': str(self.contract_dir),
            'global_local_batch': 256,
            'global_map_subset': 96,
            'jvp_backend': 'explicit_layer_jvp_compile_stable_backbone_only',
        }

    def setup(self, stage):
        del stage
        self.training_target_ema.requires_grad_(False)
        self.training_target_ema.eval()

    def on_train_start(self):
        super().on_train_start()
        self._last_optimizer_wall_time = time.monotonic()
        if self._canonical_gate_tokens_cpu is None:
            from packed_dataset import PackedTokenDataset
            dataset = PackedTokenDataset(
                self.config.data.packed_dir, 'validation')
            tokens = []
            for index in range(256):
                item = dataset[index]
                value = item['input_ids'] if isinstance(item, dict) else item
                tokens.append(torch.as_tensor(value, dtype=torch.long))
            self._canonical_gate_tokens_cpu = torch.stack(tokens).cpu()
        if self._calibration_tokens_cpu is None:
            from packed_dataset import PackedTokenDataset
            dataset = PackedTokenDataset(self.config.data.packed_dir, 'train')
            tokens = []
            # This immutable bank is disjoint from the live sampler cursor.
            offset = int(self.plan_seed % max(1, len(dataset) - 128))
            for index in range(offset, offset + 128):
                item = dataset[index]
                value = item['input_ids'] if isinstance(item, dict) else item
                tokens.append(torch.as_tensor(value, dtype=torch.long))
            self._calibration_tokens_cpu = torch.stack(tokens).cpu()

    def _seed(self, stream, step, logical_index=0):
        stream_id = int.from_bytes(
            hashlib.sha256(stream.encode('utf-8')).digest()[:4], 'little')
        return int(
            (self.plan_seed + stream_id + step * 1000003
             + logical_index * 10007) % (2 ** 63 - 1))

    def _generator(self, stream, step, logical_index=0):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self._seed(stream, step, logical_index))
        return generator

    def _noise_rows(self, count, length, stream, step, global_indices):
        rows = []
        for index in global_indices.tolist():
            rows.append(torch.randn(
                (1, length, self.vocab_size), device=self.device,
                dtype=torch.float32,
                generator=self._generator(stream, step, int(index))))
        if not rows:
            return torch.empty(
                (0, length, self.vocab_size), device=self.device,
                dtype=torch.float32)
        if len(rows) != count:
            raise RuntimeError('Noise-row logical index mismatch.')
        return torch.cat(rows, dim=0)

    def _draw(self, generator, low, high, law):
        if not low < high:
            raise ValueError(f'Invalid contracted draw interval [{low}, {high}].')
        unit = torch.rand((), device=self.device, generator=generator).clamp_min(
            torch.finfo(torch.float32).tiny)
        if law == 'uniform':
            return float(low + (high - low) * unit)
        if law == 'log_uniform' and low > 0.0:
            return float(torch.exp(
                math.log(low) + (math.log(high) - math.log(low)) * unit))
        raise ValueError(f'Unsupported contracted law {law!r}.')

    def _local_components(self, step):
        local = self.sampler_contract['local']
        stage = _stage(local['stages'], step)
        if stage['law'] == 'uniform_tau':
            return [{'law': 'uniform_tau', 'low': 0.0, 'high': 1.0,
                     'weight': 1.0}]
        output = []
        for spec, quota in zip(local['bins'], stage['quotas']):
            value = dict(spec)
            value['weight'] = float(quota) / 256.0
            if value.get('high') == 'inverse_tau(1/128)':
                point = torch.tensor([1.0 / 128.0], device=self.device)
                value['high'] = float(self._task1_physical_time(point)[0])
            output.append(value)
        return output

    def _local_times(self, step, generator):
        stage = _stage(self.sampler_contract['local']['stages'], step)
        if stage['law'] == 'uniform_tau':
            tau = torch.rand(
                256, device=self.device, generator=generator).clamp_min(
                    torch.finfo(torch.float32).tiny)
            return self._task1_physical_time(tau)
        values = []
        for spec, quota in zip(
                self._local_components(step), stage['quotas']):
            count = int(quota)
            if spec['law'] == 'atom':
                values.append(torch.full(
                    (count,), float(spec['t']), device=self.device))
                continue
            draw = torch.rand(count, device=self.device, generator=generator)
            coordinate = (
                float(spec['low'])
                + (float(spec['high']) - float(spec['low'])) * draw)
            values.append(
                self._task1_physical_time(coordinate)
                if spec['law'] == 'uniform_tau' else coordinate)
        result = torch.cat(values)
        return result[torch.randperm(256, device=self.device, generator=generator)]

    def _conditioned_source(self, step, r_max, generator):
        choices, masses = [], []
        tau_max = float(self._t_to_tau(torch.tensor(
            [r_max], device=self.device))[0])
        for spec in self._local_components(step):
            if spec['law'] == 'atom':
                if float(spec['t']) < r_max:
                    choices.append((spec, None, None))
                    masses.append(float(spec['weight']))
                continue
            low, high = float(spec['low']), float(spec['high'])
            upper = min(high, tau_max if spec['law'] == 'uniform_tau' else r_max)
            if upper > low:
                choices.append((spec, low, upper))
                masses.append(float(spec['weight']) * (upper - low) / (high - low))
        probabilities = torch.tensor(masses, device=self.device)
        if not choices or float(probabilities.sum()) <= 0.0:
            raise ValueError('Conditioned local source law has zero mass.')
        index = int(torch.multinomial(
            probabilities, 1, generator=generator).item())
        spec, low, upper = choices[index]
        if spec['law'] == 'atom':
            return float(spec['t'])
        coordinate = self._draw(generator, low, upper, 'uniform')
        if spec['law'] == 'uniform_tau':
            return float(self._task1_physical_time(torch.tensor(
                [coordinate], device=self.device))[0])
        return coordinate

    def _map_rows(self, step, generator):
        map_config = self.sampler_contract['map']
        stage = _stage(map_config['stages'], step)
        if not int(stage['map_batch']):
            return []
        cap = float(stage['terminal_cap'])
        rows = []
        for label, raw_count in stage['counts'].items():
            count = int(raw_count)
            spec = map_config['classes'][label]
            kind = spec['kind']
            if kind == 'deployment':
                grids = map_config['deployment']['grids']
                indices = spec['interval_indices']
                per = count // (len(grids) * len(indices))
                if per * len(grids) * len(indices) != count:
                    raise ValueError('Deployment quota is not balanced.')
                for grid_name, grid_values in grids.items():
                    for interval in indices:
                        for sample_index in range(per):
                            nodes = list(float(value) for value in grid_values)
                            jittered = (sample_index + interval + step) % 2 == 1
                            if jittered:
                                low, high = map_config['deployment'][
                                    'jitter_interior_uniform']
                                for node in range(1, len(nodes) - 1):
                                    nodes[node] += self._draw(
                                        generator, float(low), float(high),
                                        'uniform')
                            if any(a >= b for a, b in zip(nodes, nodes[1:])):
                                raise ValueError('Jitter produced a nonmonotone grid.')
                            r, s = nodes[interval], nodes[interval + 1]
                            rows.append(self._row(
                                label, r, s, interval, grid_name, jittered,
                                step))
                continue
            for _ in range(count):
                if kind == 'full_span':
                    r, s = float(spec['r']), float(spec['s'])
                elif kind == 'prefix':
                    r = float(spec['r'])
                    eta = self._draw(
                        generator, float(spec['eta_low']),
                        min(float(spec['eta_high']), cap), spec['eta_law'])
                    s = r + (1.0 - r) * eta
                elif kind == 'continuous':
                    eta_low, eta_high = (
                        float(spec['eta_low']), float(spec['eta_high']))
                    r_max = (cap - eta_low) / (1.0 - eta_low)
                    r = self._conditioned_source(step, r_max, generator)
                    eta_upper = min(eta_high, (cap - r) / (1.0 - r))
                    eta = self._draw(
                        generator, eta_low, eta_upper, spec['eta_law'])
                    s = r + (1.0 - r) * eta
                else:
                    raise ValueError(f'Unknown map class kind {kind!r}.')
                rows.append(self._row(label, r, s, None, None, False, step))
        if len(rows) != int(stage['map_batch']):
            raise ValueError('Contracted map quota mismatch.')
        order = torch.randperm(len(rows), device=self.device, generator=generator)
        return [rows[index] for index in order.tolist()]

    def _row(self, label, r, s, interval, grid, jittered, step):
        if not 0.0 <= r < s <= 0.9500001:
            raise ValueError(f'Infeasible map pair r={r}, s={s}.')
        hard = label in {'L', 'H'} or (label == 'D' and interval == 3)
        hard_config = self.sampler_contract['map']['hard_weight']
        hard_weight = (
            1.0 if step < int(hard_config['start_step']) else _linear_knots(
                step, ((hard_config['start_step'], hard_config['start_value']),
                       (hard_config['end_step'], hard_config['end_value']))))
        return {
            'class': label, 'r': r, 's': s,
            'eta': (s - r) / (1.0 - r),
            'interval': interval, 'grid': grid, 'jittered': jittered,
            'hard': hard, 'weight': hard_weight if hard else 1.0,
        }

    def sample_step_plan(self, step, accumulation_step):
        local_times = self._local_times(
            step, self._generator('local_time', step))
        rows = self._map_rows(step, self._generator('map_pair', step))
        subset = torch.randperm(
            256, device=self.device,
            generator=self._generator('map_subset', step))[:len(rows)]
        map_mask = torch.zeros(256, dtype=torch.bool, device=self.device)
        map_mask[subset] = True
        row_for_global = [None] * 256
        for global_index, row in zip(subset.tolist(), rows):
            row_for_global[global_index] = row
        closure_candidates = [
            index for index, row in enumerate(rows)
            if row['class'] in {'S', 'M'}]
        if (self._canonical_non_k1_mass(step) > 0.0
                and len(closure_candidates) >= 32):
            order = torch.randperm(
                len(closure_candidates), device=self.device,
                generator=self._generator('closure_subset', step))[:32]
            for offset in order.tolist():
                rows[closure_candidates[offset]]['closure'] = True
        for row in rows:
            row.setdefault('closure', False)
        sc_groups = torch.rand(
            8, device=self.device,
            generator=self._generator('local_sc', step)) < 0.25
        start = int(accumulation_step) * int(self.config.loader.batch_size)
        stop = start + int(self.config.loader.batch_size)
        indices = torch.arange(start, stop, device=self.device)
        plan = {
            'global_indices': indices,
            'local_t': local_times[start:stop],
            'local_sc': sc_groups[(indices // 32).long()],
            'map_mask': map_mask[start:stop],
            'map_rows': [row_for_global[index] for index in range(start, stop)],
        }
        if accumulation_step == 0:
            self._last_plan_audit = {
                'step': step, 'subset': subset.detach().cpu(),
                'local_t': local_times.detach().cpu(), 'rows': rows,
                'sc_groups': sc_groups.detach().cpu(),
            }
        return plan

    def local_posterior(self, state, t, sc_cache=None, *, model=None,
                        dropout_enabled=True, bias_step=None,
                        need_log_probability=True, need_probability=True):
        model = self.backbone if model is None else model
        previous_mode = model.training
        model.train(bool(dropout_enabled))
        tau = self._t_to_tau(t.float().clamp(0.0, 1.0))
        condition = self._task1_condition(tau, t).to(self.device)
        residual = model(
            model.vocab_embed(state), self._process_sigma(condition),
            inputs_are_embeddings=True, x_self_cond=sc_cache)
        bias_weight = (
            self._current_token_bias_weight() if bias_step is None else
            langflow_token_bias_weight(
                int(bias_step), self.token_bias_warmup_steps))
        bias = flm_vocab_gaussian_bias(
            state, t, bias_weight, self.flm_time_eps)
        model.train(previous_mode)
        logits = residual.float() + bias.float()
        log_probability = (
            F.log_softmax(logits, dim=-1) if need_log_probability else None)
        probability = F.softmax(logits, dim=-1) if need_probability else None
        return logits, log_probability, probability

    def _finite_jvp_call(self, *args, **kwargs):
        if self._compiled_finite_jvp is None:
            target = self.backbone.forward_with_jvp
            if bool(getattr(
                    self.config.algo, 'task1_compile_finite_jvp', True)):
                target = torch.compile(
                    target, dynamic=True,
                    options={
                        'max_autotune': True,
                        'triton.cudagraphs': False,
                    })
            self._compiled_finite_jvp = target
        return self._compiled_finite_jvp(*args, **kwargs)

    def finite_with_eta_jvp(self, state, r, eta):
        tau = self._t_to_tau(r.float().clamp(0.0, 1.0))
        condition = self._task1_condition(tau, r).to(self.device)
        features = torch.stack((r.float(), eta.float()), dim=-1)
        feature_tangent = torch.stack(
            (torch.zeros_like(eta), torch.ones_like(eta)), dim=-1)
        embedding = self.backbone.vocab_embed(state)
        zeros = torch.zeros_like(embedding)
        previous_mode = self.backbone.training
        self.backbone.eval()
        try:
            if self.line == 'F':
                ((residual, output_features),
                 (tangent, feature_tangent_out)) = self._finite_jvp_call(
                    embedding, zeros, self._process_sigma(condition),
                    sigma_jvp=torch.zeros_like(self._process_sigma(condition)),
                    inputs_are_embeddings=True, x_self_cond=None,
                    x_self_cond_jvp=None, finite_time_features=features,
                    finite_time_features_jvp=feature_tangent,
                    return_output_features=True)
            else:
                residual, tangent = self._finite_jvp_call(
                    embedding, zeros, self._process_sigma(condition),
                    sigma_jvp=torch.zeros_like(self._process_sigma(condition)),
                    inputs_are_embeddings=True, x_self_cond=None,
                    x_self_cond_jvp=None, finite_time_features=features,
                    finite_time_features_jvp=feature_tangent)
        finally:
            self.backbone.train(previous_mode)
        bias = flm_vocab_gaussian_bias(
            state, r, self._current_token_bias_weight(), self.flm_time_eps)
        output = {
            'logits': residual.float() + bias.float(),
            'dlogits': tangent.float(),
        }
        if self.line == 'F':
            head = self.backbone.finite_correction_head
            output['b_raw'] = head(output_features).float()
            output['db_raw'] = F.linear(
                feature_tangent_out, head.weight, None).float()
        return output

    def _p_objective_tokens(self, logits, tangent, eta, teacher_log):
        eta_view = eta[:, None, None]
        log_reference = F.log_softmax(logits.float(), dim=-1)
        reference = log_reference.exp()
        velocity = eta_view * (1.0 - eta_view) * tangent.float()
        mean = (reference * velocity).sum(dim=-1, keepdim=True)
        gate = 1.0 + velocity - mean
        raw = reference * gate
        student_transform = _student_log_calibration
        teacher_transform = _teacher_log_calibration_chunk
        if bool(getattr(
                self.config.algo, 'task1_compile_calibration_kernels', True)):
            options = {
                'max_autotune': True,
                'triton.cudagraphs': False,
            }
            if self._compiled_student_calibration is None:
                self._compiled_student_calibration = torch.compile(
                    _student_log_calibration, dynamic=True, options=options)
            if self._compiled_teacher_calibration_chunk is None:
                self._compiled_teacher_calibration_chunk = torch.compile(
                    _teacher_log_calibration_chunk,
                    dynamic=True, options=options)
            student_transform = self._compiled_student_calibration
            teacher_transform = self._compiled_teacher_calibration_chunk
        student_log = student_transform(
            log_reference, gate, float(self.kappa))
        teacher_calibrated = _teacher_log_calibration(
            log_reference, teacher_log, float(self.kappa),
            batch_chunk_size=4,
            chunk_transform=teacher_transform).exp()
        cross_entropy = -(teacher_calibrated * student_log).sum(dim=-1)
        negative_mass = (-raw).clamp_min(0.0).sum(dim=-1)
        return (
            cross_entropy
            + float(self.negative_mass_weight) * negative_mass.square(),
            reference)

    @torch.no_grad()
    def _target_chain(self, state, t, query_count, bias_step=None):
        cache = None
        output = []
        for _ in range(query_count):
            _, log_probability, probability = self.local_posterior(
                state, t, cache, model=self.training_target_ema,
                dropout_enabled=False, bias_step=bias_step)
            probability = probability.detach()
            log_probability = log_probability.detach()
            output.append((probability, log_probability))
            cache = detached_self_conditioning_embedding(
                probability, self.training_target_ema.vocab_embed).detach()
        return output

    def _finalize_canonical_transition(self, step):
        start = int(self.canonical_transition_start)
        if start < 0:
            return
        width = int(self.train_contract['canonical_sc']['transition_steps'])
        if step - start >= width:
            self.canonical_previous_k.copy_(self.canonical_k)
            self.canonical_transition_start.fill_(-1)
            self.controller_revision.add_(1)

    @torch.no_grad()
    def canonical_target(self, state, t):
        self.training_target_ema.eval()
        self._finalize_canonical_transition(int(self.global_step))
        current_k = int(self.canonical_k)
        old_k = int(self.canonical_previous_k)
        start = int(self.canonical_transition_start)
        transition = int(self.train_contract['canonical_sc']['transition_steps'])
        alpha = (
            1.0 if start < 0 else
            min(max((int(self.global_step) - start) / transition, 0.0), 1.0))

        chain = self._target_chain(state, t, max(old_k, current_k))
        old_probability, old_log = chain[old_k - 1]
        if current_k == old_k or alpha <= 0.0:
            probability, log_probability = old_probability, old_log
            queries = old_k
        else:
            new_probability, new_log = chain[current_k - 1]
            queries = max(old_k, current_k)
            if alpha >= 1.0:
                probability, log_probability = new_probability, new_log
            else:
                log_probability = torch.logaddexp(
                    old_log + math.log1p(-alpha),
                    new_log + math.log(alpha))
                probability = log_probability.exp()
        return TargetPosterior(
            prob=probability.detach(), log_prob=log_probability.detach(),
            actual_queries=queries,
            provenance={
                'K_old': old_k, 'K_new': current_k, 'alpha': alpha,
                'bias_step': int(self.global_step),
                'target_ema_updates': int(self.training_target_ema_updates),
            })

    def _gate_token_loss(
            self, probability, clean_tokens, log_probability=None):
        if self.line == 'F':
            return 0.5 * (
                probability.square().sum(dim=-1)
                - 2.0 * probability.gather(
                    -1, clean_tokens.unsqueeze(-1)).squeeze(-1) + 1.0)
        if log_probability is None:
            raise ValueError('P canonical gate requires native log-probability.')
        return -log_probability.gather(
            -1, clean_tokens.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _paired_gate_statistics(old, new):
        difference = old - new
        count = max(1, difference.numel())
        standard_error = difference.std(unbiased=True) / math.sqrt(count)
        old_mean = old.mean()
        new_mean = new.mean()
        return {
            'old_mean': float(old_mean), 'new_mean': float(new_mean),
            'improvement_fraction': float(
                (old_mean - new_mean) / old_mean.clamp_min(1e-12)),
            'difference_mean': float(difference.mean()),
            'difference_se': float(standard_error),
        }

    @torch.no_grad()
    def _run_canonical_gate(self, completed_step, candidate_k):
        if self._canonical_gate_tokens_cpu is None:
            raise RuntimeError('Canonical fixed validation bank is unavailable.')
        current_k = int(self.canonical_k)
        if candidate_k <= current_k:
            return
        self.training_target_ema.eval()
        nodes = torch.linspace(0.0, 1.0, 51, device=self.device)
        old_values, new_values = [], []
        sensitive = {0: ([], []), 1: ([], []), 2: ([], [])}
        p21_square = 0.0
        p43_square = 0.0
        p21_count = 0
        p43_count = 0
        query_batch = int(getattr(
            self.config.algo, 'task1_canonical_gate_batch', 4))
        for node_index, node in enumerate(nodes):
            t_value = min(float(node), 1.0 - self.flm_time_eps)
            for start in range(0, 256, query_batch):
                stop = min(start + query_batch, 256)
                clean = self._canonical_gate_tokens_cpu[start:stop].to(
                    self.device)
                t = torch.full(
                    (stop - start,), t_value, device=self.device)
                global_indices = torch.arange(start, stop, device=self.device)
                noise = self._noise_rows(
                    stop - start, clean.shape[1],
                    f'canonical_gate_{node_index}', completed_step,
                    global_indices)
                state = self.corrupt_vocab_state(clean, t, noise=noise)
                chain = self._target_chain(
                    state, t, candidate_k, bias_step=completed_step)
                old_probability, old_log_probability = chain[current_k - 1]
                new_probability, new_log_probability = chain[candidate_k - 1]
                old_loss = self._gate_token_loss(
                    old_probability, clean,
                    old_log_probability).mean(dim=1).cpu()
                new_loss = self._gate_token_loss(
                    new_probability, clean,
                    new_log_probability).mean(dim=1).cpu()
                old_values.append(old_loss)
                new_values.append(new_loss)
                for bin_index, (low, high) in enumerate(
                        self.train_contract['canonical_sc']['gate'][
                            'sensitive_bins']):
                    if float(low) <= t_value < float(high):
                        sensitive[bin_index][0].append(old_loss)
                        sensitive[bin_index][1].append(new_loss)
                if len(chain) >= 2:
                    delta = chain[1][0] - chain[0][0]
                    p21_square += float(delta.square().mean())
                    p21_count += 1
                if len(chain) >= 4:
                    delta = chain[3][0] - chain[2][0]
                    p43_square += float(delta.square().mean())
                    p43_count += 1
        old = torch.cat(old_values)
        new = torch.cat(new_values)
        statistics = self._paired_gate_statistics(old, new)
        gate = self.train_contract['canonical_sc']['gate']
        accepted = (
            statistics['improvement_fraction']
            >= float(gate['require_mean_improvement_fraction'])
            and statistics['difference_mean']
            >= float(gate['require_paired_standard_errors'])
            * statistics['difference_se'])
        sensitive_statistics = []
        for old_parts, new_parts in sensitive.values():
            if not old_parts:
                continue
            value = self._paired_gate_statistics(
                torch.cat(old_parts), torch.cat(new_parts))
            worsening_fraction = -value['improvement_fraction']
            worsening = -value['difference_mean']
            if (worsening_fraction
                    > float(gate['max_sensitive_bin_worsening_fraction'])
                    or worsening > float(gate['max_sensitive_bin_worsening_SE'])
                    * value['difference_se']):
                accepted = False
            sensitive_statistics.append(value)
        incremental = None
        if candidate_k == 4:
            incremental = {
                'p4_minus_p3_mse': p43_square / max(1, p43_count),
                'p2_minus_p1_mse': p21_square / max(1, p21_count),
            }
            if incremental['p4_minus_p3_mse'] > incremental['p2_minus_p1_mse']:
                accepted = False
        event = {
            'completed_updates': completed_step,
            'current_K': current_k, 'candidate_K': candidate_k,
            'accepted': bool(accepted), 'statistics': statistics,
            'sensitive_bins': sensitive_statistics,
            'increment_condition': incremental,
        }
        self._canonical_gate_events.append(event)
        if self.trainer.is_global_zero:
            write_controller_event(
                Path(self.config.checkpointing.save_dir)
                / 'canonical_gate_events.jsonl', event)
        if accepted:
            self.canonical_previous_k.fill_(current_k)
            self.canonical_k.fill_(candidate_k)
            self.canonical_transition_start.fill_(completed_step)
        self.controller_revision.add_(1)

    def _local_objective(
            self, clean_tokens, valid_tokens, plan, step,
            noise_stream='local_noise'):
        indices = plan['global_indices']
        noise = self._noise_rows(
            clean_tokens.shape[0], clean_tokens.shape[1],
            noise_stream, step, indices)
        state = self.corrupt_vocab_state(clean_tokens, plan['local_t'], noise=noise)
        cache = None
        if bool(plan['local_sc'].any()):
            with torch.no_grad():
                _, _, cold = self.local_posterior(
                    state, plan['local_t'], None, dropout_enabled=True,
                    need_log_probability=False)
                proposed = detached_self_conditioning_embedding(
                    cold, self.embed_probabilities)
                cache = torch.zeros_like(proposed)
                cache[plan['local_sc']] = proposed[plan['local_sc']]
        logits, log_probability, probability = self.local_posterior(
            state, plan['local_t'], cache, dropout_enabled=True,
            need_log_probability=self.line == 'P',
            need_probability=self.line == 'F')
        if self.line == 'F':
            token_loss = 0.5 * (
                probability.square().sum(dim=-1)
                - 2.0 * probability.gather(
                    -1, clean_tokens.unsqueeze(-1)).squeeze(-1) + 1.0)
        else:
            token_loss = -log_probability.gather(
                -1, clean_tokens.unsqueeze(-1)).squeeze(-1)
        mask = valid_tokens.float()
        return (token_loss * mask).sum() / mask.sum().clamp_min(1.0), token_loss

    def _map_objective(
            self, clean_tokens, plan, step, noise_stream='map_noise'):
        selected_local = plan['map_mask'].nonzero(as_tuple=True)[0]
        if not selected_local.numel():
            zero = clean_tokens.new_zeros((), dtype=torch.float32)
            return zero, 0, zero, 0
        rows = [plan['map_rows'][index] for index in selected_local.tolist()]
        global_indices = plan['global_indices'][selected_local]
        tokens = clean_tokens[selected_local]
        total = clean_tokens.new_zeros((), dtype=torch.float32)
        closure_total = clean_tokens.new_zeros((), dtype=torch.float32)
        closure_count = 0
        for begin in range(0, len(rows), self.map_microbatch):
            end = min(begin + self.map_microbatch, len(rows))
            chunk_rows = rows[begin:end]
            chunk_tokens = tokens[begin:end]
            chunk_indices = global_indices[begin:end]
            r = torch.tensor(
                [row['r'] for row in chunk_rows], device=self.device)
            s = torch.tensor(
                [row['s'] for row in chunk_rows], device=self.device)
            eta = torch.tensor(
                [row['eta'] for row in chunk_rows], device=self.device)
            weights = torch.tensor(
                [row['weight'] for row in chunk_rows], device=self.device)
            noise = self._noise_rows(
                len(chunk_rows), chunk_tokens.shape[1], noise_stream, step,
                chunk_indices)
            state = self.corrupt_vocab_state(chunk_tokens, r, noise=noise)
            output = self.finite_with_eta_jvp(state, r, eta)
            if self.line == 'F':
                _, endpoint_value = _f_map_tokens(
                    output['logits'], output['dlogits'], output['b_raw'],
                    output['db_raw'], eta,
                    torch.zeros_like(output['logits']))
            else:
                endpoint_value = F.softmax(output['logits'], dim=-1)
            endpoint = (
                (1.0 - eta[:, None, None]) * state
                + eta[:, None, None] * endpoint_value)
            target = self.canonical_target(endpoint.detach(), s)
            if self.line == 'F':
                tokens_loss, _ = _f_map_tokens(
                    output['logits'], output['dlogits'], output['b_raw'],
                    output['db_raw'], eta, target.prob)
            else:
                tokens_loss, _ = self._p_objective_tokens(
                    output['logits'], output['dlogits'], eta,
                    target.log_prob)
            total = total + (
                tokens_loss.mean(dim=1) * weights).sum()
            closure_indices = [
                index for index, row in enumerate(chunk_rows)
                if row['closure']]
            if closure_indices:
                chosen = torch.tensor(
                    closure_indices, device=self.device, dtype=torch.long)
                cold_logits, _, cold_probability = self.local_posterior(
                    endpoint[chosen].detach(), s[chosen], None,
                    dropout_enabled=False)
                if self.line == 'F':
                    closure_tokens = 0.5 * (
                        cold_probability - target.prob[chosen].detach()
                    ).square().sum(dim=-1)
                else:
                    closure_tokens = -(
                        target.prob[chosen].detach()
                        * F.log_softmax(cold_logits.float(), dim=-1)
                    ).sum(dim=-1)
                closure_total = closure_total + closure_tokens.mean(dim=1).sum()
                closure_count += len(closure_indices)
        closure_mean = (
            closure_total / float(closure_count)
            if closure_count else closure_total)
        return total / float(len(rows)), len(rows), closure_mean, closure_count

    def _lambda_map(self, step):
        return _rho_map(self.sampler_contract, step) * float(self.map_calibration)

    def _canonical_non_k1_mass(self, step):
        current = int(self.canonical_k)
        previous = int(self.canonical_previous_k)
        start = int(self.canonical_transition_start)
        if start < 0:
            return 1.0 if current > 1 else 0.0
        transition = int(self.train_contract['canonical_sc']['transition_steps'])
        alpha = min(max((step - start) / transition, 0.0), 1.0)
        return ((1.0 - alpha) if previous > 1 else 0.0) + (
            alpha if current > 1 else 0.0)

    def _lambda_close(self, step):
        rho = _linear_knots(
            step, self.train_contract['objective']['rho_closure_schedule'])
        return (
            rho * float(self.closure_calibration)
            * self._canonical_non_k1_mass(step))

    def _calibration_due(self, step):
        specification = self.train_contract['gradient_calibration']
        first = int(specification['first_step'])
        if step < first or step == int(self.last_calibration_step):
            return False
        if step <= 10000:
            cadence = int(specification['recalibrate_every_steps_until_10k'])
            return (step - first) % cadence == 0
        return step in {
            int(value) for value in specification['recalibrate_after_10k']}

    def _shared_calibration_parameters(self):
        excluded_prefixes = (
            'finite_time_conditioner.', 'finite_correction_head.')
        named = [
            (name, parameter)
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
            and not name.startswith(excluded_prefixes)]
        named.sort(key=lambda item: item[0])
        return named

    @staticmethod
    def _accumulated_gradient_norm(scalars, parameters):
        accumulated = [None] * len(parameters)
        for scalar, multiplier in scalars:
            gradients = torch.autograd.grad(
                scalar * float(multiplier), parameters,
                retain_graph=False, allow_unused=True)
            for index, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                value = gradient.detach().float()
                if accumulated[index] is None:
                    accumulated[index] = value
                else:
                    accumulated[index].add_(value)
        squares = [value.square().sum() for value in accumulated
                   if value is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())

    def _calibration_pack_norms(
            self, step, pack_index, parameters, raw_class_weights=True):
        if self._calibration_tokens_cpu is None:
            raise RuntimeError('Calibration train-only bank is unavailable.')
        pack_tokens = self._calibration_tokens_cpu[
            pack_index * 32:(pack_index + 1) * 32].to(self.device)
        valid = torch.ones_like(pack_tokens, dtype=torch.long)
        time_plan = self._local_times(
            step, self._generator('calibration_local_time', step, pack_index))[:32]
        sc_enabled = bool(torch.rand(
            (), device=self.device,
            generator=self._generator(
                'calibration_local_sc', step, pack_index)) < 0.25)
        local_chunk = int(self.config.loader.batch_size)
        def local_scalars():
            for begin in range(0, 32, local_chunk):
                end = min(begin + local_chunk, 32)
                count = end - begin
                plan = {
                    'global_indices': torch.arange(
                        pack_index * 32 + begin,
                        pack_index * 32 + end, device=self.device),
                    'local_t': time_plan[begin:end],
                    'local_sc': torch.full(
                        (count,), sc_enabled, device=self.device,
                        dtype=torch.bool),
                }
                scalar, _ = self._local_objective(
                    pack_tokens[begin:end], valid[begin:end], plan, step,
                    noise_stream='calibration_local_noise')
                yield scalar, count / 32.0
        local_norm = self._accumulated_gradient_norm(
            local_scalars(), parameters)

        all_rows = self._map_rows(
            step, self._generator('calibration_map_pair', step))
        if len(all_rows) != 96:
            raise RuntimeError('Calibration requires an active 96-row map stage.')
        rows = [dict(
                    row,
                    weight=(1.0 if raw_class_weights else row['weight']),
                    closure=False)
                for row in all_rows[pack_index * 24:(pack_index + 1) * 24]]
        def map_scalars():
            for begin in range(0, 24, self.map_microbatch):
                end = min(begin + self.map_microbatch, 24)
                count_expected = end - begin
                plan = {
                    'global_indices': torch.arange(
                        pack_index * 24 + begin,
                        pack_index * 24 + end, device=self.device),
                    'map_mask': torch.ones(
                        count_expected, device=self.device, dtype=torch.bool),
                    'map_rows': rows[begin:end],
                }
                scalar, count, _, _ = self._map_objective(
                    pack_tokens[begin:end], plan, step,
                    noise_stream='calibration_map_noise')
                if count != count_expected:
                    raise RuntimeError('Calibration map pack lost rows.')
                yield scalar, count / 24.0
        map_norm = self._accumulated_gradient_norm(
            map_scalars(), parameters)

        eligible = [
            dict(row, weight=1.0, closure=True)
            for row in all_rows if row['class'] in {'S', 'M'}]
        closure_norm = None
        if len(eligible) >= 32:
            order = torch.randperm(
                len(eligible), device=self.device,
                generator=self._generator(
                    'calibration_closure_subset', step))
            selected = [
                eligible[index] for index in order[
                    pack_index * 8:(pack_index + 1) * 8].tolist()]
            closure_plan = {
                'global_indices': torch.arange(
                    pack_index * 8, (pack_index + 1) * 8,
                    device=self.device),
                'map_mask': torch.ones(
                    8, device=self.device, dtype=torch.bool),
                'map_rows': selected,
            }
            _, _, closure_scalar, closure_count = self._map_objective(
                pack_tokens[:8], closure_plan, step,
                noise_stream='calibration_closure_noise')
            if closure_count != 8:
                raise RuntimeError('Calibration closure pack lost rows.')
            closure_norm = self._accumulated_gradient_norm(
                ((closure_scalar, 1.0),), parameters)
        return local_norm, map_norm, closure_norm

    def _run_gradient_calibration(self, step):
        specification = self.train_contract['gradient_calibration']
        named = self._shared_calibration_parameters()
        names = [name for name, _ in named]
        parameters = tuple(parameter for _, parameter in named)
        fingerprint = hashlib.sha256(
            '\n'.join(names).encode('utf-8')).hexdigest()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        prior_mode = self.backbone.training
        ratios = []
        closure_ratios = []
        pack_records = []
        try:
            for pack_index in range(int(specification['probe_packs'])):
                local_norm, map_norm, closure_norm = self._calibration_pack_norms(
                    step, pack_index, parameters)
                if (not math.isfinite(local_norm)
                        or not math.isfinite(map_norm)
                        or min(local_norm, map_norm) < 0.0):
                    raise FloatingPointError('Invalid gradient calibration norm.')
                denominator = max(
                    map_norm,
                    float(specification['denominator_local_fraction_floor'])
                    * local_norm,
                    float(specification['absolute_norm_floor']))
                ratios.append(local_norm / denominator)
                if closure_norm is not None:
                    closure_denominator = max(
                        closure_norm,
                        float(specification[
                            'denominator_local_fraction_floor']) * local_norm,
                        float(specification['absolute_norm_floor']))
                    closure_ratios.append(local_norm / closure_denominator)
                pack_records.append({
                    'pack': pack_index, 'local_norm': local_norm,
                    'raw_map_norm': map_norm,
                    'closure_norm': closure_norm,
                })
        finally:
            self.backbone.train(prior_mode)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        value = min(
            float(specification['scale_upper']),
            float(torch.tensor(ratios).median()))
        self.map_calibration.fill_(value)
        if len(closure_ratios) == int(specification['probe_packs']):
            closure_value = min(
                float(specification['scale_upper']),
                float(torch.tensor(closure_ratios).median()))
            self.closure_calibration.fill_(closure_value)
        self.last_calibration_step.fill_(step)
        self.controller_revision.add_(1)
        event = {
            'completed_updates': step,
            'shared_parameter_names_sha256': fingerprint,
            'packs': pack_records, 'ratios': ratios,
            'closure_ratios': closure_ratios,
            'C_map': value,
            'C_close': float(self.closure_calibration),
            'training_rng_restored': True,
            'optimizer_and_grad_buffers_mutated': False,
        }
        self._calibration_events.append(event)
        if self.trainer.is_global_zero:
            write_controller_event(
                Path(self.config.checkpointing.save_dir)
                / 'gradient_calibration_events.jsonl', event)

    def _gradient_audit_due(self, step):
        cadence = int(self.train_contract[
            'gradient_calibration']['audit_every_steps'])
        map_stage = _stage(self.sampler_contract['map']['stages'], step)
        return step > 0 and int(map_stage['map_batch']) > 0 and step % cadence == 0

    def _run_gradient_safety_audit(self, step):
        specification = self.train_contract['gradient_calibration']
        parameters = tuple(
            parameter for _, parameter in self._shared_calibration_parameters())
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        prior_mode = self.backbone.training
        map_ratios, closure_ratios, packs = [], [], []
        try:
            for pack_index in range(int(specification['probe_packs'])):
                local_norm, weighted_map_norm, closure_norm = (
                    self._calibration_pack_norms(
                        step, pack_index, parameters,
                        raw_class_weights=False))
                map_ratios.append(
                    local_norm / max(weighted_map_norm, 1e-12))
                if closure_norm is not None:
                    closure_ratios.append(
                        local_norm / max(closure_norm, 1e-12))
                packs.append({
                    'pack': pack_index, 'local_norm': local_norm,
                    'weighted_map_norm': weighted_map_norm,
                    'closure_norm': closure_norm,
                })
        finally:
            self.backbone.train(prior_mode)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        factor = float(specification['weighted_ratio_safety_factor'])
        map_bound = factor * float(torch.tensor(map_ratios).median())
        map_before = float(self.map_calibration)
        self.map_calibration.fill_(min(map_before, map_bound))
        close_before = float(self.closure_calibration)
        close_bound = None
        if len(closure_ratios) == int(specification['probe_packs']):
            close_bound = factor * float(
                torch.tensor(closure_ratios).median())
            self.closure_calibration.fill_(min(close_before, close_bound))
        self.controller_revision.add_(1)
        event = {
            'completed_updates': step, 'packs': packs,
            'map_C_before': map_before,
            'map_C_bound': map_bound,
            'map_C_after': float(self.map_calibration),
            'closure_C_before': close_before,
            'closure_C_bound': close_bound,
            'closure_C_after': float(self.closure_calibration),
            'action': 'downward_only_then_linear_recheck',
            'controlled_after_action': True,
        }
        self._gradient_audit_events.append(event)
        if self.trainer.is_global_zero:
            write_controller_event(
                Path(self.config.checkpointing.save_dir)
                / 'gradient_safety_audit_events.jsonl', event)

    def _loss(self, clean_tokens, valid_tokens,
              current_accumulation_step=None, train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        del xT, given_t, not_sampling_t
        if not train_mode:
            batch = clean_tokens.shape[0]
            quick_nodes = self.eval_contract[
                'local_validation']['quick_probe_times']
            node_index = (
                torch.arange(batch, device=self.device)
                + self._validation_batch_index * batch) % len(quick_nodes)
            node_tensor = torch.tensor(
                quick_nodes, device=self.device, dtype=torch.float32)
            t = node_tensor[node_index]
            noise = torch.randn(
                (batch, clean_tokens.shape[1], self.vocab_size),
                device=self.device, dtype=torch.float32,
                generator=self._generator('validation', int(self.global_step)))
            state = self.corrupt_vocab_state(clean_tokens, t, noise=noise)
            logits, _, probability = self.local_posterior(
                state, t, None, dropout_enabled=False)
            token_loss = (
                0.5 * (probability.square().sum(dim=-1)
                       - 2.0 * probability.gather(
                           -1, clean_tokens.unsqueeze(-1)).squeeze(-1) + 1.0)
                if self.line == 'F' else F.cross_entropy(
                    logits.transpose(1, 2), clean_tokens, reduction='none'))
            mask = valid_tokens.float()
            nlls = (token_loss * mask).sum()
            return Loss(
                loss=nlls / mask.sum().clamp_min(1.0), nlls=nlls,
                prior_loss=0.0, num_tokens=mask.sum())

        step = int(self.global_step)
        accumulation = int(current_accumulation_step or 0)
        if accumulation == 0 and self._calibration_due(step):
            self._run_gradient_calibration(step)
        if accumulation == 0 and self._gradient_audit_due(step):
            self._run_gradient_safety_audit(step)
        plan = self.sample_step_plan(step, accumulation)
        local_loss, local_tokens = self._local_objective(
            clean_tokens, valid_tokens, plan, step)
        map_loss, selected, closure_loss, closure_selected = (
            self._map_objective(clean_tokens, plan, step))
        scale = int(self.trainer.accumulate_grad_batches) * selected / 96.0
        closure_scale = (
            int(self.trainer.accumulate_grad_batches)
            * closure_selected / 32.0)
        lambda_map = self._lambda_map(step)
        lambda_close = self._lambda_close(step)
        total = (
            local_loss + lambda_map * scale * map_loss
            + lambda_close * closure_scale * closure_loss)
        for name, value in (
                ('local_loss', local_loss.detach()),
                ('map_loss', map_loss.detach()),
                ('closure_loss', closure_loss.detach()),
                ('lambda_map', total.new_tensor(lambda_map)),
                ('lambda_close', total.new_tensor(lambda_close)),
                ('map_selected', total.new_tensor(float(selected))),
                ('closure_selected', total.new_tensor(float(closure_selected)))):
            self.log(
                f'task1_tvm_final/{self.line}/{name}', value,
                on_step=True, on_epoch=False, sync_dist=True)
        self.log('loss', total.detach(), prog_bar=True,
                 on_step=True, on_epoch=False, sync_dist=True)
        mask = valid_tokens.float()
        return Loss(
            loss=total, nlls=(local_tokens.detach() * mask).sum(),
            prior_loss=0.0, num_tokens=mask.sum())

    def validation_step(self, batch, batch_idx):
        self._validation_batch_index = int(batch_idx)
        return super().validation_step(batch, batch_idx)

    def configure_optimizers(self):
        finite_g = tuple(self.backbone.finite_time_conditioner.parameters())
        finite_b = tuple(
            self.backbone.finite_correction_head.parameters()
            if self.backbone.finite_correction_head is not None else ())
        excluded = {id(parameter) for parameter in finite_g + finite_b}
        legacy = tuple(
            parameter for parameter in self.backbone.parameters()
            if parameter.requires_grad and id(parameter) not in excluded)
        groups = [
            {'params': legacy, 'name': 'legacy'},
            {'params': finite_g, 'name': 'finite_G'},
        ]
        if finite_b:
            groups.append({'params': finite_b, 'name': 'finite_B'})
        optimizer = torch.optim.AdamW(
            groups, lr=float(self.config.optim.lr),
            betas=tuple(self.train_contract['optimizer']['betas']),
            eps=float(self.train_contract['optimizer']['eps']),
            weight_decay=float(self.train_contract['optimizer']['weight_decay']))
        return optimizer

    def _set_group_learning_rates(self, optimizer):
        step = int(self.global_step)
        base = _linear_knots(
            step + 1, self.train_contract['optimizer']['lr_schedule'])
        rho = _rho_map(self.sampler_contract, step)
        final = float(self.train_contract['objective']['rho_map_final'])
        gate = min(1.0, rho / final) if final else 0.0
        for group in optimizer.param_groups:
            name = group['name']
            multiplier = gate if name == 'finite_G' else (
                0.1 * gate if name == 'finite_B' else 1.0)
            group['lr'] = base * multiplier

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        squares = []
        finite = True
        for parameter in self.backbone.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            finite = finite and bool(torch.isfinite(gradient).all())
            squares.append(gradient.square().sum())
        if not finite:
            raise FloatingPointError(
                'Nonfinite training gradient; no nan_to_num recovery allowed.')
        norm = (
            torch.stack(squares).sum().sqrt() if squares
            else torch.zeros((), device=self.device))
        self.log(
            f'task1_tvm_final/{self.line}/gradient_norm', norm,
            on_step=True, on_epoch=False, sync_dist=True)

    @torch.no_grad()
    def _update_training_target_ema(self):
        update = int(self.training_target_ema_updates) + 1
        decay = min(self.target_decay, 1.0 - 1.0 / (update + 1.0))
        for target, online in zip(
                self.training_target_ema.parameters(),
                self.backbone.parameters()):
            target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)
        self.training_target_ema_updates.add_(1)
        self.training_target_ema.eval()

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs.get('optimizer')
        if optimizer is None and len(args) >= 3:
            optimizer = args[2]
        if optimizer is None:
            raise RuntimeError('Lightning optimizer_step did not expose optimizer.')
        self._set_group_learning_rates(optimizer)
        for group in optimizer.param_groups:
            self.log(
                f'task1_tvm_final/{self.line}/lr_{group["name"]}',
                torch.tensor(float(group['lr']), device=self.device),
                on_step=True, on_epoch=False, sync_dist=True)
        update = int(self.global_step) + 1
        if self.ema is not None:
            self.ema.decay = min(
                self.eval_decay, 1.0 - 1.0 / (update + 1.0))
        super().optimizer_step(*args, **kwargs)
        self._update_training_target_ema()
        completed = int(self.global_step) + 1
        self.completed_updates_counter.fill_(completed)
        self.cumulative_sequences.fill_(completed * 256)
        self.cumulative_tokens.fill_(completed * 256 * 128)
        now = time.monotonic()
        if self._last_optimizer_wall_time is not None:
            self.gpu_wall_seconds.add_(now - self._last_optimizer_wall_time)
        self._last_optimizer_wall_time = now
        self._finalize_canonical_transition(completed)
        for attempt in self.train_contract['canonical_sc']['gate_attempts']:
            if (completed == int(attempt['step'])
                    and int(attempt['candidate_K']) > int(self.canonical_k)):
                self._run_canonical_gate(
                    completed, int(attempt['candidate_K']))

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['task1_tvm_final'] = {
            'contract_version': CONTRACT_VERSION,
            'line': self.line,
            'contract_dir': str(self.contract_dir),
            'controller_revision': int(self.controller_revision),
            'last_plan_audit': self._last_plan_audit,
            'canonical_gate_events': self._canonical_gate_events,
            'calibration_events': self._calibration_events,
            'gradient_audit_events': self._gradient_audit_events,
            'random_streams': [
                'plan', 'local_noise', 'map_noise', 'validation'],
        }

    def on_load_checkpoint(self, checkpoint):
        record = checkpoint.get('task1_tvm_final')
        if record is not None and (
                record.get('contract_version') != CONTRACT_VERSION
                or record.get('line') != self.line):
            raise RuntimeError('Checkpoint contract identity does not match.')
        if record is not None:
            self._canonical_gate_events = list(
                record.get('canonical_gate_events', []))
            self._calibration_events = list(
                record.get('calibration_events', []))
            self._gradient_audit_events = list(
                record.get('gradient_audit_events', []))
        super().on_load_checkpoint(checkpoint)

    def _eval_mode(self):
        role = str(getattr(
            self.config.algo, 'task1_eval_weight_role', 'eval_ema'))
        if role == 'target_ema':
            if self._evaluation_online_backbone is not None:
                raise RuntimeError('Target-EMA evaluation swap is already active.')
            self._evaluation_online_backbone = [self.backbone]
            self._modules['backbone'] = self.training_target_ema
            self.backbone.eval()
            self.noise.eval()
            return
        if role in {'online', 'canonical_target'}:
            self.backbone.eval()
            self.noise.eval()
            return
        if role != 'eval_ema':
            raise ValueError(f'Unknown evaluation weight role {role!r}.')
        super()._eval_mode()

    def _train_mode(self):
        if self._evaluation_online_backbone is not None:
            self._modules['backbone'] = self._evaluation_online_backbone[0]
            self._evaluation_online_backbone = None
            self.backbone.train()
            self.noise.train()
            return
        role = str(getattr(
            self.config.algo, 'task1_eval_weight_role', 'eval_ema'))
        if role in {'online', 'canonical_target'}:
            self.backbone.train()
            self.noise.train()
            return
        super()._train_mode()

    def _generation_noise(self, num_samples):
        base = int(getattr(
            self.config.sampling, 'task1_initial_noise_seed',
            self.eval_contract['seed']))
        rows = []
        for offset in range(num_samples):
            sample_id = self._generation_sample_offset + offset
            generator = torch.Generator(device=self.device)
            generator.manual_seed(base + sample_id)
            rows.append(torch.randn(
                (1, self.num_tokens, self.vocab_size),
                device=self.device, dtype=torch.float32,
                generator=generator))
        self._generation_sample_offset += num_samples
        return torch.cat(rows, dim=0)

    def _finite_generation_query(self, state, r, eta):
        tau = self._t_to_tau(r.float().clamp(0.0, 1.0))
        condition = self._task1_condition(tau, r).to(self.device)
        features = torch.stack((r.float(), eta.float()), dim=-1)
        residual, output_features = self.backbone(
            self.backbone.vocab_embed(state),
            self._process_sigma(condition), inputs_are_embeddings=True,
            x_self_cond=None, finite_time_features=features,
            return_output_features=True)
        bias = flm_vocab_gaussian_bias(
            state, r, self._current_token_bias_weight(), self.flm_time_eps)
        probability = F.softmax(residual.float() + bias.float(), dim=-1)
        if self.line == 'F':
            correction = self.backbone.finite_correction_head(
                output_features).float()
            correction = correction - correction.mean(dim=-1, keepdim=True)
            return probability + eta[:, None, None] * correction
        return probability

    @torch.no_grad()
    def _generate_finite(self, num_samples, grid):
        state = self._generation_noise(num_samples)
        for left, right in zip(grid, grid[1:]):
            r = torch.full((num_samples,), float(left), device=self.device)
            s = torch.full((num_samples,), float(right), device=self.device)
            eta = (s - r) / (1.0 - r)
            endpoint = self._finite_generation_query(state, r, eta)
            state = (
                (1.0 - eta[:, None, None]) * state
                + eta[:, None, None] * endpoint)
        self.last_sampling_nfe = len(grid) - 1
        return state.argmax(dim=-1)

    @torch.no_grad()
    def _generate_local(self, num_samples, terminal, queries, rolling):
        state = self._generation_noise(num_samples)
        nodes = torch.linspace(
            0.0, float(terminal), int(queries) + 1, device=self.device)
        cache = None
        for left, right in zip(nodes[:-1], nodes[1:]):
            t = left.expand(num_samples)
            _, _, probability = self.local_posterior(
                state, t, cache if rolling else None,
                dropout_enabled=False)
            alpha = ((right - left) / (1.0 - left)).clamp(0.0, 1.0)
            state = (1.0 - alpha) * state + alpha * probability
            if rolling:
                cache = detached_self_conditioning_embedding(
                    probability, self.backbone.vocab_embed).detach()
        self.last_sampling_nfe = int(queries)
        return state.argmax(dim=-1)

    @torch.no_grad()
    def _generate_canonical(self, num_samples, forward_budget):
        state = self._generation_noise(num_samples)
        effective_k = max(int(self.canonical_k), int(self.canonical_previous_k))
        macro_steps = int(forward_budget) // effective_k
        if macro_steps <= 0:
            raise ValueError('Canonical forward budget is smaller than K.')
        nodes = torch.linspace(0.0, 0.95, macro_steps + 1, device=self.device)
        actual_queries = 0
        for left, right in zip(nodes[:-1], nodes[1:]):
            t = left.expand(num_samples)
            target = self.canonical_target(state, t)
            alpha = ((right - left) / (1.0 - left)).clamp(0.0, 1.0)
            state = (1.0 - alpha) * state + alpha * target.prob
            actual_queries += target.actual_queries
        self.last_sampling_nfe = actual_queries
        return state.argmax(dim=-1)

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        del eps
        mode = str(getattr(
            self.config.algo, 'task1_eval_mode', 'finite'))
        if mode == 'finite':
            grid = [float(value) for value in getattr(
                self.config.algo, 'task1_eval_physical_grid')]
            return self._generate_finite(num_samples, grid)
        if num_steps is None:
            raise ValueError('Local/canonical evaluation requires num_steps.')
        if mode == 'legacy_rolling_T1':
            return self._generate_local(
                num_samples, 1.0, int(num_steps), rolling=True)
        if mode == 'matched_rolling_T095':
            return self._generate_local(
                num_samples, 0.95, int(num_steps), rolling=True)
        if mode == 'matched_cold_T095':
            return self._generate_local(
                num_samples, 0.95, int(num_steps), rolling=False)
        if mode == 'canonical_fixed_budget_T095':
            return self._generate_canonical(
                num_samples, int(num_steps))
        raise ValueError(f'Unknown Task1 evaluation mode {mode!r}.')


def write_controller_event(path, event):
    """Append one explicit controller decision; used by the runtime adapter."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, sort_keys=True) + '\n')
