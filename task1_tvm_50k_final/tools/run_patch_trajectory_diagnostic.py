#!/usr/bin/env python3
"""Bounded F-line rollout/reference diagnostic used by the 30k/32k gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from run_20k_audit import _compose_model, _load_tokens


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


@torch.no_grad()
def canonical_probability(model, state, t, k: int, backbone) -> torch.Tensor:
    cache = None
    probability = None
    for _ in range(k):
        _, _, probability = model.local_posterior(
            state, t, cache, model=backbone, dropout_enabled=False)
        cache = backbone.vocab_embed(probability).detach()
    return probability


def integration_nodes(grid: list[float], subdivisions: int) -> list[float]:
    values = {round(0.95 * index / subdivisions, 15)
              for index in range(subdivisions + 1)}
    values.update(round(value, 15) for value in grid)
    return sorted(values)


@torch.no_grad()
def integrate_reference(model, initial, grid, k, subdivisions):
    requested = {round(value, 15) for value in grid}
    state = initial.detach().float()
    saved = {0.0: state.detach().cpu()}
    nodes = integration_nodes(grid, subdivisions)
    forwards = 0
    for left, right in zip(nodes, nodes[1:]):
        t = torch.full((state.shape[0],), left, device=model.device)
        probability = canonical_probability(model, state, t, k, model.backbone)
        alpha = (right - left) / (1.0 - left)
        state = (1.0 - alpha) * state + alpha * probability
        forwards += k
        if round(right, 15) in requested:
            saved[round(right, 15)] = state.detach().cpu()
    return saved, forwards, len(nodes) - 1


@torch.no_grad()
def integrate_segment(model, initial, left, right, k, maximum_dt=0.95 / 512):
    steps = int(math.ceil((right - left) / maximum_dt))
    nodes = torch.linspace(left, right, steps + 1, device=model.device)
    state = initial.detach().float()
    forwards = 0
    for begin, end in zip(nodes[:-1], nodes[1:]):
        t = begin.expand(state.shape[0])
        probability = canonical_probability(model, state, t, k, model.backbone)
        alpha = (end - begin) / (1.0 - begin)
        state = (1.0 - alpha) * state + alpha * probability
        forwards += k
    return state, forwards, steps


def per_sequence(value: torch.Tensor) -> list[float]:
    return value.mean(dim=1).detach().double().cpu().tolist()


@torch.no_grad()
def source_metrics(model, state, r_value, s_value, teacher_k,
                   reference_at_s, reference_posterior):
    count = state.shape[0]
    r = torch.full((count,), r_value, device=model.device)
    s = torch.full((count,), s_value, device=model.device)
    eta = (s - r) / (1.0 - r)
    output = model.finite_with_eta_jvp(state, r, eta)
    value, value_tangent = model._f_objective_terms(
        output['logits'], output['dlogits'], output['b_raw'],
        output['db_raw'], eta)
    correction = output['b_raw'].float()
    correction = correction - correction.mean(dim=-1, keepdim=True)
    eta_view = eta[:, None, None]
    q_value = value + eta_view * (1.0 - eta_view) * value_tangent
    endpoint = (1.0 - eta_view) * state + eta_view * value
    # Trajectory comparisons use one matched eval-EMA weight family throughout.
    teacher = canonical_probability(
        model, endpoint.detach(), s, teacher_k, model.backbone)
    residual = q_value - teacher
    terminal = residual / (1.0 - s[:, None, None])
    predicted_velocity = (
        -state + value + eta_view * value_tangent
    ) / (1.0 - r[:, None, None])
    target_velocity = (
        teacher - endpoint.detach()) / (1.0 - s[:, None, None])
    endpoint_error = endpoint - reference_at_s
    map_posterior = canonical_probability(
        model, endpoint, s, teacher_k, model.backbone)
    reference_log = reference_posterior.clamp_min(1e-30).log()
    map_log = map_posterior.clamp_min(1e-30).log()
    reference_top2 = reference_at_s.topk(2, dim=-1).values
    margin = reference_top2[..., 0] - reference_top2[..., 1]
    delta_inf = endpoint_error.abs().amax(dim=-1)
    result = {
        'signal_residual_E2': per_sequence(residual.square().sum(dim=-1)),
        'raw_terminal_residual_E2': per_sequence(terminal.square().sum(dim=-1)),
        'signal_Q_vs_pT_cosine': per_sequence(F.cosine_similarity(
            q_value, teacher, dim=-1, eps=1e-12)),
        'raw_velocity_cosine': per_sequence(F.cosine_similarity(
            predicted_velocity, target_velocity, dim=-1, eps=1e-12)),
        'd_eta_A_l2': per_sequence(value_tangent.norm(dim=-1)),
        'eta_1_minus_eta_d_eta_A_l2': per_sequence(
            (eta_view * (1.0 - eta_view) * value_tangent).norm(dim=-1)),
        'eta_B_l2': per_sequence((eta_view * correction).norm(dim=-1)),
        'eta_squared_B_l2': per_sequence(
            (eta_view.square() * correction).norm(dim=-1)),
        'signed_A_negative_mass': per_sequence((-value).clamp_min(0).sum(dim=-1)),
        'signed_Q_negative_mass': per_sequence((-q_value).clamp_min(0).sum(dim=-1)),
        'A_mass_abs_error': per_sequence((value.sum(dim=-1) - 1.0).abs()),
        'Q_mass_abs_error': per_sequence((q_value.sum(dim=-1) - 1.0).abs()),
        'endpoint_state_error': per_sequence(endpoint_error.square().sum(dim=-1)),
        'endpoint_state_error_over_signal_power': per_sequence(
            endpoint_error.square().sum(dim=-1) / max(s_value * s_value, 1e-12)),
        'endpoint_state_error_over_noise_power': per_sequence(
            endpoint_error.square().sum(dim=-1)
            / max((1.0 - s_value) ** 2, 1e-12)),
        'endpoint_posterior_KL_reference_to_map': per_sequence(
            (reference_posterior * (reference_log - map_log)).sum(dim=-1)),
        'terminal_argmax_disagreement': per_sequence(
            (endpoint.argmax(dim=-1) != reference_at_s.argmax(dim=-1)).float()),
        'terminal_margin_risk_fraction': per_sequence(
            (margin <= 2.0 * delta_inf).float()),
    }
    return endpoint.detach(), result


def bootstrap_ratio(rollout: list[float], reference: list[float], seed: int):
    left = np.asarray(rollout, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    ratio = float(left.mean() / max(right.mean(), 1e-30))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(left), size=(1000, len(left)))
    values = left[indices].mean(axis=1) / np.maximum(
        right[indices].mean(axis=1), 1e-30)
    return ratio, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--grid-name', required=True)
    parser.add_argument('--grid', type=json.loads, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=424242)
    parser.add_argument('--seed-count', type=int, default=16)
    parser.add_argument('--microbatch', type=int, default=2)
    args = parser.parse_args()
    if len(args.grid) != 5 or args.grid[0] != 0.0 or args.grid[-1] != 0.95:
        raise ValueError('The rollout gate requires one four-step grid from 0 to 0.95.')

    bindings = json.loads(args.bindings.read_text(encoding='utf-8'))
    sidecar = Path(str(args.checkpoint) + '.sha256')
    if not sidecar.exists():
        raise FileNotFoundError(f'Missing checkpoint SHA sidecar: {sidecar}')
    checkpoint_sha256 = sidecar.read_text(encoding='utf-8').split()[0]
    protocol_hash = hashlib.sha256(
        b'task1-f-rollout-reference-diagnostic-v2').hexdigest()
    compose_args = SimpleNamespace(
        line='f', model_type='eval_ema', output=args.output_dir / 'compose.json',
        contract_dir=args.contract_dir, checkpoint=args.checkpoint, step=args.step)
    model, repo = _compose_model(compose_args, bindings, {})
    model.config.algo.task1_compile_finite_jvp = True
    model.config.algo.task1_compile_calibration_kernels = True
    teacher_k = max(int(model.canonical_k), int(model.canonical_previous_k))
    validation = Path(bindings['data'].get('owt_validation')
                      or bindings['data']['owt_train'])
    clean_bank = _load_tokens(repo, validation, args.seed_count)

    records = {interval: {source: {} for source in (
        'data', 'canonical_reference_path', 'own_finite_rollout')}
        for interval in range(4)}
    prefix_original: list[float] = []
    prefix_reset: list[float] = []
    solver_floor: list[float] = []
    decomposition = {name: [] for name in (
        'a_squared', 'b_squared', 'cross_term', 'sum_squared',
        'identity_error', 'isolated_reference_last_state_mse')}
    reference_forward_count = None
    reference_steps = None

    for start in range(0, args.seed_count, args.microbatch):
        stop = min(start + args.microbatch, args.seed_count)
        noises = []
        for sample_id in range(start, stop):
            generator = torch.Generator(device=model.device)
            generator.manual_seed(args.seed + sample_id)
            noises.append(torch.randn(
                (1, model.num_tokens, model.vocab_size), device=model.device,
                dtype=torch.float32, generator=generator))
        initial = torch.cat(noises, dim=0)
        refs_cpu, forwards, ref_steps = integrate_reference(
            model, initial, args.grid, teacher_k, 512)
        reference_forward_count = forwards
        reference_steps = ref_steps
        refs = {node: value.to(model.device) for node, value in refs_cpu.items()}
        reference_posteriors = {}
        for node in args.grid[1:]:
            t = torch.full((stop - start,), node, device=model.device)
            reference_posteriors[node] = canonical_probability(
                model, refs[round(node, 15)], t, teacher_k, model.backbone)

        if args.step in {30000, 40000} and start < 4:
            floor_count = min(4 - start, stop - start)
            half_refs, _, _ = integrate_reference(
                model, initial[:floor_count], args.grid, teacher_k, 1024)
            delta = (half_refs[0.95] - refs_cpu[0.95][:floor_count]).float()
            solver_floor.extend(per_sequence(delta.square().sum(dim=-1)))

        rollout = initial.detach()
        rollout_at_t2 = None
        for interval, (r_value, s_value) in enumerate(
                zip(args.grid, args.grid[1:])):
            reference_source = refs[round(r_value, 15)]
            clean = clean_bank[start:stop].to(model.device)
            data_noises = []
            for sample_id in range(start, stop):
                generator = torch.Generator(device=model.device)
                generator.manual_seed(
                    args.seed + 100000 + interval * 1000 + sample_id)
                data_noises.append(torch.randn(
                    (1, model.num_tokens, model.vocab_size),
                    device=model.device, dtype=torch.float32,
                    generator=generator))
            data_state = model.corrupt_vocab_state(
                clean, torch.full((stop - start,), r_value, device=model.device),
                noise=torch.cat(data_noises, dim=0))
            sources = {
                'data': data_state,
                'canonical_reference_path': reference_source,
                'own_finite_rollout': rollout,
            }
            rollout_next = None
            for source, state in sources.items():
                endpoint, values = source_metrics(
                    model, state, r_value, s_value, teacher_k,
                    refs[round(s_value, 15)], reference_posteriors[s_value])
                if source == 'own_finite_rollout':
                    rollout_next = endpoint
                for metric, sequence_values in values.items():
                    records[interval][source].setdefault(metric, []).extend(
                        sequence_values)
            if interval == 3 and args.step == 32000 and start < 8:
                repair_count = min(8 - start, stop - start)
                repaired_reference, _, _ = integrate_segment(
                    model, data_state[:repair_count], r_value, s_value,
                    teacher_k)
                repaired_t = torch.full(
                    (repair_count,), s_value, device=model.device)
                repaired_posterior = canonical_probability(
                    model, repaired_reference, repaired_t, teacher_k,
                    model.backbone)
                _, repaired_values = source_metrics(
                    model, data_state[:repair_count], r_value, s_value,
                    teacher_k, repaired_reference, repaired_posterior)
                repaired = records[interval].setdefault(
                    'data_same_source_flow', {})
                for metric, sequence_values in repaired_values.items():
                    repaired.setdefault(metric, []).extend(sequence_values)
            if interval == 3 and args.step in {32000, 36000, 40000}:
                canonical_from_rollout, _, _ = integrate_segment(
                    model, rollout, r_value, s_value, teacher_k)
                a = canonical_from_rollout - refs[round(s_value, 15)]
                b = rollout_next - canonical_from_rollout
                total = rollout_next - refs[round(s_value, 15)]
                a2 = a.square().sum(dim=-1)
                b2 = b.square().sum(dim=-1)
                cross = 2.0 * (a * b).sum(dim=-1)
                total2 = total.square().sum(dim=-1)
                for name, tensor in (
                        ('a_squared', a2), ('b_squared', b2),
                        ('cross_term', cross), ('sum_squared', total2),
                        ('identity_error', (a2 + b2 + cross - total2).abs())):
                    decomposition[name].extend(per_sequence(tensor))
                reset_source = refs[round(r_value, 15)]
                count = reset_source.shape[0]
                r = torch.full((count,), r_value, device=model.device)
                s = torch.full((count,), s_value, device=model.device)
                eta = (s - r) / (1.0 - r)
                reset_value = model._finite_generation_query(reset_source, r, eta)
                reset_endpoint = (
                    (1.0 - eta[:, None, None]) * reset_source
                    + eta[:, None, None] * reset_value)
                reset_error = (
                    reset_endpoint - refs[round(s_value, 15)]).square().sum(dim=-1)
                decomposition['isolated_reference_last_state_mse'].extend(
                    per_sequence(reset_error))
            rollout = rollout_next
            if interval == 1:
                rollout_at_t2 = rollout.detach()

        terminal_reference = refs[0.95]
        original_delta = rollout - terminal_reference
        prefix_original.extend(per_sequence(
            original_delta.square().sum(dim=-1)))
        reset = refs[round(args.grid[2], 15)]
        for r_value, s_value in zip(args.grid[2:-1], args.grid[3:]):
            count = reset.shape[0]
            r = torch.full((count,), r_value, device=model.device)
            s = torch.full((count,), s_value, device=model.device)
            eta = (s - r) / (1.0 - r)
            value = model._finite_generation_query(reset, r, eta)
            reset = (1.0 - eta[:, None, None]) * reset + eta[:, None, None] * value
        reset_delta = reset - terminal_reference
        prefix_reset.extend(per_sequence(reset_delta.square().sum(dim=-1)))

        del refs, refs_cpu, reference_posteriors, initial, rollout, reset
        torch.cuda.empty_cache()

    rows = []
    intervals_passing = []
    for interval, (r_value, s_value) in enumerate(zip(args.grid, args.grid[1:])):
        for source, metrics in records[interval].items():
            if source == 'data':
                pairing = 'unmatched_cross_source'
                usable = False
                source_kind = 'analytical'
            elif source == 'data_same_source_flow':
                pairing = 'same_source_flow'
                usable = True
                source_kind = 'analytical'
            else:
                pairing = 'same_initial_noise'
                usable = True
                source_kind = (
                    'canonical_reference' if source == 'canonical_reference_path'
                    else 'own_rollout')
            row = {
                'schema_version': 'f30-final-v2',
                'record_type': (
                    'legacy_cross_source' if source == 'data' else 'map'),
                'checkpoint_step': args.step, 'grid': args.grid_name,
                'completed_step': args.step,
                'checkpoint_sha256': checkpoint_sha256,
                'protocol_hash': protocol_hash,
                'interval': interval, 'r': r_value, 's': s_value,
                'eta': (s_value - r_value) / (1.0 - r_value),
                'source': source,
                'seed_count': len(next(iter(metrics.values()))) if metrics else 0,
                'source_kind': source_kind, 'pairing_mode': pairing,
                'usable_for_gate': usable,
                'student_weight_family': 'eval_ema',
                'teacher_weight_family': 'eval_ema',
                'reference_weight_family': 'eval_ema',
                'F_Q_KL': None,
                'F_Q_KL_reason': 'signed_A_and_Q_are_not_categorical_distributions',
            }
            for metric, values in metrics.items():
                row[metric] = float(np.mean(values))
            rows.append(row)
        rollout_values = records[interval]['own_finite_rollout'][
            'signal_residual_E2']
        reference_values = records[interval]['canonical_reference_path'][
            'signal_residual_E2']
        ratio, low, high = bootstrap_ratio(
            rollout_values, reference_values,
            args.seed + args.step + interval)
        reference_mean = float(np.mean(reference_values))
        for row in rows[-3:]:
            row['rollout_over_reference_signal_E2_ratio'] = ratio
            row['ratio_bootstrap_95_low'] = low
            row['ratio_bootstrap_95_high'] = high
            row['reference_signal_E2_probe_floor'] = 1.0e-8
        if (interval > 0 and reference_mean >= 1.0e-8
                and ratio >= 2.0 and low > 1.0):
            intervals_passing.append(interval)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'interval_diagnostics.jsonl').open(
            'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    decomposition_record = {
        'schema_version': 'f30-final-v2',
        'record_type': 'decomposition',
        'completed_step': args.step,
        'checkpoint_sha256': checkpoint_sha256,
        'protocol_hash': protocol_hash,
        'source_kind': 'own_rollout',
        'pairing_mode': 'same_initial_noise',
        'usable_for_gate': args.step in {32000, 36000, 40000},
        'student_weight_family': 'eval_ema',
        'teacher_weight_family': 'eval_ema',
        'reference_weight_family': 'eval_ema',
        'seed_count': len(decomposition['a_squared']),
        'values': {
            name: float(np.mean(values)) if values else None
            for name, values in decomposition.items()
        },
        'per_sequence': decomposition,
    }
    atomic_json(args.output_dir / 'last_interval_decomposition.json',
                decomposition_record)
    original_mean = float(np.mean(prefix_original))
    reset_mean = float(np.mean(prefix_reset))
    reduction = 1.0 - reset_mean / max(original_mean, 1e-30)
    gate = {
        'checkpoint_step': args.step,
        'schema_version': 'f30-final-v2',
        'weight_family': 'eval_ema finite field and canonical K2 reference',
        'canonical_K': teacher_k,
        'grid_name': args.grid_name,
        'grid': args.grid,
        'noise_seed': args.seed,
        'seed_count': args.seed_count,
        'reference_maximum_step': 0.95 / 512,
        'reference_actual_integrator_steps': reference_steps,
        'reference_actual_forward_count': reference_forward_count,
        'solver_floor_half_step_seed_count': len(solver_floor),
        'solver_floor_terminal_signal_E2': (
            float(np.mean(solver_floor)) if solver_floor else None),
        'noninitial_intervals_passing_rollout_gap': intervals_passing,
        'minimum_intervals_required': 2,
        'prefix_reset_time_index': 2,
        'terminal_signal_error_original': original_mean,
        'terminal_signal_error_prefix_reset': reset_mean,
        'terminal_signal_error_reduction': reduction,
        'minimum_prefix_reset_reduction': 0.25,
        'rollout_gap_gate_passed_at_checkpoint': (
            len(intervals_passing) >= 2 and reduction >= 0.25),
        'training_action': 'evidence_only; sampler decisions use decision_policy.py',
        'source_semantics': {
            'data': 'analytical x_r; cross-source endpoint fields are tagged unusable',
            'data_same_source_flow': 'eight analytical last-interval states integrated canonically from the same x_r',
            'canonical_reference_path': '512-step canonical local Euler path from shared initial noise',
            'own_finite_rollout': 'four-step finite chain from shared initial noise',
        },
        'protocol_hash': protocol_hash,
        'checkpoint_sha256': checkpoint_sha256,
    }
    atomic_json(args.output_dir / 'gate_diagnostic.json', gate)


if __name__ == '__main__':
    main()
