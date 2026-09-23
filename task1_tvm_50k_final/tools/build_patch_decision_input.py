#!/usr/bin/env python3
"""Build matched v2 decision evidence from completed checkpoint reviews."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from paired_statistics import paired_nll_difference


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _review(root, step):
    if step == 30000 and (root / 'review_30k').exists():
        return root / 'review_30k'
    return root / f'step_{step:06d}'


def _sample(root, step, name):
    path = _review(root, step) / 'generation' / name / 'samples.json'
    return _load(path) if path.exists() else None


def _diagnostic(root, step, role):
    path = (_review(root, step) / 'diagnostics'
            / f'step_{step:06d}__{role}.json')
    return _load(path) if path.exists() else None


def _trajectory(root, step):
    path = _review(root, step) / 'trajectory_B' / 'interval_diagnostics.jsonl'
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _gate(root, step):
    path = _review(root, step) / 'trajectory_B' / 'gate_diagnostic.json'
    return _load(path) if path.exists() else None


def _scores(payload, count=64):
    if payload is None:
        return None
    rows = payload.get('per_sample_scores')
    if not isinstance(rows, list) or len(rows) < count:
        return None
    rows = rows[:count]
    return ([float(row['nll']) for row in rows],
            [float(row['scored_tokens']) for row in rows])


def _paired(new, old, count=64):
    left, right = _scores(new, count), _scores(old, count)
    if left is None or right is None:
        return None
    return paired_nll_difference(
        left[0], left[1], right[0], right[1], repeats=2000)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _health(new, old):
    if new is None or old is None:
        return False
    new_q, old_q = new.get('sample_quality', {}), old.get('sample_quality', {})
    keys = ('mean_sample_unigram_entropy_nats', 'distinct_1', 'distinct_2',
            'repeated_4gram_fraction', 'max_single_token_fraction',
            'special_token_frequency')
    if not all(_finite(new_q.get(key)) and _finite(old_q.get(key)) for key in keys):
        return False
    return (
        old_q[keys[0]] - new_q[keys[0]] <= .10
        and new_q[keys[1]] / max(old_q[keys[1]], 1e-30) >= .90
        and new_q[keys[2]] / max(old_q[keys[2]], 1e-30) >= .90
        and new_q[keys[3]] - old_q[keys[3]] <= .005
        and new_q[keys[4]] - old_q[keys[4]] <= .01
        and new_q[keys[5]] - old_q[keys[5]] <= .005)


def _local_ratios(root, new_step, old_step):
    aggregate, sensitive_mse, sensitive_ce = [], [], []
    sensitive = {'[0.5,0.75)', '[0.75,0.9)'}
    for role in ('online', 'target_ema', 'eval_ema'):
        new, old = _diagnostic(root, new_step, role), _diagnostic(root, old_step, role)
        if new is None or old is None:
            continue
        new_rows, old_rows = new.get('local_curve', []), old.get('local_curve', [])
        old_by_bin = {row.get('noise_bin'): row for row in old_rows}
        if new_rows and old_rows:
            aggregate.append(
                sum(float(row['loss']) for row in new_rows) / len(new_rows)
                / max(sum(float(row['loss']) for row in old_rows) / len(old_rows), 1e-30))
        for row in new_rows:
            label = row.get('noise_bin')
            previous = old_by_bin.get(label)
            if label not in sensitive or previous is None:
                continue
            sensitive_mse.append(float(row['loss']) / max(float(previous['loss']), 1e-30))
            sensitive_ce.append(float(row['CE_nats']) / max(float(previous['CE_nats']), 1e-30))
    return {
        'aggregate': max(aggregate) if aggregate else None,
        'sensitive_mse': max(sensitive_mse) if sensitive_mse else None,
        'sensitive_ce': max(sensitive_ce) if sensitive_ce else None,
    }


def _weighted_ratio(root, step):
    data = _diagnostic(root, step, 'online')
    if data is None:
        return None
    controller = data.get('controller', {})
    local = controller.get('local_grad_norm_median')
    raw_map = controller.get('map_grad_norm_raw_median')
    calibration = controller.get('C_map')
    if not all(_finite(value) for value in (local, raw_map, calibration)):
        return None
    return .30 * float(calibration) * float(raw_map) / max(float(local), 1e-30)


def _late_share(root, step):
    rows = _trajectory(root, step)
    if rows is None:
        return None
    values = {int(row['interval']): float(row['signal_residual_E2'])
              for row in rows
              if row.get('source') == 'canonical_reference_path'
              and row.get('usable_for_gate') is True}
    if set(values) != {0, 1, 2, 3}:
        return None
    return values[3] / max(sum(values.values()), 1e-30)


def _last_reference_mse(root, step):
    rows = _trajectory(root, step) or []
    for row in rows:
        if (row.get('source') == 'canonical_reference_path'
                and int(row.get('interval', -1)) == 3
                and row.get('usable_for_gate') is True):
            return float(row['endpoint_state_error'])
    return None


def _general_l_signal(root, step):
    data = _diagnostic(root, step, 'eval_ema')
    if data is None:
        return None
    for row in data.get('map_metrics', []):
        if row.get('bucket') == 'L':
            return float(row['signal_residual_E_squared_vocab_sum'])
    return None


def _protocol_ok(payloads):
    existing = [value for value in payloads if value is not None]
    if len(existing) != len(payloads):
        return False
    return (len({value.get('protocol_id') for value in existing}) == 1
            and len({value.get('generation_seed') for value in existing}) == 1
            and all(value.get('comparable') is True for value in existing))


def _risk_row(root, baseline, step):
    result = {'completed_step': step}
    for grid in ('A', 'B'):
        comparison = _paired(
            _sample(root, step, f'finite_4_{grid}'),
            _sample(root, baseline, f'finite_4_{grid}'))
        if comparison:
            result[f'eval_{grid}_delta_nll'] = comparison['delta_nll']
            result[f'eval_{grid}_delta_nll_ci_low'] = comparison['ci_low']
    online = _paired(
        _sample(root, step, 'online_finite_4_B_64'),
        _sample(root, baseline, 'online_finite_4_B_64'))
    if online:
        result['online_B_delta_nll'] = online['delta_nll']
    canonical = _paired(
        _sample(root, step, 'canonical_K2_T095_64'),
        _sample(root, baseline, 'canonical_K2_T095_64'))
    if canonical:
        result['canonical_ppl_ratio'] = canonical['ppl_ratio']
    return result


def preliminary_risk(root, baseline, step):
    row = _risk_row(root, baseline, step)
    required = []
    for grid in ('A', 'B'):
        required.extend((row.get(f'eval_{grid}_delta_nll'),
                         row.get(f'eval_{grid}_delta_nll_ci_low')))
    required.append(row.get('online_B_delta_nll'))
    return (all(_finite(value) for value in required)
            and all(row[f'eval_{grid}_delta_nll'] > math.log(1.15)
                    and row[f'eval_{grid}_delta_nll_ci_low'] > 0
                    for grid in ('A', 'B'))
            and row['online_B_delta_nll'] >= 0)


def build(root, step, profile, lineage_verified):
    report = {
        'schema_version': 'f30-final-v2', 'completed_step': step,
        'current_profile': profile, 'lineage_verified': lineage_verified,
        'protocol_compatible': True, 'all_required_metrics_ready': True,
        'generation_health': True, 'risk_checks': [], 'metrics': {},
    }
    if step == 32000:
        return report
    if step == 36000:
        old_step = 32000
        online = _paired(_sample(root, step, 'online_finite_4_B_64'),
                         _sample(root, old_step, 'online_finite_4_B_64'))
        eval_b = _paired(_sample(root, step, 'eval_finite_4_B_64'),
                         _sample(root, old_step, 'eval_finite_4_B_64'))
        gap = _paired(_sample(root, step, 'eval_finite_4_B_64'),
                      _sample(root, step, 'canonical_K2_T095_64'))
        canonical = _paired(_sample(root, step, 'canonical_K2_T095_64'),
                            _sample(root, old_step, 'canonical_K2_T095_64'))
        last, old_last = _last_reference_mse(root, step), _last_reference_mse(root, old_step)
        floor_gate = _gate(root, 30000)
        floor = floor_gate.get('solver_floor_terminal_signal_E2') if floor_gate else None
        local = _local_ratios(root, step, old_step)
        metrics = {
            'online_B_delta_nll_32_36': online['delta_nll'] if online else None,
            'eval_B_delta_nll_32_36': eval_b['delta_nll'] if eval_b else None,
            'late_signal_share_34': _late_share(root, 34000),
            'late_signal_share_36': _late_share(root, 36000),
            'terminal_ref_state_mse_ratio_36_32': (
                last / old_last if _finite(last) and _finite(old_last) and old_last > 0 else None),
            'terminal_mse_over_matching_floor_36': (
                last / floor if _finite(last) and _finite(floor) and floor > 0 else None),
            'eval_B_minus_canonical_nll_36': gap['delta_nll'] if gap else None,
            'eval_B_minus_canonical_nll_ci_low_36': gap['ci_low'] if gap else None,
            'canonical_ppl_ratio_36_32': canonical['ppl_ratio'] if canonical else None,
            'local_aggregate_mse_ratio_36_32': local['aggregate'],
            'sensitive_local_mse_max_ratio_36_32': local['sensitive_mse'],
            'sensitive_local_ce_max_ratio_36_32': local['sensitive_ce'],
            'weighted_map_ratio_36': _weighted_ratio(root, step),
        }
        report['metrics'] = metrics
        report['risk_checks'] = [_risk_row(root, 32000, value)
                                 for value in (34000, 36000)]
        payloads = [_sample(root, value, name) for value in (32000, 36000)
                    for name in ('finite_4_B', 'online_finite_4_B_64',
                                 'eval_finite_4_B_64', 'canonical_K2_T095_64')]
        report['protocol_compatible'] = _protocol_ok(payloads)
        report['generation_health'] = _health(
            _sample(root, 36000, 'finite_4_B'),
            _sample(root, 32000, 'finite_4_B'))
        report['all_required_metrics_ready'] = all(_finite(value) for value in metrics.values())
    elif step == 40000 and profile == 'late_trial':
        old_step = 36000
        online = _paired(_sample(root, step, 'online_finite_4_B_64'),
                         _sample(root, old_step, 'online_finite_4_B_64'))
        eval_b = _paired(_sample(root, step, 'eval_finite_4_B_64'),
                         _sample(root, old_step, 'eval_finite_4_B_64'))
        eval_a = _paired(_sample(root, step, 'finite_4_A'),
                         _sample(root, old_step, 'finite_4_A'))
        canonical = _paired(_sample(root, step, 'canonical_K2_T095_64'),
                            _sample(root, old_step, 'canonical_K2_T095_64'))
        last, old_last = _last_reference_mse(root, step), _last_reference_mse(root, old_step)
        new_l, old_l = _general_l_signal(root, step), _general_l_signal(root, old_step)
        local = _local_ratios(root, step, old_step)
        metrics = {
            'online_B_ppl_ratio_40_36': online['ppl_ratio'] if online else None,
            'online_B_delta_nll_ci_high_40_36': online['ci_high'] if online else None,
            'terminal_ref_state_mse_ratio_40_36': (
                last / old_last if _finite(last) and _finite(old_last) and old_last > 0 else None),
            'eval_B_ppl_ratio_40_36': eval_b['ppl_ratio'] if eval_b else None,
            'eval_A_ppl_ratio_40_36': eval_a['ppl_ratio'] if eval_a else None,
            'canonical_ppl_ratio_40_36': canonical['ppl_ratio'] if canonical else None,
            'general_L_signal_mse_ratio_40_36': (
                new_l / old_l if _finite(new_l) and _finite(old_l) and old_l > 0 else None),
            'local_aggregate_mse_ratio_40_36': local['aggregate'],
            'sensitive_local_mse_max_ratio_40_36': local['sensitive_mse'],
            'sensitive_local_ce_max_ratio_40_36': local['sensitive_ce'],
            'weighted_map_ratio_40': _weighted_ratio(root, step),
        }
        report['metrics'] = metrics
        report['risk_checks'] = [_risk_row(root, 36000, value)
                                 for value in (38000, 40000)]
        payloads = [_sample(root, value, name) for value in (36000, 40000)
                    for name in ('finite_4_A', 'finite_4_B',
                                 'online_finite_4_B_64', 'eval_finite_4_B_64',
                                 'canonical_K2_T095_64')]
        report['protocol_compatible'] = _protocol_ok(payloads)
        report['generation_health'] = _health(
            _sample(root, 40000, 'finite_4_B'),
            _sample(root, 36000, 'finite_4_B'))
        report['all_required_metrics_ready'] = all(_finite(value) for value in metrics.values())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--current-profile', required=True)
    parser.add_argument('--lineage-verified', action='store_true')
    parser.add_argument('--risk-canonical-needed', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.risk_canonical_needed:
        baseline = 32000 if args.step == 34000 else 36000
        print('true' if preliminary_risk(
            args.output_root, baseline, args.step) else 'false')
        return
    result = build(args.output_root, args.step, args.current_profile,
                   args.lineage_verified)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
