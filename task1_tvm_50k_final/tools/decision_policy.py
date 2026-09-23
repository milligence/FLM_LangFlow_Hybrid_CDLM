"""Pure fail-closed profile policy for the final F-line continuation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = 'f30-final-v2'
PROFILES = {'pre32', 'uniform', 'late_trial', 'late_fixed', 'off'}


def _number(mapping, key):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _within(mapping, key, low, high):
    value = _number(mapping, key)
    return value is not None and low <= value <= high


def profile_plan(profile):
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    patch = profile not in {'pre32', 'off'}
    intervals = ([0, 0, 2, 6] if profile in {'late_trial', 'late_fixed'}
                 else [2, 2, 2, 2] if patch else [0, 0, 0, 0])
    return {
        'counts': {'S': 12, 'M': 24, 'L': 16 if patch else 24,
                   'D_original': 24, 'D_patch': 8 if patch else 0,
                   'Z': 6, 'H': 6},
        'D_patch_grid': 'B', 'D_patch_source': 'analytical',
        'D_patch_interval_rows': intervals,
        'D_patch_exact_rows': [count // 2 for count in intervals],
        'D_patch_jitter_rows': [count // 2 for count in intervals],
        'map_batch': 96, 'loss_denominator': 96, 'rho_map': 0.30,
    }


def _risk_guard(report, expected_steps):
    rows = report.get('risk_checks')
    if not isinstance(rows, list) or len(rows) != 2:
        return False
    if [row.get('completed_step') for row in rows
            if isinstance(row, dict)] != list(expected_steps):
        return False
    for row in rows:
        for grid in ('A', 'B'):
            delta = _number(row, f'eval_{grid}_delta_nll')
            low = _number(row, f'eval_{grid}_delta_nll_ci_low')
            if delta is None or low is None or delta <= math.log(1.15) or low <= 0:
                return False
        online = _number(row, 'online_B_delta_nll')
        canonical = _number(row, 'canonical_ppl_ratio')
        if (online is None or online < 0 or canonical is None
                or canonical <= 0 or canonical > 1.10):
            return False
    return True


def _gate36(report):
    metrics = report.get('metrics', {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        'ready': report.get('all_required_metrics_ready') is True,
        'protocol': report.get('protocol_compatible') is True,
        'distribution_health': report.get('generation_health') is True,
        'online_not_already_improving': _within(
            metrics, 'online_B_delta_nll_32_36', math.log(.95) + 1e-12, math.inf),
        'eval_not_already_improving': _within(
            metrics, 'eval_B_delta_nll_32_36', math.log(.95) + 1e-12, math.inf),
        'late_share_34': _within(metrics, 'late_signal_share_34', .70, 1.0),
        'late_share_36': _within(metrics, 'late_signal_share_36', .70, 1.0),
        'last_map_not_already_improving': _within(
            metrics, 'terminal_ref_state_mse_ratio_36_32', .85 + 1e-12, math.inf),
        'above_numeric_floor': _within(
            metrics, 'terminal_mse_over_matching_floor_36', 100., math.inf),
        'nontrivial_finite_canonical_gap': _within(
            metrics, 'eval_B_minus_canonical_nll_36', math.log(1.10), math.inf),
        'gap_supported': _within(
            metrics, 'eval_B_minus_canonical_nll_ci_low_36', 1e-15, math.inf),
        'canonical_not_degrading': _within(
            metrics, 'canonical_ppl_ratio_36_32', 1e-15, 1.10),
        'local_aggregate_protected': _within(
            metrics, 'local_aggregate_mse_ratio_36_32', 0., 1.01),
        'sensitive_local_mse_protected': _within(
            metrics, 'sensitive_local_mse_max_ratio_36_32', 0., 1.03),
        'sensitive_local_ce_protected': _within(
            metrics, 'sensitive_local_ce_max_ratio_36_32', 0., 1.03),
        'gradient_budget_in_range': _within(
            metrics, 'weighted_map_ratio_36', .15, .45),
    }


def _gate40(report):
    metrics = report.get('metrics', {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        'ready': report.get('all_required_metrics_ready') is True,
        'protocol': report.get('protocol_compatible') is True,
        'distribution_health': report.get('generation_health') is True,
        'online_improvement': _within(metrics, 'online_B_ppl_ratio_40_36', 1e-15, .95),
        'online_improvement_supported': _within(
            metrics, 'online_B_delta_nll_ci_high_40_36', -math.inf, -1e-15),
        'late_transport_improvement': _within(
            metrics, 'terminal_ref_state_mse_ratio_40_36', 0., .90),
        'eval_B_protected': _within(metrics, 'eval_B_ppl_ratio_40_36', 1e-15, 1.05),
        'eval_A_protected': _within(metrics, 'eval_A_ppl_ratio_40_36', 1e-15, 1.10),
        'canonical_protected': _within(metrics, 'canonical_ppl_ratio_40_36', 1e-15, 1.10),
        'general_L_protected': _within(metrics, 'general_L_signal_mse_ratio_40_36', 0., 1.20),
        'local_aggregate_protected': _within(metrics, 'local_aggregate_mse_ratio_40_36', 0., 1.01),
        'sensitive_local_mse_protected': _within(metrics, 'sensitive_local_mse_max_ratio_40_36', 0., 1.03),
        'sensitive_local_ce_protected': _within(metrics, 'sensitive_local_ce_max_ratio_40_36', 0., 1.03),
        'gradient_budget_in_range': _within(metrics, 'weighted_map_ratio_40', .15, .45),
    }


def decide(report):
    if report.get('schema_version') != SCHEMA:
        raise ValueError('Incorrect metrics schema; do not merge protocols.')
    step = report.get('completed_step')
    if isinstance(step, bool) or not isinstance(step, int) or not 30000 <= step <= 50000:
        raise ValueError('completed_step must be an integer in [30000, 50000].')
    current = report.get('current_profile')
    if current not in PROFILES:
        raise ValueError('Unknown current_profile.')
    checks = {}
    status, target = 'hold', current
    reason = 'No scientific change scheduled at this checkpoint.'
    if report.get('lineage_verified') is not True:
        status, reason = 'blocked', 'Approved exact-30k lineage is not verified.'
    elif step < 32000:
        if current != 'pre32':
            raise ValueError('Unexpected pre-32k profile.')
    elif step == 32000 and current == 'pre32':
        status, target = 'apply', 'uniform'
        reason = 'Activate approved analytical B D32 at update 32001.'
    elif step == 36000 and current == 'uniform':
        if report.get('protocol_compatible') is True and _risk_guard(report, (34000, 36000)):
            status, target = 'apply', 'off'
            reason = 'Two-checkpoint severe risk guard; reverse sampling only.'
        else:
            checks = _gate36(report)
            if all(checks.values()):
                status, target = 'apply', 'late_trial'
                reason = 'Single bounded [0,0,2,6] trial through 40k.'
            else:
                reason = 'Gate36 incomplete or failed; keep uniform D32.'
    elif step == 40000 and current == 'late_trial':
        if report.get('protocol_compatible') is True and _risk_guard(report, (38000, 40000)):
            status, target = 'apply', 'off'
            reason = 'Severe trial risk guard; reverse sampling only.'
        else:
            checks = _gate40(report)
            status = 'apply'
            if all(checks.values()):
                target, reason = 'late_fixed', 'Retain [0,0,2,6] through 50k.'
            else:
                target, reason = 'uniform', 'Trial ended; restore [2,2,2,2].'
    elif step > 40000 and current == 'late_trial':
        status, target = 'apply', 'uniform'
        reason = 'Missed 40k adjudication; trial cannot be extended.'
    elif step > 32000 and current == 'pre32':
        status, reason = 'blocked', 'Missing 32k activation event.'
    changed_plan = profile_plan(target) != profile_plan(current)
    return {
        'schema_version': SCHEMA, 'completed_step': step,
        'action_first_update': step + 1 if status == 'apply' and step < 50000 else None,
        'status': status, 'old_profile': current, 'new_profile': target,
        'reason': reason, 'checks': checks, 'plan': profile_plan(target),
        'recalibrate_existing_lambda': status == 'apply' and changed_plan,
        'reset_model_optimizer_ema_rng': False,
        'training_launched_or_mutated_by_this_helper': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = decide(json.loads(args.report.read_text(encoding='utf-8')))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
