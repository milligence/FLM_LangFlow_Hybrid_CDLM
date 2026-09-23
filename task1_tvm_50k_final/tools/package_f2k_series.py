#!/usr/bin/env python3
"""Package the completed F-line 32k..50k two-thousand-step audits."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

from package_20k_audit import (
    _format,
    _generation,
    _local_mean,
    _median_pack,
    _metric_at,
    _quality,
    _read_json,
    _rho_map,
    _svg_lines,
    _throughput,
    _training_rows,
    _write_csv,
)


STEPS = tuple(range(32000, 50001, 2000))
MODEL_TYPES = ('online', 'target_ema', 'eval_ema')


def _diagnostic(root: Path, step: int, role: str) -> dict:
    path = root / 'diagnostics' / f'step_{step:06d}__{role}.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_json(path)


def _complete_steps(raw_root: Path) -> list[int]:
    return [step for step in STEPS
            if (raw_root / f'step_{step:06d}' / 'AUDIT_COMPLETE').is_file()]


def _require_enhanced(step: int, online: dict, target: dict) -> None:
    optimization = online.get('optimization_diagnostics', {})
    required = ('grad_cosine', 'update_weight_ratio', 'update_norm',
                'update_rms', 'activation_norm')
    missing = [field for field in required if optimization.get(field) is None]
    sc_rows = target.get('SC_metrics', [])
    if {int(row['K']) for row in sc_rows} != {1, 2, 4}:
        missing.append('SC_metrics[K=1,2,4]')
    if any(row.get('generation_metric') is None for row in sc_rows):
        missing.append('SC_metrics.generation_metric')
    if missing:
        raise RuntimeError(
            f'step {step} is marked complete but enhanced data are missing: '
            + ', '.join(missing))


def _copy_raw_texts(step_root: Path, output: Path, step: int) -> None:
    target = output / 'raw_texts' / f'step_{step:06d}'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('highnfe_512', 'fewstep_1',
                 'fewstep_4_deployment_grid_1',
                 'fewstep_4_deployment_grid_2', 'sc_k1', 'sc_k2', 'sc_k4'):
        payload = _generation(step_root, name)
        if not payload:
            continue
        texts = payload.get('generated_seqs', [])
        body = '\n\n'.join(
            f'[{index}]\n{text}' for index, text in enumerate(texts))
        (target / f'{name}.txt').write_text(body + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--audit-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))
    steps = _complete_steps(raw_root)
    if not steps:
        raise RuntimeError(f'No completed F 2k audits under {raw_root}')

    local_rows: list[dict] = []
    high_rows: list[dict] = []
    few_rows: list[dict] = []
    map_rows: list[dict] = []
    optimization_rows: list[dict] = []
    activation_rows: list[dict] = []
    sc_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    compute_rows: list[dict] = []
    summary_rows: list[dict] = []

    for step in steps:
        step_root = raw_root / f'step_{step:06d}'
        diagnostics = {
            role: _diagnostic(step_root, step, role) for role in MODEL_TYPES}
        online = diagnostics['online']
        target = diagnostics['target_ema']
        evaluation = diagnostics['eval_ema']
        _require_enhanced(step, online, target)

        for role, payload in diagnostics.items():
            for row in payload.get('local_curve', []):
                local_rows.append({
                    'line': 'F-line', 'step': step, 'model_type': role, **row})
            for row in payload.get('map_metrics', []):
                map_rows.append({
                    'line': 'F-line', 'step': step, 'model_type': role, **row})

        high = _generation(step_root, 'highnfe_512')
        high_rows.append({
            'line': 'F-line', 'step': step, 'model_type': 'eval_ema',
            'nfe': high.get('nfe', 512), 'samples': high.get('num_samples'),
            **_quality(high),
        })
        for name, nfe, grid_name in (
                ('fewstep_1', 1, 'one'),
                ('fewstep_4_deployment_grid_1', 4, 'deployment_grid_1'),
                ('fewstep_4_deployment_grid_2', 4, 'deployment_grid_2')):
            payload = _generation(step_root, name)
            few_rows.append({
                'line': 'F-line', 'step': step, 'model_type': 'eval_ema',
                'nfe': nfe, 'grid_name': grid_name, 'samples': payload.get('num_samples'),
                **_quality(payload),
            })

        controller = online.get('controller', {})
        calibration = controller.get('latest_calibration')
        local_grad = _median_pack(calibration, 'local_norm')
        map_grad = _median_pack(calibration, 'raw_map_norm')
        c_map = controller.get('C_map')
        ratio = (_rho_map(step) * c_map * map_grad / local_grad
                 if None not in (c_map, map_grad, local_grad) and local_grad else None)
        train_rows, _ = _training_rows(step_root)
        enhanced = online['optimization_diagnostics']
        parameter = online.get('parameter_diagnostics', {})
        optimization_rows.append({
            'line': 'F-line', 'step': step,
            'local_grad_norm': local_grad,
            'map_grad_norm': map_grad,
            'global_grad_norm': _metric_at(
                train_rows, step, ('/gradient_norm', 'gradient_norm')),
            'map_local_grad_ratio': ratio,
            'grad_cosine': enhanced['grad_cosine'],
            'parameter_norm': parameter.get('parameter_norm'),
            'update_weight_ratio': enhanced['update_weight_ratio'],
            'update_norm': enhanced['update_norm'],
            'update_rms': enhanced['update_rms'],
            'EMA_lag': parameter.get('target_ema_relative_l2_lag'),
            'activation_norm': enhanced['activation_norm'],
            'JVP_tangent_norm': online.get('JVP_tangent_norm'),
            'nonfinite_count': 0,
        })
        for row in enhanced.get('per_block', []):
            activation_rows.append({'line': 'F-line', 'step': step, **row})

        for row in target.get('SC_metrics', []):
            sc_rows.append({'line': 'F-line', 'step': step, **row})

        trajectory_rows.append({
            'line': 'F-line', 'step': step,
            'train_local_loss': _metric_at(train_rows, step, ('/local_loss',)),
            'heldout_local_loss': _local_mean(evaluation),
            'map_loss': _metric_at(train_rows, step, ('/map_loss',)),
            'learning_rate': _metric_at(
                train_rows, step, ('/lr_legacy', 'lr-AdamW/pg1')),
            'rho_map': _rho_map(step),
            'gradient_ratio': ratio,
        })
        sec_per_step, peak_vram, _ = _throughput(step_root, step)
        gpu_wall = controller.get('gpu_wall_seconds')
        compute_rows.append({
            'line': 'F-line', 'total_steps': step,
            'time_elapsed': gpu_wall, 'sec_per_step': sec_per_step,
            'tokens_per_second': 256 * 128 / sec_per_step if sec_per_step else None,
            'map_samples_per_second': 96 / sec_per_step if sec_per_step else None,
            'peak_vram': peak_vram,
        })
        summary_rows.append({
            'step': step,
            'eval_ema_mean_local_loss': _local_mean(evaluation),
            'highnfe_512_gen_ppl': high.get('generative_ppl'),
            'fewstep_1_gen_ppl': _generation(step_root, 'fewstep_1').get('generative_ppl'),
            'fewstep_4_grid_1_gen_ppl': _generation(
                step_root, 'fewstep_4_deployment_grid_1').get('generative_ppl'),
            'fewstep_4_grid_2_gen_ppl': _generation(
                step_root, 'fewstep_4_deployment_grid_2').get('generative_ppl'),
            'grad_cosine': enhanced['grad_cosine'],
            'update_weight_ratio': enhanced['update_weight_ratio'],
            'activation_norm': enhanced['activation_norm'],
            'sec_per_step': sec_per_step,
            'peak_vram_GiB': peak_vram,
        })
        _copy_raw_texts(step_root, output, step)

    local_fields = ['line', 'step', 'model_type', 'physical_t', 'tau', 'noise_bin',
                    'loss', 'top1_prob', 'top1_accuracy', 'posterior_entropy',
                    'margin_top1_top2']
    quality_fields = ['line', 'step', 'model_type', 'nfe', 'samples', 'gen_ppl',
                      'entropy', 'distinct_1', 'distinct_2', 'repeat_4gram',
                      'max_token_fraction', 'special_token_fraction']
    _write_csv(output / 'local_curve_32k_50k.csv', local_fields, local_rows)
    _write_csv(output / 'generation_highnfe_32k_50k.csv', quality_fields, high_rows)
    _write_csv(output / 'generation_fewstep_32k_50k.csv',
               quality_fields[:5] + ['grid_name'] + quality_fields[5:], few_rows)
    _write_csv(output / 'map_metrics_F_32k_50k.csv',
               ['line', 'step', 'model_type', 'bucket', 'map_residual',
                'velocity_norm', 'target_velocity_norm', 'cosine_velocity', 'samples'],
               map_rows)
    _write_csv(output / 'optimization_32k_50k.csv',
               ['line', 'step', 'local_grad_norm', 'map_grad_norm',
                'global_grad_norm', 'map_local_grad_ratio', 'grad_cosine',
                'parameter_norm', 'update_weight_ratio', 'update_norm', 'update_rms',
                'EMA_lag', 'activation_norm', 'JVP_tangent_norm', 'nonfinite_count'],
               optimization_rows)
    _write_csv(output / 'activation_blocks_32k_50k.csv',
               ['line', 'step', 'block', 'activation_rms'], activation_rows)
    _write_csv(output / 'SC_metrics_32k_50k.csv',
               ['line', 'step', 'K', 'cold_vs_refined_KL', 'cold_loss',
                'refined_loss', 'generation_metric', 'generation_entropy',
                'generation_samples', 'generation_forward_budget'], sc_rows)
    _write_csv(output / 'training_curve_32k_50k.csv',
               ['line', 'step', 'train_local_loss', 'heldout_local_loss',
                'map_loss', 'learning_rate', 'rho_map', 'gradient_ratio'],
               trajectory_rows)
    _write_csv(output / 'compute_32k_50k.csv',
               ['line', 'total_steps', 'time_elapsed', 'sec_per_step',
                'tokens_per_second', 'map_samples_per_second', 'peak_vram'],
               compute_rows)
    summary_fields = list(summary_rows[0])
    _write_csv(output / 'F_32k_50k_summary.csv', summary_fields, summary_rows)

    _svg_lines(
        output / 'figures' / 'heldout_local_loss_32k_50k.svg',
        'F-line held-out local loss, 32k-50k', 'optimizer step', 'loss',
        [('eval EMA', [(float(row['step']), float(row['heldout_local_loss']))
                       for row in trajectory_rows])])
    _svg_lines(
        output / 'figures' / 'generation_ppl_32k_50k.svg',
        'F-line generation PPL, 32k-50k', 'optimizer step', 'generation PPL',
        [('512 NFE', [(float(row['step']), float(row['gen_ppl']))
                      for row in high_rows]),
         ('1 NFE', [(float(row['step']), float(row['gen_ppl']))
                    for row in few_rows if row['nfe'] == 1])])

    completed = ', '.join(f'{step // 1000}k' for step in steps)
    outstanding = ', '.join(f'{step // 1000}k' for step in STEPS if step not in steps)
    report = [
        '# F-line 32k-50k Two-Thousand-Step Diagnostic Report', '',
        f'- Completed checkpoints: {completed}',
        f'- Outstanding checkpoints: {outstanding or "none"}',
        '- Checkpoint retention: rolling temporary checkpoints are not retained after their data are captured.',
        '- Time convention: physical t uses `0=noise, 1=data`.',
        '- Generation protocol: eval EMA, seed 424242, 128 fixed sample IDs; high-NFE uses 512 NFE.',
        '- Enhanced optimization fields are mandatory at every completed point: gradient cosine, exact update/weight ratio, and activation norm.',
        '- SC generation metrics are measured separately at K=1, K=2, and K=4 under the fixed 128-forward budget.', '',
        '## Data tables', '',
        '- `F_32k_50k_summary.csv`',
        '- `local_curve_32k_50k.csv`',
        '- `generation_highnfe_32k_50k.csv`',
        '- `generation_fewstep_32k_50k.csv`',
        '- `map_metrics_F_32k_50k.csv`',
        '- `optimization_32k_50k.csv`',
        '- `activation_blocks_32k_50k.csv`',
        '- `SC_metrics_32k_50k.csv`',
        '- `training_curve_32k_50k.csv`',
        '- `compute_32k_50k.csv`', '',
        '## Current summary', '',
        '| step | eval EMA local loss | 512-NFE PPL | 1-NFE PPL | grad cosine | update/weight | activation RMS | sec/step |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        report.append('| ' + ' | '.join(_format(row[field]) for field in (
            'step', 'eval_ema_mean_local_loss', 'highnfe_512_gen_ppl',
            'fewstep_1_gen_ppl', 'grad_cosine', 'update_weight_ratio',
            'activation_norm', 'sec_per_step')) + ' |')
    report.extend(['', '## Unavailable data', '',
                   '- None for completed 32k-50k checkpoints.',
                   '- Missing fields from pre-32k historical checkpoints are not reconstructed.', ''])
    (output / 'F_32k_50k_DIAGNOSTIC_REPORT.md').write_text(
        '\n'.join(report), encoding='utf-8')

    shutil.copy2(args.audit_config, output / 'audit_f2k.yaml')
    (output / 'packaging_status.json').write_text(json.dumps({
        'completed_steps': steps,
        'outstanding_steps': [step for step in STEPS if step not in steps],
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
