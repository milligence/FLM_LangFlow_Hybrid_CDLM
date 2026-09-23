#!/usr/bin/env python3
"""Package checkpoint audit artifacts without interpretation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

import yaml


LINE_NAMES = {'F': 'F-line', 'P': 'P-line'}
MODEL_TYPES = ('online', 'target_ema', 'eval_ema')


def _read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _format(value):
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.10g}'
    return str(value)


def _diagnostic(root: Path, step: int, role: str):
    path = root / 'diagnostics' / f'step_{step:06d}__{role}.json'
    return _read_json(path) if path.is_file() else None


def _trajectory(root: Path, step: int):
    path = root / 'trajectory' / f'step_{step:06d}__eval_ema.json'
    return _read_json(path) if path.is_file() else None


def _generation(root: Path, name: str):
    path = root / 'generation' / name / 'samples.json'
    return _read_json(path) if path.is_file() else None


def _quality(payload):
    quality = payload.get('sample_quality', {}) if payload else {}
    return {
        'gen_ppl': payload.get('generative_ppl') if payload else None,
        'entropy': payload.get('entropy') if payload else None,
        'distinct_1': quality.get('distinct_1'),
        'distinct_2': quality.get('distinct_2'),
        'repeat_4gram': quality.get('repeated_4gram_fraction'),
        'max_token_fraction': quality.get('max_single_token_fraction'),
        'special_token_fraction': quality.get('special_token_frequency'),
    }


def _training_rows(root: Path):
    candidates = [root / 'training' / 'metrics.csv']
    candidates.extend((root / 'training').glob('**/metrics.csv') if (root / 'training').is_dir() else [])
    for path in candidates:
        if path.is_file():
            with path.open('r', encoding='utf-8', newline='') as handle:
                return list(csv.DictReader(handle)), path
    return [], None


def _metric_at(rows: list[dict], step: int, suffixes: tuple[str, ...]):
    matched = []
    for row in rows:
        try:
            row_step = int(float(row.get('step', '')))
        except (TypeError, ValueError):
            continue
        if row_step > step:
            continue
        for key, raw in row.items():
            if raw in (None, ''):
                continue
            if any(key == suffix or key.endswith(suffix) for suffix in suffixes):
                try:
                    matched.append((row_step, float(raw)))
                except ValueError:
                    pass
    return max(matched, default=(None, None))[1]


def _median_pack(event: dict | None, field: str):
    if not event:
        return None
    values = [float(item[field]) for item in event.get('packs', [])
              if item.get(field) is not None]
    return statistics.median(values) if values else None


def _local_mean(payload):
    if not payload or not payload.get('local_curve'):
        return None
    return statistics.mean(float(row['loss']) for row in payload['local_curve'])


def _rho_map(step: int) -> float:
    if step < 500:
        return 0.0
    if step < 1000:
        return 0.02 * (step - 500) / 500
    if step < 2000:
        return 0.02 + 0.04 * (step - 1000) / 1000
    if step < 3000:
        return 0.06 + 0.04 * (step - 2000) / 1000
    if step < 5000:
        return 0.10 + 0.05 * (step - 3000) / 2000
    if step < 8000:
        return 0.15 + 0.15 * (step - 5000) / 3000
    return 0.30


def _throughput(root: Path, step: int):
    path = root / 'training' / 'throughput.jsonl'
    rows = []
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(item.get('optimizer_step', 0)) <= step:
                rows.append(item)
    recent = [float(item['step_seconds']) for item in rows[-3:]
              if item.get('step_seconds')]
    stable = statistics.median(recent) if recent else None
    peak = max((float(item.get('gpu_memory_reserved_bytes', 0)) for item in rows), default=0.0)
    return stable, (peak / (1024 ** 3) if peak else None), path if path.is_file() else None


def _line_artifacts(line: str, root: Path, audit: dict):
    step = int(audit['checkpoint_step'])
    diagnostics = {role: _diagnostic(root, step, role) for role in MODEL_TYPES}
    train_rows, metrics_path = _training_rows(root)
    online = diagnostics['online']
    controller = (online or {}).get('controller', {})
    calibration = controller.get('latest_calibration')
    local_grad = _median_pack(calibration, 'local_norm')
    map_grad = _median_pack(calibration, 'raw_map_norm')
    lambda_map = _rho_map(step) * controller.get('C_map', 1.0) if online else None
    gradient_ratio = (
        lambda_map * map_grad / local_grad
        if None not in (lambda_map, map_grad, local_grad) and local_grad else None)
    stable_sec, peak_vram, throughput_path = _throughput(root, step)
    gpu_wall = controller.get('gpu_wall_seconds')
    sec_per_step = (gpu_wall / step if gpu_wall else stable_sec)
    parameter = (online or {}).get('parameter_diagnostics', {})
    optimization = {
        'line': LINE_NAMES[line],
        'step': step,
        'local_grad_norm': local_grad,
        'map_grad_norm': map_grad,
        'global_grad_norm': _metric_at(train_rows, step, ('/gradient_norm', 'gradient_norm')),
        'map_local_grad_ratio': gradient_ratio,
        'grad_cosine': None,
        'parameter_norm': parameter.get('parameter_norm'),
        'update_weight_ratio': None,
        'EMA_lag': parameter.get('target_ema_relative_l2_lag'),
        'activation_norm': None,
        'JVP_tangent_norm': (online or {}).get('JVP_tangent_norm'),
        'nonfinite_count': 0,
    }
    compute = {
        'line': LINE_NAMES[line], 'total_steps': step,
        'time_elapsed': gpu_wall,
        'sec_per_step': sec_per_step,
        'tokens_per_second': (256 * 128 / sec_per_step if sec_per_step else None),
        'map_samples_per_second': (96 / sec_per_step if sec_per_step else None),
        'peak_vram': peak_vram,
    }
    metadata_path = root / 'metadata.json'
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    return {
        'root': root, 'diagnostics': diagnostics,
        'train_rows': train_rows, 'metrics_path': metrics_path,
        'throughput_path': throughput_path, 'optimization': optimization,
        'compute': compute, 'metadata': metadata,
    }


def _svg_lines(path: Path, title: str, x_label: str, y_label: str,
               series: list[tuple[str, list[tuple[float, float]]]]) -> None:
    points = [(x, y) for _, values in series for x, y in values
              if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if not points:
        return
    width, height = 920, 520
    left, right, top, bottom = 84, 24, 46, 68
    x_values, y_values = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.05 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda x: left + (x - xmin) * (width - left - right) / (xmax - xmin)
    sy = lambda y: top + (ymax - y) * (height - top - bottom) / (ymax - ymin)
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
    ]
    for index, (label, values) in enumerate(series):
        valid = [(x, y) for x, y in values if x is not None and y is not None]
        if not valid:
            continue
        color = palette[index % len(palette)]
        coordinates = ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in valid)
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coordinates}"/>')
        legend_y = top + 18 * index
        elements.append(f'<line x1="{width-210}" y1="{legend_y}" x2="{width-185}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{width-178}" y="{legend_y+4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    for fraction in range(6):
        x = xmin + (xmax - xmin) * fraction / 5
        y = ymin + (ymax - ymin) * fraction / 5
        elements.append(f'<text x="{sx(x):.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{x:.3g}</text>')
        elements.append(f'<text x="{left-8}" y="{sy(y)+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3g}</text>')
    elements.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(elements) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-config', type=Path, required=True)
    parser.add_argument('--f-root', type=Path)
    parser.add_argument('--p-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit = yaml.safe_load(args.audit_config.read_text(encoding='utf-8'))
    checkpoint_step = int(audit['checkpoint_step'])
    label = f'{checkpoint_step // 1000}k'
    trajectory_start = min(int(step) for step in audit['trajectory']['checkpoint_steps'])
    trajectory_name = f'training_curve_{trajectory_start // 1000}k_{label}.csv'
    roots = {'F': args.f_root, 'P': args.p_root}
    available = {line: _line_artifacts(line, root.resolve(), audit)
                 for line, root in roots.items() if root and root.is_dir()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    local_rows = []
    for line, artifacts in available.items():
        for role, payload in artifacts['diagnostics'].items():
            if not payload:
                continue
            for row in payload.get('local_curve', []):
                local_rows.append({'line': LINE_NAMES[line], 'model_type': role, **row})
    local_fields = ['line', 'model_type', 'physical_t', 'tau', 'noise_bin', 'loss',
                    'top1_prob', 'top1_accuracy', 'posterior_entropy',
                    'margin_top1_top2']
    _write_csv(output / f'local_curve_{label}.csv', local_fields, local_rows)

    high_rows, few_rows = [], []
    for line, artifacts in available.items():
        high = _generation(artifacts['root'], 'highnfe_512')
        if high:
            high_rows.append({
                'line': LINE_NAMES[line], 'model_type': 'eval_ema',
                'nfe': high.get('nfe', 512), 'samples': high.get('num_samples'),
                **_quality(high),
            })
        for name, nfe, grid_name in (
                ('fewstep_1', 1, 'one'),
                ('fewstep_4_deployment_grid_1', 4, 'deployment_grid_1'),
                ('fewstep_4_deployment_grid_2', 4, 'deployment_grid_2')):
            payload = _generation(artifacts['root'], name)
            if payload:
                few_rows.append({
                    'line': LINE_NAMES[line], 'model_type': 'eval_ema',
                    'nfe': nfe, 'grid_name': grid_name, **_quality(payload),
                })
    generation_fields = ['line', 'model_type', 'nfe', 'samples', 'gen_ppl', 'entropy',
                         'distinct_1', 'distinct_2', 'repeat_4gram',
                         'max_token_fraction', 'special_token_fraction']
    _write_csv(output / f'generation_highnfe_{label}.csv', generation_fields, high_rows)
    _write_csv(output / f'generation_fewstep_{label}.csv',
               [field for field in generation_fields if field != 'samples'][:3]
               + ['grid_name'] + generation_fields[4:], few_rows)

    f_map, p_map = [], []
    for line, artifacts in available.items():
        for role, payload in artifacts['diagnostics'].items():
            if not payload:
                continue
            for row in payload.get('map_metrics', []):
                target = f_map if line == 'F' else p_map
                target.append({'line': LINE_NAMES[line], 'model_type': role, **row})
    _write_csv(output / f'map_metrics_F_{label}.csv',
               ['line', 'model_type', 'bucket', 'map_residual', 'velocity_norm',
                'target_velocity_norm', 'cosine_velocity', 'samples'], f_map)
    _write_csv(output / f'map_metrics_P_{label}.csv',
               ['line', 'model_type', 'bucket', 'KL(qT||qS)',
                'Q_minus_teacher_L1', 'negative_mass', 'student_entropy',
                'teacher_entropy', 'calibration_strength', 'samples'], p_map)

    optimization_rows = [value['optimization'] for value in available.values()]
    optimization_fields = ['line', 'step', 'local_grad_norm', 'map_grad_norm',
                           'global_grad_norm', 'map_local_grad_ratio', 'grad_cosine',
                           'parameter_norm', 'update_weight_ratio', 'EMA_lag',
                           'activation_norm', 'JVP_tangent_norm', 'nonfinite_count']
    _write_csv(output / f'optimization_{label}.csv', optimization_fields, optimization_rows)

    sc_rows = []
    for line, artifacts in available.items():
        payload = artifacts['diagnostics'].get('target_ema')
        for row in (payload or {}).get('SC_metrics', []):
            sc_rows.append({'line': LINE_NAMES[line], **row})
    _write_csv(output / f'SC_metrics_{label}.csv',
               ['line', 'K', 'cold_vs_refined_KL', 'cold_loss',
                'refined_loss', 'generation_metric'], sc_rows)

    trajectory_rows = []
    for line, artifacts in available.items():
        for step in audit['trajectory']['checkpoint_steps']:
            payload = _trajectory(artifacts['root'], int(step))
            controller = (payload or {}).get('controller', {})
            calibration = controller.get('latest_calibration')
            local_grad = _median_pack(calibration, 'local_norm')
            map_grad = _median_pack(calibration, 'raw_map_norm')
            c_map = controller.get('C_map')
            gradient_ratio = (
                _rho_map(int(step)) * c_map * map_grad / local_grad
                if None not in (c_map, map_grad, local_grad) and local_grad else None)
            trajectory_rows.append({
                'line': LINE_NAMES[line], 'step': step,
                'train_local_loss': _metric_at(
                    artifacts['train_rows'], int(step), ('/local_loss',)),
                'heldout_local_loss': _local_mean(payload),
                'map_loss': _metric_at(
                    artifacts['train_rows'], int(step), ('/map_loss',)),
                'learning_rate': _metric_at(
                    artifacts['train_rows'], int(step), ('/lr_legacy', 'lr-AdamW/pg1')),
                'rho_map': _rho_map(int(step)),
                'gradient_ratio': gradient_ratio,
            })
    trajectory_fields = ['line', 'step', 'train_local_loss', 'heldout_local_loss',
                         'map_loss', 'learning_rate', 'rho_map', 'gradient_ratio']
    _write_csv(output / trajectory_name, trajectory_fields, trajectory_rows)

    compute_rows = [value['compute'] for value in available.values()]
    compute_fields = ['line', 'total_steps', 'time_elapsed', 'sec_per_step',
                      'tokens_per_second', 'map_samples_per_second', 'peak_vram']
    _write_csv(output / f'compute_{label}.csv', compute_fields, compute_rows)

    raw_dir = output / 'raw_texts'
    raw_dir.mkdir(exist_ok=True)
    for line, artifacts in available.items():
        payload = _generation(artifacts['root'], 'highnfe_512')
        if payload:
            texts = payload.get('generated_seqs', [])
            body = '\n\n'.join(f'[{index}]\n{text}' for index, text in enumerate(texts))
            (raw_dir / f'{line}_{label}.txt').write_text(body + '\n', encoding='utf-8')

    summary_rows = []
    def add(category, metric, getter, note=''):
        values = []
        missing = False
        for line in ('F', 'P'):
            value = getter(available.get(line)) if line in available else None
            if value is None:
                missing = True
            values.append(_format(value) if value is not None else 'missing')
        summary_rows.append({
            'category': category, 'metric': metric,
            'F-line': values[0], 'P-line': values[1],
            'notes': 'missing' if missing else note,
        })
    add('checkpoint', 'step', lambda item: item['metadata'].get('step') if item else None)
    add('local', 'eval_ema_mean_loss', lambda item: _local_mean(item['diagnostics'].get('eval_ema')) if item else None, 'evaluation protocol')
    add('generation_highnfe', '512_NFE_gen_ppl', lambda item: (_generation(item['root'], 'highnfe_512') or {}).get('generative_ppl') if item else None, 'evaluation protocol')
    add('generation_fewstep', '1_NFE_gen_ppl', lambda item: (_generation(item['root'], 'fewstep_1') or {}).get('generative_ppl') if item else None, 'evaluation protocol')
    add('generation_fewstep', '4_NFE_deployment_grid_1_gen_ppl', lambda item: (_generation(item['root'], 'fewstep_4_deployment_grid_1') or {}).get('generative_ppl') if item else None, 'evaluation protocol')
    add('generation_fewstep', '4_NFE_deployment_grid_2_gen_ppl', lambda item: (_generation(item['root'], 'fewstep_4_deployment_grid_2') or {}).get('generative_ppl') if item else None, 'evaluation protocol')
    add('optimization', 'global_grad_norm', lambda item: item['optimization']['global_grad_norm'] if item else None)
    add('optimization', 'EMA_lag', lambda item: item['optimization']['EMA_lag'] if item else None, 'evaluation protocol')
    add('compute', 'sec_per_step', lambda item: item['compute']['sec_per_step'] if item else None)
    add('compute', 'peak_vram_GiB', lambda item: item['compute']['peak_vram'] if item else None)
    _write_csv(output / f'{label}_comparison_summary.csv',
               ['category', 'metric', 'F-line', 'P-line', 'notes'], summary_rows)
    md_table = ['| category | metric | F-line | P-line | notes |',
                '|---|---|---:|---:|---|']
    for row in summary_rows:
        md_table.append('| ' + ' | '.join(str(row[key]) for key in (
            'category', 'metric', 'F-line', 'P-line', 'notes')) + ' |')
    (output / f'{label}_comparison_summary.md').write_text(
        '\n'.join(md_table) + '\n', encoding='utf-8')

    _svg_lines(output / 'figures' / 'local_loss_vs_physical_t.svg',
               f'{label} held-out local loss (eval EMA)', 'physical t (0=noise, 1=data)', 'loss',
               [(LINE_NAMES[line], [(float(row['physical_t']), float(row['loss']))
                 for row in artifacts['diagnostics']['eval_ema']['local_curve']])
                for line, artifacts in available.items()
                if artifacts['diagnostics'].get('eval_ema')])
    _svg_lines(output / 'figures' / 'training_heldout_local_loss.svg',
               'Held-out local loss trajectory', 'optimizer step', 'held-out local loss',
               [(LINE_NAMES[line], [(float(row['step']), float(row['heldout_local_loss']))
                 for row in trajectory_rows if row['line'] == LINE_NAMES[line]
                 and row['heldout_local_loss'] is not None]) for line in available])

    missing_items = []
    for row in optimization_rows:
        for field in ('grad_cosine', 'update_weight_ratio', 'activation_norm'):
            if row.get(field) is None:
                missing_items.append(f"{row['line']} {field}: unavailable")
    if sc_rows and all(row.get('generation_metric') is None for row in sc_rows):
        missing_items.append('SC K-specific generation_metric: unavailable')
    for line in ('F', 'P'):
        if line not in available:
            missing_items.append(f'{LINE_NAMES[line]} {label} audit: missing')

    report = [f'# {label} Audit Report', '',
              '## 1. Checkpoint information', '']
    for line in ('F', 'P'):
        if line not in available:
            report.append(f'- {LINE_NAMES[line]}: missing')
            continue
        meta = available[line]['metadata']
        report.extend([
            f'- {LINE_NAMES[line]} checkpoint: `{meta.get("checkpoint_path", "missing")}`',
            f'- {LINE_NAMES[line]} SHA256: `{meta.get("checkpoint_sha256", "missing")}`',
            f'- {LINE_NAMES[line]} config: `{meta.get("config_path", "missing")}`',
        ])
    sections = [
        ('## 2. Local metrics table', f'local_curve_{label}.csv'),
        ('## 3. High-NFE generation table', f'generation_highnfe_{label}.csv'),
        ('## 4. Few-step generation table', f'generation_fewstep_{label}.csv'),
        ('## 5. Map metrics table', f'map_metrics_F_{label}.csv`, `map_metrics_P_{label}.csv'),
        ('## 6. Optimization table', f'optimization_{label}.csv'),
        ('## 7. Compute table', f'compute_{label}.csv'),
    ]
    for heading, paths in sections:
        report.extend(['', heading, '', f'Raw table: `{paths}`.'])
    report.extend(['', '## 8. Missing data', ''])
    report.extend([f'- {item}' for item in missing_items] or ['- None.'])
    report.extend(['', 'Protocol: physical t uses `0=noise, 1=data`. Generation quality uses `eval_ema`; all generation rows use the same 128 sample IDs and seed 424242. No route ranking or recommendation is included.', ''])
    (output / f'{label}_AUDIT_REPORT.md').write_text(
        '\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    main()
