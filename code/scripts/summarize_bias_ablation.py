#!/usr/bin/env python3
"""Build the registered four-arm bias-ablation evidence bundle."""

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = (
    'frozen_mse_warmup_bias',
    'frozen_mse_no_bias',
    'frozen_mse_full_bias',
    'frozen_mse_independent_prototype',
)
LABELS = {
    'frozen_mse_warmup_bias': 'A warmup/shared-C',
    'frozen_mse_no_bias': 'B no-bias',
    'frozen_mse_full_bias': 'C full/shared-C',
    'frozen_mse_independent_prototype': 'D warmup/independent-B',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric(payload, name):
    return payload.get('metrics', {}).get(f'val/hybrid_{name}')


def numeric_step(payload):
    return int(payload['checkpoint_global_step'])


def matching_diagnostic(payload, field):
    values = payload.get(field, {})
    step = str(numeric_step(payload))
    if step in values:
        return values[step]
    numeric = [key for key in values if str(key).isdigit()]
    return values[max(numeric, key=int)] if numeric else values.get('final', {})


def svg_curve(path, title, ylabel, series):
    width, height = 820, 460
    left, top, right, bottom = 80, 45, 25, 65
    points = [(x, y) for rows in series.values() for x, y in rows
              if y is not None and math.isfinite(float(y))]
    if not points:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>\n')
        return
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + 1
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value):
        return top + (ymax - value) / (ymax - ymin) * plot_h

    colors = ('#2563eb', '#dc2626', '#059669', '#7c3aed')
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-family="sans-serif">optimizer step</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">{ylabel}</text>',
    ]
    for index, (name, rows) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coords = ' '.join(
            f'{sx(x):.1f},{sy(float(y)):.1f}' for x, y in rows
            if y is not None and math.isfinite(float(y)))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(
            f'<text x="{left + 10}" y="{top + 20 + index*18}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">{LABELS.get(name, name)}</text>')
    lines.extend([
        f'<text x="{left}" y="{height-bottom+20}" font-family="sans-serif" font-size="11">{xmin:g}</text>',
        f'<text x="{width-right}" y="{height-bottom+20}" text-anchor="end" font-family="sans-serif" font-size="11">{xmax:g}</text>',
        f'<text x="{left-8}" y="{height-bottom}" text-anchor="end" font-family="sans-serif" font-size="11">{ymin:.4g}</text>',
        f'<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="sans-serif" font-size="11">{ymax:.4g}</text>',
        '</svg>',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def dominates(candidate, control):
    return (
        candidate['raw_brier'] < control['raw_brier']
        and candidate['pY'] > control['pY']
        and candidate['top1'] >= control['top1'])


def noncollapsed(row):
    return (
        row['entropy'] >= 3.5
        and row['repeated_4gram'] <= 0.05
        and row['max_token'] <= 0.10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    reports = root / 'reports'
    curves = reports / 'curves'
    reports.mkdir(parents=True, exist_ok=True)
    curves.mkdir(parents=True, exist_ok=True)

    manifests = {
        arm: read_json(root / 'experiments' / arm / 'run_manifest.json')
        for arm in ARMS}
    posterior_payloads = []
    for arm in ARMS:
        for path in sorted((root / 'posterior' / arm).glob('*/posterior.json')):
            payload = read_json(path)
            payload['_arm'] = arm
            payload['_path'] = str(path)
            posterior_payloads.append(payload)

    training_rows = []
    for arm, manifest in manifests.items():
        training_rows.append({
            'arm': arm,
            'optimizer_steps': manifest['optimizer_steps'],
            'nominal_tokens_seen': manifest['nominal_tokens_seen'],
            'wall_seconds': manifest.get('wall_seconds'),
            'tokens_per_second': manifest.get('nominal_tokens_per_second'),
            'peak_cuda_memory_bytes': manifest.get('peak_cuda_memory_bytes'),
            'ema_decay': manifest.get('ema_contract', {}).get('decay'),
            'ema_actual_update_count': manifest.get(
                'ema_contract', {}).get('actual_update_count'),
        })
    write_csv(root / 'training_summary.csv', training_rows)

    posterior_rows = []
    gamma_rows = []
    gradient_rows = []
    bias_rows = []
    online_ema_rows = []
    prototype_rows = []
    for payload in posterior_payloads:
        arm = payload['_arm']
        step = numeric_step(payload)
        row = {
            'arm': arm, 'step': step, 'weights': payload['weights'],
            'ema_decay': (payload.get('ema') or {}).get('decay'),
            'ema_num_updates': (payload.get('ema') or {}).get('num_updates'),
            'ema_uses_num_update_warmup': (
                payload.get('ema') or {}).get('uses_num_update_warmup'),
            'ema_shadow_initialization': (
                payload.get('ema') or {}).get('shadow_initialization'),
            'raw_brier': metric(payload, 'raw_brier'),
            'diagnostic_ce': metric(payload, 'token_ce'),
            'pY': metric(payload, 'target_probability'),
            'margin': metric(payload, 'target_margin'),
            'top1': metric(payload, 'top1_accuracy'),
            'top10': metric(payload, 'top10_true_inclusion'),
            'top100': metric(payload, 'top100_true_inclusion'),
            'entropy': metric(payload, 'posterior_entropy'),
            'p_max': metric(payload, 'max_probability'),
            'S2': metric(payload, 'probability_square_sum'),
            'abs_gY': metric(payload, 'abs_mse_target_logit_gradient'),
            'one_minus_pY': metric(payload, 'ce_target_gradient_magnitude'),
            'grad_ratio': metric(payload, 'target_gradient_ratio'),
            'fraction_pY_lt_1e_6': metric(payload, 'target_probability_lt_1e_6'),
            'fraction_pY_lt_1e_5': metric(payload, 'target_probability_lt_1e_5'),
            'fraction_pY_lt_1e_4': metric(payload, 'target_probability_lt_1e_4'),
            'fraction_pY_lt_1_over_V': metric(payload, 'target_probability_lt_uniform'),
        }
        posterior_rows.append(row)
        gradient_rows.append({
            key: row[key] for key in (
                'arm', 'step', 'weights', 'pY', 'S2', 'abs_gY',
                'one_minus_pY', 'grad_ratio', 'fraction_pY_lt_1e_6',
                'fraction_pY_lt_1e_5', 'fraction_pY_lt_1e_4',
                'fraction_pY_lt_1_over_V')})
        for gamma_bin, frequency_groups in matching_diagnostic(
                payload, 'posterior_by_gamma_frequency').items():
            for frequency, values in frequency_groups.items():
                gamma_rows.append({
                    'arm': arm, 'step': step, 'weights': payload['weights'],
                    'gamma_bin': gamma_bin, 'frequency': frequency,
                    **values})
        bias_rows.append({
            'arm': arm, 'step': step, 'weights': payload['weights'],
            **matching_diagnostic(payload, 'bias_logit_diagnostics')})
        if arm == 'frozen_mse_independent_prototype':
            route = manifests[arm].get(
                'gradient_route_diagnostics', {}).get(str(step), {})
            prototype_rows.append({
                'arm': arm, 'step': step, 'weights': payload['weights'],
                **matching_diagnostic(payload, 'prototype_diagnostics'),
                'B_gradient_norm': route.get(
                    'classification_prototype_gradient_norm'),
                'B_update_norm': route.get(
                    'classification_prototype_update_norm'),
                'C_gradient_norm': route.get(
                    'physical_codebook_gradient_norm')})

    write_csv(root / 'posterior_by_step.csv', posterior_rows)
    write_csv(root / 'posterior_by_gamma_freq.csv', gamma_rows)
    write_csv(root / 'target_gradient_stats.csv', gradient_rows)
    write_csv(root / 'bias_logit_stats.csv', bias_rows)
    write_csv(root / 'prototype_geometry.csv', prototype_rows)

    final_step = max(manifest['optimizer_steps'] for manifest in manifests.values())
    for arm in ARMS:
        values = {
            row['weights']: row for row in posterior_rows
            if row['arm'] == arm and row['step'] == final_step}
        if 'online' in values and 'ema' in values:
            row = {
                'arm': arm, 'step': final_step,
                'ema_decay': values['ema']['ema_decay'],
                'ema_num_updates': values['ema']['ema_num_updates'],
                'ema_uses_num_update_warmup': values['ema'][
                    'ema_uses_num_update_warmup'],
                'ema_shadow_initialization': values['ema'][
                    'ema_shadow_initialization'],
            }
            for name in ('raw_brier', 'diagnostic_ce', 'pY', 'top1',
                         'entropy', 'grad_ratio'):
                row[f'online_{name}'] = values['online'][name]
                row[f'ema_{name}'] = values['ema'][name]
                row[f'ema_minus_online_{name}'] = (
                    values['ema'][name] - values['online'][name])
            online_ema_rows.append(row)
    write_csv(root / 'online_vs_ema.csv', online_ema_rows)

    generation_rows = []
    generation_payloads = {}
    for arm in ARMS:
        generation_payloads[arm] = {}
        for path in sorted((root / 'generations' / arm).glob('*/samples.json')):
            payload = read_json(path)
            step = int(payload['checkpoint_global_step'])
            weights = payload.get('weights', 'ema')
            generation_payloads[arm][(step, weights)] = payload
            generation_rows.append({
                'arm': arm, 'step': step, 'weights': weights,
                'num_samples': payload['num_samples'],
                'gen_ppl': payload['generative_ppl'],
                'nfe': payload['nfe'], 'solver': payload['solver'],
                'temperature': payload['temperature'],
                'generation_seed': payload['generation_seed'],
                **payload['sample_quality'],
            })
    write_csv(root / 'generation_metrics.csv', generation_rows)

    checkpoints = []
    for arm in ARMS:
        for path in sorted((root / 'experiments' / arm / 'checkpoints').glob('*.ckpt')):
            step = int(path.stem.split('_')[-1])
            roles = [f'step_{step}']
            if step == int(manifests[arm]['optimizer_steps']):
                roles.append('final')
            checkpoints.append(f"{arm}\t{','.join(roles)}\t{path}")
    (root / 'checkpoints.txt').write_text(
        '\n'.join(checkpoints) + '\n', encoding='utf-8')

    with (root / 'representative_samples.md').open('w', encoding='utf-8') as handle:
        handle.write('# Representative fixed-seed samples\n\n')
        for arm in ARMS:
            payload = generation_payloads[arm].get((final_step, 'ema'))
            handle.write(f'## {LABELS[arm]}\n\n')
            for sample in (payload or {}).get('generated_seqs', [])[:4]:
                handle.write(f'```text\n{sample}\n```\n\n')

    ema_rows = [row for row in posterior_rows if row['weights'] == 'ema']
    svg_curve(curves / 'posterior_pY.svg', 'EMA posterior pY', 'pY', {
        arm: sorted((row['step'], row['pY']) for row in ema_rows
                    if row['arm'] == arm) for arm in ARMS})
    svg_curve(curves / 'target_gradient_ratio.svg',
              'EMA target-gradient ratio', '|gY| / (1-pY)', {
        arm: sorted((row['step'], row['grad_ratio']) for row in ema_rows
                    if row['arm'] == arm) for arm in ARMS})
    svg_curve(curves / 'generation_repetition.svg',
              'EMA repeated 4-gram fraction', 'fraction', {
        arm: sorted((row['step'], row['repeated_4gram_fraction'])
                    for row in generation_rows
                    if row['arm'] == arm and row['weights'] == 'ema')
        for arm in ARMS})

    final_posterior = {
        row['arm']: row for row in ema_rows if row['step'] == final_step}
    final_generation = {
        row['arm']: {
            'gen_ppl': row['gen_ppl'],
            'entropy': row['mean_sample_unigram_entropy_nats'],
            'repeated_4gram': row['repeated_4gram_fraction'],
            'max_token': row['max_single_token_fraction'],
        } for row in generation_rows
        if row['step'] == final_step and row['weights'] == 'ema'}
    control = final_posterior['frozen_mse_warmup_bias']
    alternatives = ARMS[1:]
    improved = {
        arm: dominates(final_posterior[arm], control) for arm in alternatives}
    all_collapsed = all(not noncollapsed(final_generation[arm]) for arm in ARMS)
    if all_collapsed and any(improved.values()):
        case = 'CASE E'
        case_reason = 'posterior improved versus warmup, but every final EMA generation failed the collapse gate'
    elif not any(improved.values()):
        case = 'CASE D'
        case_reason = 'no intervention jointly improved raw Brier, pY, and Top-1 versus warmup'
    else:
        eligible = [arm for arm in alternatives
                    if improved[arm] and noncollapsed(final_generation[arm])]
        candidates = eligible or [arm for arm in alternatives if improved[arm]]
        best = max(candidates, key=lambda arm: (
            final_posterior[arm]['pY'], final_posterior[arm]['top1'],
            -final_posterior[arm]['raw_brier']))
        case = {
            'frozen_mse_no_bias': 'CASE A',
            'frozen_mse_full_bias': 'CASE B',
            'frozen_mse_independent_prototype': 'CASE C',
        }[best]
        case_reason = f'{LABELS[best]} is the strongest posterior-improving candidate under the collapse gate'

    low_noise = [row for row in gamma_rows
                 if row['weights'] == 'ema' and row['step'] == final_step
                 and row['gamma_bin'] in {'0.0-0.1', '0.1-0.2'}
                 and row['frequency'] in {'mid', 'low'}]
    tail_by_arm = {}
    for arm in ARMS:
        values = [float(row['target_probability_lt_1e_4'])
                  for row in low_noise if row['arm'] == arm]
        tail_by_arm[arm] = sum(values) / len(values) if values else math.inf
    best_tail = min(tail_by_arm, key=tail_by_arm.get)
    gating_values = [float(row['target_gradient_ratio'])
                     for row in low_noise]
    gating_supported = (
        sum(value < 1.0 for value in gating_values)
        > len(gating_values) / 2 if gating_values else False)

    first_repeat = {}
    for arm in ARMS:
        rows = sorted((row for row in generation_rows
                       if row['arm'] == arm and row['weights'] == 'ema'),
                      key=lambda row: row['step'])
        failed = [row['step'] for row in rows
                  if row['repeated_4gram_fraction'] > 0.05]
        first_repeat[arm] = min(failed) if failed else None
    observed = {arm: step for arm, step in first_repeat.items()
                if step is not None}
    earliest = (min(observed, key=observed.get) if observed else None)

    def comparison(arm):
        return ('yes' if improved[arm] else 'no or mixed')

    healthy = [arm for arm in ARMS
               if noncollapsed(final_generation[arm])
               and final_generation[arm]['gen_ppl'] <= 500
               and (arm == ARMS[0] or improved.get(arm, False))]
    ratio_text = ', '.join(
        f"{LABELS[arm]}={final_posterior[arm]['grad_ratio']:.6g}"
        for arm in ARMS)
    brier_decoupled = any(
        (final_posterior[arm]['raw_brier'] < control['raw_brier'])
        != (final_posterior[arm]['pY'] > control['pY']
            and final_posterior[arm]['top1'] >= control['top1'])
        for arm in alternatives)
    earliest_text = (
        f'{LABELS[earliest]} at step {observed[earliest]}'
        if earliest else 'none at evaluated checkpoints')
    healthy_text = (
        ', '.join(LABELS[arm] for arm in healthy) if healthy else 'none')
    online_rows = {
        row['arm']: row for row in posterior_rows
        if row['weights'] == 'online' and row['step'] == final_step}
    ema_order = sorted(ARMS, key=lambda arm: final_posterior[arm]['pY'],
                       reverse=True)
    online_order = sorted(ARMS, key=lambda arm: online_rows[arm]['pY'],
                          reverse=True)
    ema_material = ema_order != online_order
    max_ema_py_delta = max(
        online_ema_rows, key=lambda row: abs(row['ema_minus_online_pY']))
    report = f'''# Four-GPU Gaussian-bias ablation report

## Data and curves

- Shared optimizer steps: {final_step}
- [Training summary](training_summary.csv)
- [Posterior by step](posterior_by_step.csv)
- [Gamma x frequency posterior](posterior_by_gamma_freq.csv)
- [Generation metrics](generation_metrics.csv)
- [pY curve](reports/curves/posterior_pY.svg)
- [Target-gradient ratio curve](reports/curves/target_gradient_ratio.svg)
- [Generation repetition curve](reports/curves/generation_repetition.svg)

## Registered questions

1. No-bias better than warmup bias: **{comparison('frozen_mse_no_bias')}**.
2. Full matched bias better than warmup bias: **{comparison('frozen_mse_full_bias')}**.
3. Independent B better than shared-C warmup bias: **{comparison('frozen_mse_independent_prototype')}**.
4. Best protection against very low pY in low-noise mid/low-frequency buckets: **{LABELS[best_tail]}**.
5. Does measured |gY| versus 1-pY support softmax-Brier gradient gating: **{'yes' if gating_supported else 'no or mixed'}**; final EMA ratios are {ratio_text}.
6. Raw Brier can decouple from pY/Top-1: **{'yes' if brier_decoupled else 'not observed'}**.
7. Earliest repeated-generation failure: **{earliest_text}**.
8. Material online/EMA difference: **{'yes' if ema_material else 'no'}** by whether final pY arm ordering changes; largest pY delta is {LABELS[max_ema_py_delta['arm']]}={max_ema_py_delta['ema_minus_online_pY']:.6g}.
9. Posterior-better, non-collapsed, reasonable-Gen.PPL arm: **{healthy_text}**.

## Final classification

**{case}** — {case_reason}.
'''
    (root / 'BIAS_ABLATION_4GPU_REPORT.md').write_text(
        report, encoding='utf-8')
    (root / 'README.md').write_text(
        '# Bias ablation evidence\n\nCanonical result: '
        '[BIAS_ABLATION_4GPU_REPORT.md](BIAS_ABLATION_4GPU_REPORT.md).\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
