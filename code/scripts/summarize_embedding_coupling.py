#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


RUNS = ('original_ce', 'original_mse', 'dbias_mse', 'frozen_mse', 'frozen_ce')


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def last_metric(csv_path, name):
    if not csv_path.exists():
        return ''
    value = ''
    with csv_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get(name, '') != '':
                value = row[name]
    return value


def final_geometry(manifest):
    values = manifest.get('embedding_nearest_neighbor_cosine_distance', {})
    if 'final' in values:
        return values['final']
    numeric = [key for key in values if key.isdigit()]
    return values[max(numeric, key=int)] if numeric else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    reports = args.root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    manifests = {}
    evals = {}
    for run in RUNS:
        directory = args.root / 'experiments' / run
        manifests[run] = read_json(directory / 'run_manifest.json')
        evals[run] = read_json(directory / 'eval' / 'samples.json')

    metrics_rows = []
    geometry_rows = []
    generation_rows = []
    posterior_rows = []
    gradient_rows = []
    for run in RUNS:
        manifest = manifests[run]
        evaluation = evals[run]
        train_csv = args.root / 'experiments' / run / 'local_metrics' / 'metrics.csv'
        geometry = final_geometry(manifest)
        metrics_rows.append({
            'run': run,
            'loss': manifest['loss_contract']['loss_type'],
            'weights': 'ema',
            'raw_brier': last_metric(train_csv, 'val/hybrid_raw_brier'),
            'diagnostic_ce': last_metric(train_csv, 'val/hybrid_token_ce'),
            'pY': last_metric(train_csv, 'val/hybrid_target_probability'),
            'margin': last_metric(train_csv, 'val/hybrid_target_margin'),
            'nn_distance_mean': geometry.get('nearest_cosine_distance_mean', ''),
            'nn_p05': geometry.get('nearest_cosine_distance_p05', ''),
            'effective_rank': geometry.get('centered_effective_rank', ''),
            'gen_ppl': evaluation['generative_ppl'],
            'entropy': evaluation['sample_quality']['mean_sample_unigram_entropy_nats'],
            'distinct_1': evaluation['sample_quality']['distinct_1'],
            'distinct_2': evaluation['sample_quality']['distinct_2'],
            'rep_4gram': evaluation['sample_quality']['repeated_4gram_fraction'],
        })
        geometry_rows.append({'run': run, **geometry})
        generation_rows.append({
            'run': run,
            'weights': evaluation.get('weights', 'ema'),
            'bias_weight': evaluation.get('token_bias_weight', ''),
            'nfe': evaluation['nfe'],
            'solver': evaluation.get('solver', 'euler'),
            'generation_seed': evaluation.get('generation_seed', 42),
            'gen_ppl': evaluation['generative_ppl'],
            **evaluation['sample_quality'],
        })
        posterior = manifest.get('posterior_by_gamma_frequency', {})
        if posterior:
            step = max(posterior, key=int)
            for gamma_bin, frequency_groups in posterior[step].items():
                for frequency, values in frequency_groups.items():
                    posterior_rows.append({
                        'run': run, 'step': step, 'gamma_bin': gamma_bin,
                        'frequency': frequency, **values})
        for step, values in manifest.get('gradient_route_diagnostics', {}).items():
            gradient_rows.append({'run': run, 'step': step, **values})

    write_csv(reports / 'metrics_table.csv', metrics_rows)
    write_csv(reports / 'embedding_geometry.csv', geometry_rows)
    write_csv(reports / 'generation_metrics.csv', generation_rows)
    write_csv(reports / 'posterior_by_gamma_freq.csv', posterior_rows)
    write_csv(reports / 'gradient_route_stats.csv', gradient_rows)

    with (reports / 'representative_samples.md').open('w') as handle:
        handle.write('# Fixed-seed representative samples\n\n')
        handle.write('All samples use EMA weights, seed 42, length 128, 128 NFE, Euler, temperature 1.0.\n')
        for run in RUNS:
            handle.write(f'\n## {run}\n')
            for sample in evals[run]['generated_seqs'][:3]:
                handle.write(f'\n```text\n{sample}\n```\n')

    def noncollapsed(run):
        quality = evals[run]['sample_quality']
        return (quality['mean_sample_unigram_entropy_nats'] >= 3.5
                and quality['repeated_4gram_fraction'] <= 0.05
                and quality['max_single_token_fraction'] <= 0.10)

    fce_ok = noncollapsed('frozen_ce')
    fmse_ok = noncollapsed('frozen_mse')
    if fce_ok and fmse_ok:
        case = 2
        next_step = '结构解耦 C + S + W'
    elif fce_ok and not fmse_ok:
        case = 3
        next_step = 'MSE loss / posterior parameterization'
    else:
        case = 4
        next_step = '先核查固定初始 codebook 与 gamma/noise 匹配；不启动 P2'
    dbias_quality = evals['dbias_mse']['sample_quality']
    original_quality = evals['original_mse']['sample_quality']
    dbias_improved = (
        dbias_quality['repeated_4gram_fraction']
        < 0.75 * original_quality['repeated_4gram_fraction']
        and dbias_quality['mean_sample_unigram_entropy_nats']
        > original_quality['mean_sample_unigram_entropy_nats'] + 0.3)
    summary = f'''# Causal matrix summary

The 64-sample generation results are development diagnostics and are **non-comparable**, not formal quality claims.

Operational non-collapse rule for this decision: entropy >= 3.5 nats, repeated 4-gram fraction <= 0.05, and maximum-token fraction <= 0.10.

- Decision-tree case: **Case {case}**.
- D-bias material generation improvement: **{str(dbias_improved).lower()}** (requires both >=25% repetition reduction and >=0.3-nat entropy recovery).
- Next registered direction: **{next_step}**.

## Required questions

1. **Is jointly learned physical codebook necessary for MSE degradation?** {'No: frozen MSE still collapsed.' if not fmse_ok else 'The controlled frozen-MSE diagnostic did not collapse; joint learning remains implicated.'}
2. **Is candidate Gaussian-bias gradient the main geometry-contraction route?** {'Generation evidence supports an important contribution; consult geometry and route tables before calling it dominant.' if dbias_improved else 'No material generation rescue was observed under the registered threshold; the evidence does not establish it as dominant.'}
3. **Does fixed-codebook MSE retain posterior/frequency bias?** See `posterior_by_gamma_freq.csv`; interpret jointly with frozen-CE.
4. **Next investment?** {next_step}.
'''
    (reports / 'causal_matrix_summary.md').write_text(summary)


if __name__ == '__main__':
    main()
