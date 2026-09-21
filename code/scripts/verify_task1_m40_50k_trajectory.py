#!/usr/bin/env python3
"""Validate the selected M40-to-50k 1024x512 trajectory artifact."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROTOCOL_ID = (
    'owt128-gpt2large-genppl-diagnostic-'
    'm40-50k-uniform-t-512-s1024-trajectory-v1')


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def finite_metrics(metrics, name):
    require(metrics is not None, f'{name} is missing')
    require(int(metrics['element_count']) > 0, f'{name} has no elements')
    for field in ('mean_absolute', 'rms', 'max_absolute'):
        require(math.isfinite(float(metrics[field])),
                f'{name}.{field} is nonfinite')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint-label', choices=(
        'm40_stable_50k', 'm40_cooldown_50k'), required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--implementation-commit', required=True)
    args = parser.parse_args()

    samples_path = args.run_dir / 'samples.json'
    diagnostics_path = args.run_dir / 'task1_sampling_diagnostics.json'
    trajectory_path = args.run_dir / 'task1_trajectory_diagnostics.json'
    samples = json.loads(samples_path.read_text(encoding='utf-8'))
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    trajectory = json.loads(trajectory_path.read_text(encoding='utf-8'))
    require(samples['protocol_id'] == PROTOCOL_ID, 'protocol ID mismatch')
    require(samples['weights'] == 'ema', 'EMA weights required')
    require(samples['generation_seed'] == 42, 'seed 42 required')
    require((samples['num_samples'], samples['nfe']) == (1024, 512),
            'trajectory must be 1024 samples by 512 NFE')
    require(samples['checkpoint_global_step'] == 50000,
            'selected checkpoint must be 50k')
    require(samples['trajectory_diagnostics_artifact'] ==
            'task1_trajectory_diagnostics.json',
            'trajectory artifact pointer missing')
    require(len(samples['generated_token_ids']) == 1024,
            'generated token count mismatch')
    require(len(samples['per_sample_scores']) == 1024,
            'per-sample scores missing')
    require(diagnostics['grid']['physical_time_grid'] ==
            'linspace(0,1,513)', 'physical-time grid mismatch')
    require(diagnostics['initial_noise_schedule'] ==
            'base_seed_plus_sample_index', 'sample-index noise required')

    require(trajectory['schema'] ==
            'task1-uniform-t-trajectory-diagnostics-v2',
            'trajectory schema mismatch')
    require('generated tokens are never treated as ground-truth labels' in
            trajectory['artifact_semantics'], 'unlabeled semantics missing')
    identity = trajectory['identity']
    require(identity['checkpoint_label'] == args.checkpoint_label,
            'trajectory checkpoint label mismatch')
    require((identity['reference_nfe'], identity['sample_count']) ==
            (512, 1024), 'trajectory identity mismatch')
    rows = trajectory['physical_intervals']['rows']
    require(len(rows) == 512, 'trajectory interval rows incomplete')
    for index, row in enumerate(rows):
        require(row['interval_index'] == index, 'interval index mismatch')
        require(abs(row['start_physical_t'] - index / 512) < 2e-7,
                'interval physical-time mismatch')
        finite_metrics(row['vector_field'], 'vector_field')
        for field in ('top1_probability_mean', 'top2_probability_mean',
                      'top1_top2_margin_mean'):
            require(math.isfinite(float(row[field])), f'{field} nonfinite')
        require(row['top1_probability_mean'] >= row['top2_probability_mean'],
                'Top-1 probability is below Top-2')
        if index == 0:
            require(row['probability_change_direct'] is None,
                    'first row must not claim a probability delta')
            require(row['velocity_change'] is None,
                    'first row must not claim a velocity delta')
            require(row['self_conditioning']['input_available'] is False,
                    'first query must have no SC input')
        else:
            finite_metrics(row['probability_change_direct'], 'delta_p')
            finite_metrics(row['velocity_change'], 'delta_v')
            finite_metrics(row['probability_delta_reconstruction_error'],
                           'delta_p_reconstruction_error')
            require(0.0 <= row['probability_total_variation_mean'] <= 1.0,
                    'probability TV outside [0,1]')
            require(0.0 <= row['top1_flip_rate'] <= 1.0,
                    'Top-1 flip rate outside [0,1]')
            require(row['self_conditioning']['input_available'] is True,
                    'SC input missing after the first query')
            require(row['self_conditioning']['input_finite'] is True,
                    'SC input contains nonfinite values')

    fp32 = trajectory['fp32_forward_sensitivity']
    require(fp32['sample_count'] == 8, 'fp32 sample count mismatch')
    require(fp32['common_physical_nodes'] is True,
            'fp32 probes are not on common physical nodes')
    require(len(fp32['rows']) == 17, 'fp32 node rows incomplete')
    for row in fp32['rows']:
        finite_metrics(row['probability_difference'], 'fp32_probability')
        for field in ('top1_changed_rate', 'top2_set_changed_rate'):
            require(0.0 <= row[field] <= 1.0, f'{field} outside [0,1]')

    total_nll = sum(float(row['nll']) for row in samples['per_sample_scores'])
    total_tokens = sum(int(row['scored_tokens'])
                       for row in samples['per_sample_scores'])
    completion = {
        'schema': 'task1-m40-50k-trajectory-completion-v1',
        'status': 'completed',
        'completed_at_asia_shanghai': datetime.now(
            ZoneInfo('Asia/Shanghai')).isoformat(),
        'checkpoint_label': args.checkpoint_label,
        'checkpoint_path': str(args.checkpoint_path),
        'checkpoint_global_step': 50000,
        'implementation_commit': args.implementation_commit,
        'protocol_id': PROTOCOL_ID,
        'num_samples': 1024,
        'nfe': 512,
        'generative_ppl': samples['generative_ppl'],
        'recomputed_generative_ppl': math.exp(total_nll / total_tokens),
        'trajectory_schema': trajectory['schema'],
        'physical_interval_count': 512,
        'fp32_sample_count': 8,
        'fp32_node_count': 17,
        'output_streams_closed': True,
        'artifacts': {
            'samples': str(samples_path),
            'sampling_diagnostics': str(diagnostics_path),
            'trajectory_diagnostics': str(trajectory_path),
            'launcher_log': f'{args.run_dir}.launcher.log',
        },
    }
    (args.run_dir / 'completion.json').write_text(
        json.dumps(completion, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
