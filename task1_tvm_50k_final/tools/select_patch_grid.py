#!/usr/bin/env python3
"""Apply the bounded A/B/C/D scheduler review without claiming quality gains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

GRIDS = {
    'A': [0.0, 0.3392770637, 0.5814685447, 0.7246687714, 0.95],
    'B': [0.0, 0.2375, 0.475, 0.7125, 0.95],
    'C': [0.0, 0.3392770637, 0.5814685447, 0.85, 0.95],
    'D': [0.0, 0.527129195498412, 0.776393202250021, 0.894262873655944, 0.95],
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def mean_nll(payload: dict) -> float:
    scores = payload['per_sample_scores']
    return sum(float(row['nll']) for row in scores) / sum(
        int(row['scored_tokens']) for row in scores)


def health(candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
    cq, bq = candidate['sample_quality'], baseline['sample_quality']
    failures = []
    if float(candidate['entropy']) < float(baseline['entropy']) - 0.10:
        failures.append('mean_sequence_entropy_drop')
    if cq['distinct_1'] < 0.90 * bq['distinct_1']:
        failures.append('distinct_1')
    if cq['distinct_2'] < 0.90 * bq['distinct_2']:
        failures.append('distinct_2')
    if cq['repeated_4gram_fraction'] > bq['repeated_4gram_fraction'] + 0.005:
        failures.append('repeat_4gram')
    if cq['max_single_token_fraction'] > bq['max_single_token_fraction'] + 0.01:
        failures.append('max_token_fraction')
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--f20-root', type=Path, required=True)
    parser.add_argument('--f30-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    payloads: dict[int, dict[str, dict]] = {20000: {}, 30000: {}}
    for step, root in ((20000, args.f20_root), (30000, args.f30_root)):
        payloads[step]['A'] = read(
            root / 'generation/fewstep_4_deployment_grid_1/samples.json')
        payloads[step]['B'] = read(
            root / 'generation/fewstep_4_deployment_grid_2/samples.json')
        for name in ('C', 'D'):
            payloads[step][name] = read(
                args.new_root / f'step_{step:06d}' / f'grid_{name}' / 'samples.json')

    rows = []
    for step in (20000, 30000):
        legacy = min(
            (payloads[step][name] for name in ('A', 'B')),
            key=lambda item: mean_nll(item))
        legacy_name = min(('A', 'B'), key=lambda name: mean_nll(payloads[step][name]))
        for name, payload in payloads[step].items():
            healthy, failures = health(payload, legacy)
            row = {
                'checkpoint_step': step,
                'grid': name,
                'nodes': GRIDS[name],
                'generative_ppl': payload['generative_ppl'],
                'mean_scored_nll': mean_nll(payload),
                'entropy': payload['entropy'],
                **payload['sample_quality'],
                'health_qualified': healthy,
                'health_failures': failures,
                'best_legacy_grid': legacy_name,
                'nll_gap_from_best_legacy': mean_nll(payload) - mean_nll(legacy),
            }
            rows.append(row)

    by_name = {name: [row for row in rows if row['grid'] == name]
               for name in GRIDS}
    admitted = []
    for name in ('C', 'D'):
        by_step = {row['checkpoint_step']: row for row in by_name[name]}
        best20 = min(payloads[20000][key]['generative_ppl'] for key in ('A', 'B'))
        best30 = min(payloads[30000][key]['generative_ppl'] for key in ('A', 'B'))
        if (all(row['health_qualified'] for row in by_name[name])
                and by_step[30000]['generative_ppl'] <= 0.95 * best30
                and by_step[20000]['generative_ppl'] <= 1.05 * best20):
            admitted.append(name)
    if admitted:
        provisional = min(
            admitted,
            key=lambda name: max(row['nll_gap_from_best_legacy']
                                 for row in by_name[name]))
        status = 'provisional_new_grid_requires_32k_independent_confirmation'
    else:
        provisional = min(
            ('A', 'B'), key=lambda name: mean_nll(payloads[30000][name]))
        status = 'fallback_best_health_qualified_legacy_at_30k'

    with (output / 'scheduler_candidates.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    seed_fingerprint = hashlib.sha256(b'424242:first128').hexdigest()
    selected = {
        'selection_status': status,
        'G_star': provisional,
        'grid': GRIDS[provisional],
        'new_grid_admitted_at_20k30k': admitted,
        'requires_32k_independent_confirmation': provisional in {'C', 'D'},
        'seed_bank': 424242,
        'seed_bank_fingerprint': seed_fingerprint,
        'interpretation': 'bounded engineering scheduler review; no quality claim',
    }
    (output / 'selected_grid.json').write_text(
        json.dumps(selected, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
