#!/usr/bin/env python3
"""Attach K-specific generation metrics and write dynamic F 2k metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()

    target_path = (
        args.output_root / 'diagnostics'
        / f'step_{args.step:06d}__target_ema.json')
    target = json.loads(target_path.read_text(encoding='utf-8'))
    generation = {}
    for k in (1, 2, 4):
        path = args.output_root / 'generation' / f'sc_k{k}' / 'samples.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        generation[k] = {
            'generative_ppl': payload.get('generative_ppl'),
            'entropy': payload.get('entropy'),
            'samples': payload.get('num_samples'),
            'forward_budget': payload.get('nfe'),
        }
    for row in target.get('SC_metrics', []):
        metric = generation[int(row['K'])]
        row['generation_metric'] = metric['generative_ppl']
        row['generation_entropy'] = metric['entropy']
        row['generation_samples'] = metric['samples']
        row['generation_forward_budget'] = metric['forward_budget']
    atomic_json(target_path, target)
    atomic_json(args.output_root / 'metadata.json', {
        'step': args.step,
        'checkpoint_path_at_audit_time': str(args.checkpoint.resolve()),
        'checkpoint_retention': 'rolling temporary; not required after audit',
        'config_path': str(args.config.resolve()),
        'SC_generation': generation,
    })


if __name__ == '__main__':
    main()
