#!/usr/bin/env python3
"""Select the longest registered run that fits the four-card wall budget."""

import argparse
import json
import math
from pathlib import Path


ARMS = (
    'frozen_mse_warmup_bias',
    'frozen_mse_no_bias',
    'frozen_mse_full_bias',
    'frozen_mse_independent_prototype',
)


def percentile90(values):
    values = sorted(values)
    return values[max(math.ceil(0.9 * len(values)) - 1, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--elapsed-seconds', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    by_arm = {}
    for arm in ARMS:
        rows = [json.loads(line) for line in (
            args.smoke_root / arm / 'throughput.jsonl'
        ).read_text(encoding='utf-8').splitlines()]
        tail = [row['step_seconds'] for row in rows
                if int(row['optimizer_step']) > 10]
        if len(tail) < 5:
            raise SystemExit(f'{arm}: insufficient post-warmup throughput rows')
        by_arm[arm] = percentile90(tail)

    slowest_arm = max(by_arm, key=by_arm.get)
    p90_step_seconds = by_arm[slowest_arm]
    primary_deadline_seconds = 110 * 60
    evaluation_reserve_seconds = 20 * 60
    selected = None
    estimates = {}
    for steps in (10000, 8000, 6000):
        total = (args.elapsed_seconds + p90_step_seconds * steps
                 + evaluation_reserve_seconds)
        estimates[str(steps)] = total
        if selected is None and total <= primary_deadline_seconds:
            selected = steps
    payload = {
        'smoke_elapsed_seconds': args.elapsed_seconds,
        'p90_step_seconds_by_arm': by_arm,
        'slowest_arm': slowest_arm,
        'slowest_p90_step_seconds': p90_step_seconds,
        'evaluation_reserve_seconds': evaluation_reserve_seconds,
        'primary_deadline_seconds': primary_deadline_seconds,
        'estimated_total_seconds_by_candidate': estimates,
        'selected_steps': selected,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    if selected is None:
        raise SystemExit('Even 6000 steps cannot fit the registered budget.')
    print(selected)


if __name__ == '__main__':
    main()
