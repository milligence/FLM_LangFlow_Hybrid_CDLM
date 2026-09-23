"""Summarize only the three user-selected Task1 metrics across checkpoints."""

import argparse
import json
from pathlib import Path


def summarize(paths):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    rows.sort(key=lambda row: row['checkpoint_global_step'])
    if not rows:
        raise ValueError('At least one Task1 metrics file is required.')
    return {
        'metrics': rows,
        'targets': {
            'generative_perplexity': [100.0, 150.0],
            'mean_sample_unigram_entropy_nats': 'increase',
            'mauve': 'increase',
        },
        'unigram_entropy_increased': (
            None if len(rows) < 2
            else rows[-1]['mean_sample_unigram_entropy_nats']
            > rows[0]['mean_sample_unigram_entropy_nats']),
        'mauve_increased': (
            None if len(rows) < 2
            else rows[-1]['mauve'] > rows[0]['mauve']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics', nargs='+')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = summarize(args.metrics)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
