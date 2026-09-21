#!/usr/bin/env python3
"""Extract the persisted sampler CDF from an A-arm full-state checkpoint."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('output')
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    payload = {
        'sampler_t_nodes': state['sampler_t_nodes'].float().cpu(),
        'sampler_cdf_values': state['sampler_cdf_values'].float().cpu(),
        'sampler_cdf_version': int(state['sampler_cdf_version']),
        'source_checkpoint_global_step': int(checkpoint['global_step']),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        'output': str(output),
        'source_checkpoint_global_step': payload['source_checkpoint_global_step'],
        'sampler_cdf_version': payload['sampler_cdf_version'],
        'nodes': int(payload['sampler_t_nodes'].numel()),
    }))


if __name__ == '__main__':
    main()
