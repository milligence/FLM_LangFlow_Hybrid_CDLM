#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algo import langflow_alpha_sigma, langflow_gumbel_gamma


def checkpoint_codebook(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    key = next(key for key in checkpoint['state_dict']
               if key.endswith('vocab_embed.embedding'))
    raw = checkpoint['state_dict'][key].float()
    codebook = F.normalize(raw, dim=-1) * math.sqrt(raw.shape[-1])
    return codebook


@torch.inference_mode()
def diagnose(name, checkpoint_path, samples_per_bin, device):
    codebook = checkpoint_codebook(checkpoint_path)
    codebook = codebook.to(device)
    generator = torch.Generator(device=device).manual_seed(20260914)
    rows = []
    for prior_name in ('uniform',):
        prior = torch.full(
            (codebook.shape[0],), 1 / codebook.shape[0], device=device)
        for bin_index in range(10):
            q_value = torch.full(
                (samples_per_bin,), (bin_index + 0.5) / 10, device=device)
            gamma = langflow_gumbel_gamma(q_value)
            alpha, sigma = langflow_alpha_sigma(gamma)
            targets = torch.multinomial(
                prior, samples_per_bin, replacement=True, generator=generator)
            noise = torch.randn(
                samples_per_bin, codebook.shape[1],
                device=device, generator=generator)
            z = alpha[:, None] * codebook[targets] + sigma[:, None] * noise
            logits = ((alpha / sigma.square())[:, None]
                      * (z @ codebook.transpose(0, 1))) + prior.log()[None, :]
            probabilities = logits.softmax(dim=-1)
            true = probabilities.gather(1, targets[:, None]).squeeze(1)
            top_values, top_indices = probabilities.topk(2, dim=-1)
            other = torch.where(
                top_indices[:, 0] == targets, top_values[:, 1], top_values[:, 0])
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
            rows.append({
                'run': name,
                'prior': prior_name,
                'gamma_quantile_bin': f'{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}',
                'sample_count': samples_per_bin,
                'top1_recovery_accuracy': float((probabilities.argmax(-1) == targets).float().mean()),
                'true_token_posterior': float(true.mean()),
                'margin': float((true - other).mean()),
                'entropy': float(entropy.mean()),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', nargs=2, metavar=('NAME', 'CHECKPOINT'), required=True)
    parser.add_argument('--samples-per-bin', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for name, path in args.run:
        rows.extend(diagnose(name, Path(path), args.samples_per_bin, device))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
