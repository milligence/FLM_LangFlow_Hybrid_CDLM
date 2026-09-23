#!/usr/bin/env python3
"""Write the actual-code/checkpoint audit required before the F-line patch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import yaml


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ['git', *arguments], cwd=repo, text=True,
        stderr=subprocess.DEVNULL).strip()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--existing-audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bindings = read_json(args.bindings)
    repo = Path(bindings['repo_root']).resolve()
    contract = args.contract_dir.resolve()
    train = yaml.safe_load((contract / 'train_f.yaml').read_text())
    sampler = yaml.safe_load((contract / 'sampler_f.yaml').read_text())
    audit_path = contract / 'audit_30k.yaml'
    if not audit_path.is_file():
        audit_path = args.existing_audit_root / 'contracts/audit_30k.yaml'
    audit = yaml.safe_load(audit_path.read_text())
    online = read_json(
        args.existing_audit_root / 'diagnostics/step_030000__online.json')
    controller = online['controller']
    calibration = controller['latest_calibration']
    local_norm = statistics.median(
        float(row['local_norm']) for row in calibration['packs'])
    raw_map_norm = statistics.median(
        float(row['raw_map_norm']) for row in calibration['packs'])
    c_map = float(controller['C_map'])
    try:
        status = git(repo, 'status', '--porcelain=v1')
        diff = subprocess.check_output(
            ['git', 'diff', '--binary', 'HEAD'], cwd=repo)
        git_commit = git(repo, 'rev-parse', 'HEAD')
        git_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        status = ''
        diff = b''
        git_commit = bindings['provenance']['code_commit']
        git_available = False

    generation = {}
    for name in ('highnfe_512', 'fewstep_1',
                 'fewstep_4_deployment_grid_1',
                 'fewstep_4_deployment_grid_2'):
        payload = read_json(
            args.existing_audit_root / 'generation' / name / 'samples.json')
        generation[name] = {
            key: payload.get(key) for key in (
                'generative_ppl', 'entropy', 'nfe', 'solver', 'temperature',
                'generation_seed', 'weights', 'num_samples', 'sample_quality')}

    payload = {
        'audit_id': 'task1-f-patch-actual-30k-v1',
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': digest(args.checkpoint),
            'global_step': 30000,
            'resume_policy': 'exact full training state; optimizer and EMAs preserved',
        },
        'repository': {
            'path': str(repo),
            'git_repository_available': git_available,
            'git_commit': git_commit,
            'dirty': bool(status) if git_available else None,
            'status_porcelain': status.splitlines(),
            'dirty_diff_sha256': (
                hashlib.sha256(diff).hexdigest() if git_available else None),
            'source_manifest_sha256': bindings['provenance'][
                'source_sha256_manifest'],
        },
        'weights': {
            'available': ['online', 'target_ema', 'eval_ema'],
            'primary_generation': 'eval_ema',
            'canonical_training_teacher': 'target_ema',
            'target_ema_decay': train['ema']['target_decay'],
            'eval_ema_decay': train['ema']['eval_decay'],
            'actual_canonical_K': controller['canonical_K'],
            'canonical_previous_K': controller['canonical_previous_K'],
            'canonical_transition_start': controller['canonical_transition_start'],
        },
        'generation_semantics': {
            'legacy_512': {
                'field': 'local posterior', 'eta': 0.0,
                'solver': 'Euler', 'self_conditioning': 'rolling',
                'terminal': 1.0, 'actual_forward_count': 512,
                'temperature': 1.0, 'decode': 'argmax final state',
                'extra_final_denoise': False,
            },
            'finite_4': {
                'queries': 4, 'self_conditioning': 'cold every query',
                'eta_formula': '(s-r)/(1-r)',
                'update': 'x_next=(1-eta)*x+eta*A',
                'A_postprocess': 'none; no softmax or clamp of A',
                'extra_final_denoise': False,
            },
            'observed_30k': generation,
        },
        'residual_semantics': {
            'legacy_map_residual': (
                'mean over tokens/sequences of L2 norm across vocabulary of '
                '(1-s)*(predicted_terminal_velocity-target_terminal_velocity)'),
            'legacy_is_squared': False,
            'legacy_vocab_reduction': 'L2 norm across vocabulary',
            'legacy_half_factor': False,
            'legacy_cosine': 'raw predicted versus target terminal velocity',
            'new_signal_residual': 'mean_token sum_vocab (Q-p_T)^2',
            'new_raw_terminal_residual': 'mean_token sum_vocab ((Q-p_T)/(1-s))^2',
            'F_Q_KL': None,
            'F_Q_KL_reason': 'A and Q are signed and are not categorical distributions',
        },
        'training_source_frequencies_at_30k': {
            'data': 96,
            'canonical_reference_path': 0,
            'own_finite_rollout': 0,
            'D': {
                'total': 24, 'grid_A': 12, 'grid_B': 12,
                'source': 'analytical noised data',
            },
        },
        'local_probe': {
            'validation_split': audit['heldout_local']['split'],
            'sequences': audit['heldout_local']['first_sequences'],
            'physical_t_nodes': audit['heldout_local']['physical_t_nodes'],
            'bins': audit['heldout_local']['physical_bins'],
            'aggregation': 'equal weight per listed physical-t node',
            'loss': '0.5*sum_vocab(p-onehot)^2, averaged over tokens and sequences',
            'evaluator': bindings['legacy']['gpt2large_evaluator'],
            'tokenizer': bindings['data']['tokenizer'],
            'tokenizer_revision': bindings['data'].get('tokenizer_revision'),
            'noise_seed': audit['heldout_local']['noise_seed'],
        },
        'optimizer_controller_at_30k': {
            'next_update_learning_rates': {
                'legacy': 0.0005994,
                'finite_G': 0.0005994,
                'finite_B': 0.00005994,
            },
            'rho_map_target': 0.30,
            'C_map': c_map,
            'lambda_map': 0.30 * c_map,
            'local_grad_norm_median': local_norm,
            'raw_map_grad_norm_median': raw_map_norm,
            'weighted_map_to_local_ratio': (
                0.30 * c_map * raw_map_norm / local_norm),
            'C_closure': controller['C_closure'],
            'closure_rows': 32,
            'closure_classes': ['S', 'M'],
            'latest_calibration_step': calibration['completed_updates'],
        },
        'deviations_or_unavailable': [
            'The earlier 30k audit did not store new rollout/reference trajectory metrics.',
            'The patch diagnostic will measure them before any rollout-source training is enabled.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
