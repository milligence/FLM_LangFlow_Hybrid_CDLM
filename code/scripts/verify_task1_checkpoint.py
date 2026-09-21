#!/usr/bin/env python3
"""Read one Task1 checkpoint and verify the full-state resume contract."""

import argparse
import json
import math
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _nested(mapping, *path):
    value = mapping
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            break
    return value


def verify_loaded_checkpoint(
        checkpoint, expected_step, expected_target_lr=None,
        require_task1_a=False, expected_global_batch=None,
        allow_completed_transition=False, allow_active_transition=False):
    for key in ('state_dict', 'loops'):
        if not isinstance(checkpoint.get(key), dict) or not checkpoint[key]:
            raise AssertionError(f'Missing or empty checkpoint mapping: {key}')
    for key in ('optimizer_states', 'lr_schedulers'):
        if not isinstance(checkpoint.get(key), list) or not checkpoint[key]:
            raise AssertionError(f'Missing or empty checkpoint list: {key}')
    ema = checkpoint.get('ema')
    if not isinstance(ema, dict) or not ema.get('shadow_params'):
        raise AssertionError('Checkpoint EMA shadow parameters are missing.')
    global_step = int(checkpoint.get('global_step', -1))
    if global_step != int(expected_step):
        raise AssertionError(
            f'Expected global step {expected_step}, got {global_step}.')
    if int(checkpoint.get('hybrid_token_bias_step', -1)) != global_step:
        raise AssertionError('Bias warmup step does not match global step.')
    sampler = checkpoint.get('sampler')
    if (not isinstance(sampler, dict)
            or sampler.get('random_state') is None
            or 'counter' not in sampler):
        raise AssertionError('Checkpoint sampler resume state is incomplete.')
    optimizer_contract = checkpoint.get('task1_optimizer_contract')
    if not isinstance(optimizer_contract, dict):
        raise AssertionError('Task1 optimizer contract is missing.')
    if optimizer_contract.get('lr_warmup_optimizer_steps') != 2500:
        raise AssertionError('Checkpoint LR warmup contract is not 2500.')
    if optimizer_contract.get('gaussian_bias_warmup_optimizer_steps') != 5000:
        raise AssertionError('Checkpoint Gaussian-bias warmup is not 5000.')
    if optimizer_contract.get('global_optimizer_step') != global_step:
        raise AssertionError('Checkpoint optimizer contract step mismatch.')
    config = _nested(checkpoint, 'hyper_parameters', 'config')
    if expected_target_lr is not None:
        actual_target_lr = float(optimizer_contract['target_learning_rate'])
        if actual_target_lr != float(expected_target_lr):
            raise AssertionError(
                f'Expected target LR {expected_target_lr}, '
                f'got {actual_target_lr}.')
        actual_lrs = optimizer_contract.get('actual_learning_rates')
        if (not isinstance(actual_lrs, list) or not actual_lrs
                or any(float(value) != float(expected_target_lr)
                       for value in actual_lrs)):
            raise AssertionError(
                'Checkpoint optimizer groups do not all use the expected LR.')
        optimizer_group_lrs = [
            float(group.get('lr', float('nan')))
            for optimizer in checkpoint['optimizer_states']
            for group in optimizer.get('param_groups', [])]
        if (not optimizer_group_lrs
                or any(not math.isfinite(value) or value <= 0.0
                       for value in optimizer_group_lrs)):
            raise AssertionError(
                'Checkpoint optimizer param-group LR is missing, zero, or '
                'non-finite.')
        if any(value != float(expected_target_lr)
               for value in optimizer_group_lrs):
            raise AssertionError(
                'Checkpoint optimizer param-group LR does not match the '
                'declared source LR.')
        config_lr = _nested(config, 'optim', 'lr')
        if (config_lr is not None
                and float(config_lr) != float(expected_target_lr)
                and not allow_completed_transition):
            raise AssertionError(
                'Checkpoint config LR does not match optimizer state.')
    scheduler_name = optimizer_contract.get('scheduler')
    allowed_schedulers = {'constant_after_warmup'}
    if allow_completed_transition or allow_active_transition:
        allowed_schedulers.update({
            'linear_transition_then_constant',
            'cosine_transition_then_constant',
        })
    if scheduler_name not in allowed_schedulers:
        raise AssertionError(
            'Checkpoint scheduler is ended or incompatible with the '
            'continuation LR transition.')
    for scheduler_state in checkpoint['lr_schedulers']:
        if not isinstance(scheduler_state, dict):
            raise AssertionError('Checkpoint scheduler state is invalid.')
        last_epoch = scheduler_state.get('last_epoch')
        if last_epoch is None or abs(int(last_epoch) - global_step) > 1:
            raise AssertionError(
                'Checkpoint scheduler position does not match global step.')
    graft_history = checkpoint.get('task1_lr_graft_history')
    if not isinstance(graft_history, list):
        raise AssertionError('Task1 LR graft history is not serializable.')
    for record in graft_history:
        if not all(bool(record.get(name)) for name in (
                'optimizer_moments_preserved', 'ema_preserved',
                'global_step_preserved',
                'sampler_and_data_position_preserved')):
            raise AssertionError(
                'Checkpoint LR graft history reports a state reset.')
    if (allow_completed_transition and not allow_active_transition
            and scheduler_name != 'constant_after_warmup'):
        matching = [
            record for record in graft_history
            if record.get('transition_end_global_step') is not None]
        if not matching:
            raise AssertionError(
                'Completed LR transition checkpoint lacks graft history.')
        latest = matching[-1]
        if global_step < int(latest['transition_end_global_step']):
            raise AssertionError(
                'Checkpoint LR transition has not reached its endpoint.')

    if require_task1_a:
        expected = {
            ('algo', 'name'): 'langflow_flm_hybrid',
            ('algo', 'state_space'): 'vocab',
            ('algo', 'corruption'): 'flm_linear_gaussian',
            ('algo', 'model_time_condition'): 'tau',
            ('algo', 'self_condition_probability'): 0.25,
            ('algo', 'token_bias_schedule'): 'warmup',
            ('algo', 'classification_prototype_mode'): 'direct_vocab_state',
            ('algo', 'loss_type'): 'softmax_probability_mse',
            ('algo', 'prediction_target'): 'clean_token_one_hot',
            ('algo', 'output_transform'): 'softmax',
            ('algo', 'reduction'): 'vocab_sum_then_valid_token_mean',
            ('algo', 'optimization_scale'): 'one_half',
            ('model', 'length'): 128,
            ('model', 'hidden_size'): 768,
            ('model', 'n_blocks'): 12,
            ('model', 'tie_word_embeddings'): False,
            ('data', 'train'): 'openwebtext-packed-v1',
        }
        if config is None:
            raise AssertionError('Checkpoint resolved config is missing.')
        for path, wanted in expected.items():
            actual = _nested(config, *path)
            if actual != wanted:
                raise AssertionError(
                    f'Checkpoint identity mismatch for {".".join(path)}: '
                    f'expected {wanted!r}, got {actual!r}.')
        if checkpoint.get('hybrid_token_bias_schedule') != 'warmup':
            raise AssertionError(
                'Checkpoint matched-bias schedule identity is not warmup.')
        if global_step < 5000:
            raise AssertionError(
                'Task1 A continuation requires fully warmed matched bias.')
    if expected_global_batch is not None:
        actual_batch = _nested(config, 'loader', 'global_batch_size')
        if int(actual_batch) != int(expected_global_batch):
            raise AssertionError(
                f'Expected source global batch {expected_global_batch}, '
                f'got {actual_batch}.')

    resume_entry = (
        SOURCE_ROOT / 'scripts' / 'resume_task1_vocab_mse.sh'
    ).read_text(encoding='utf-8')
    generic_entry = (
        SOURCE_ROOT / 'scripts' / 'train_owt_128_langflow_hybrid.sh'
    ).read_text(encoding='utf-8')
    main_source = (SOURCE_ROOT / 'main.py').read_text(encoding='utf-8')
    for snippet, source in (
            ('RESUME_FROM_CKPT=true', resume_entry),
            ('RESUME_CKPT_PATH="$RESUME_CHECKPOINT_PATH"', resume_entry),
            ('RESUME_CONSTANT_LEARNING_RATE', resume_entry),
            ('checkpointing.resume_from_ckpt="$resume_from_ckpt"', generic_entry),
            ('trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)',
             main_source)):
        if snippet not in source:
            raise AssertionError(f'Resume entry is missing: {snippet}')

    return {
        'status': 'passed',
        'global_optimizer_step': global_step,
        'online_model_state_present': True,
        'ema_state_present': True,
        'optimizer_state_present': True,
        'lr_scheduler_state_present': True,
        'loop_state_present': True,
        'sampler_state_present': True,
        'optimizer_contract': optimizer_contract,
        'lr_graft_history_serializable': True,
        'resume_entry': 'scripts/resume_task1_vocab_mse.sh',
    }


def verify(
        checkpoint_path, expected_step, expected_target_lr=None,
        require_task1_a=False, expected_global_batch=None,
        allow_completed_transition=False, allow_active_transition=False):
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    result = verify_loaded_checkpoint(
        checkpoint, expected_step, expected_target_lr,
        require_task1_a=require_task1_a,
        expected_global_batch=expected_global_batch,
        allow_completed_transition=allow_completed_transition,
        allow_active_transition=allow_active_transition)
    result['checkpoint'] = str(checkpoint_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--expected-target-lr', type=float)
    parser.add_argument('--require-task1-a', action='store_true')
    parser.add_argument('--expected-global-batch', type=int)
    parser.add_argument('--allow-completed-transition', action='store_true')
    parser.add_argument('--allow-active-transition', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(
        args.checkpoint, args.expected_step, args.expected_target_lr,
        require_task1_a=args.require_task1_a,
        expected_global_batch=args.expected_global_batch,
        allow_completed_transition=args.allow_completed_transition,
        allow_active_transition=args.allow_active_transition)
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
