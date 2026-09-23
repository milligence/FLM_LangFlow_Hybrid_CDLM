"""Pure helpers for Task1 full-state continuation contracts."""

import math


GLOBAL_GROUP_COUNTS = {
    'v1_q30_frozen_global256': (8, 8, 32, 56, 80, 48, 24),
    'v1_m_tau25_global256': (6, 6, 24, 42, 92, 68, 18),
    # 15k no-SC probes. S1 moderately increases groups 0-2 versus the
    # stage-15k legacy quota. S2 preserves 7/8 of that quota and assigns
    # 1/8 of the global batch to uniform physical-t coverage.
    'no_sc_high_noise_global256': (24, 32, 56, 48, 48, 32, 16),
    'no_sc_legacy_uniform_physical_global256': (
        14, 28, 42, 49, 49, 28, 14, 32),
}


def global_group_counts(mode):
    try:
        counts = GLOBAL_GROUP_COUNTS[str(mode)]
    except KeyError as error:
        raise ValueError(f'Unsupported global quota mode: {mode}') from error
    if sum(counts) != 256:
        raise AssertionError('Global Task1 quota must contain 256 sequences.')
    return list(counts)


def linear_transition_lr(
        source_lr, target_lr, source_step, transition_steps, current_step):
    source_lr = float(source_lr)
    target_lr = float(target_lr)
    source_step = int(source_step)
    transition_steps = int(transition_steps)
    current_step = int(current_step)
    if source_lr <= 0.0 or target_lr <= 0.0:
        raise ValueError('Learning rates must be positive.')
    if target_lr > source_lr:
        raise ValueError('The continuation LR rule is down-only.')
    if transition_steps <= 0:
        raise ValueError('transition_steps must be positive.')
    progress = min(max(
        (current_step - source_step) / transition_steps, 0.0), 1.0)
    return source_lr + (target_lr - source_lr) * progress


def cosine_transition_lr(
        source_lr, target_lr, source_step, transition_steps, current_step):
    source_lr = float(source_lr)
    target_lr = float(target_lr)
    source_step = int(source_step)
    transition_steps = int(transition_steps)
    current_step = int(current_step)
    if source_lr <= 0.0 or target_lr <= 0.0:
        raise ValueError('Learning rates must be positive.')
    if target_lr > source_lr:
        raise ValueError('The continuation LR rule is down-only.')
    if transition_steps <= 0:
        raise ValueError('transition_steps must be positive.')
    progress = min(max(
        (current_step - source_step) / transition_steps, 0.0), 1.0)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return target_lr + (source_lr - target_lr) * multiplier
