"""Matched-sequence statistics for the F 30k-50k decision gates."""
from __future__ import annotations

import numpy as np


def _vector(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain at least two finite values.')
    return array


def paired_nll_difference(new_nll_sums, new_token_counts, old_nll_sums,
                          old_token_counts, *, repeats=2000, seed=736091):
    """Return new-minus-old NLL and a sequence-paired bootstrap interval."""
    new = _vector(new_nll_sums, 'new_nll_sums')
    new_counts = _vector(new_token_counts, 'new_token_counts')
    old = _vector(old_nll_sums, 'old_nll_sums')
    old_counts = _vector(old_token_counts, 'old_token_counts')
    if not (new.shape == new_counts.shape == old.shape == old_counts.shape):
        raise ValueError('All paired arrays must have identical shapes.')
    if ((new_counts <= 0).any() or (old_counts <= 0).any()
            or (new < 0).any() or (old < 0).any()):
        raise ValueError('Token counts must be positive and NLL sums nonnegative.')
    if not isinstance(repeats, int) or repeats < 100:
        raise ValueError('At least 100 bootstrap replicates are required.')
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(new), size=(repeats, len(new)))
    boot = (new[indices].sum(1) / new_counts[indices].sum(1)
            - old[indices].sum(1) / old_counts[indices].sum(1))
    point = new.sum() / new_counts.sum() - old.sum() / old_counts.sum()
    low, high = np.quantile(boot, [.025, .975])
    return {
        'delta_nll': float(point), 'ppl_ratio': float(np.exp(point)),
        'ci_low': float(low), 'ci_high': float(high),
        'bootstrap_repeats': repeats,
    }


def paired_ratio(new_values, old_values, *, repeats=2000, seed=736092):
    new = _vector(new_values, 'new_values')
    old = _vector(old_values, 'old_values')
    if new.shape != old.shape or (new < 0).any() or (old <= 0).any():
        raise ValueError('Ratios require matched nonnegative new values and positive old values.')
    if not isinstance(repeats, int) or repeats < 100:
        raise ValueError('At least 100 bootstrap replicates are required.')
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(new), size=(repeats, len(new)))
    boot = new[indices].mean(1) / old[indices].mean(1)
    low, high = np.quantile(boot, [.025, .975])
    return {
        'ratio': float(new.mean() / old.mean()),
        'ci_low': float(low), 'ci_high': float(high),
    }
