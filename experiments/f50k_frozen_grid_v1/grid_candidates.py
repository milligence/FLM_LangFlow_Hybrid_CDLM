"""Fixed, inference-only candidate generator from the F50K work order."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

T = 0.95
MIN_GAP = 0.025
B = (0.0, 0.2375, 0.475, 0.7125, T)
A = (0.0, 0.3392770637, 0.5814685447, 0.7246687714, T)
C = (0.0, 0.3392770637, 0.5814685447, 0.85, T)
D = (0.0, 0.5271291955, 0.7763932023, 0.8942628737, T)


def key(grid: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(float(t), 10) for t in grid)


def valid(grid: Sequence[float], nfe: int) -> bool:
    return (len(grid) == nfe + 1
            and all(math.isfinite(float(t)) for t in grid)
            and abs(grid[0]) < 1e-12 and abs(grid[-1] - T) < 1e-12
            and all(b - a >= MIN_GAP - 1e-12 for a, b in zip(grid, grid[1:])))


def unique(grids: Iterable[Sequence[float]], nfe: int) -> list[tuple[float, ...]]:
    out, seen = [], set()
    for grid in grids:
        g = tuple(float(t) for t in grid)
        if valid(g, nfe) and key(g) not in seen:
            out.append(g)
            seen.add(key(g))
    return out


def radical_inverse(index: int, base: int) -> float:
    if index < 1 or base < 2:
        raise ValueError('index>=1 and base>=2 required')
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def coarse_four() -> list[tuple[float, ...]]:
    out = unique([B, A, C, D], 4)
    for j in (1, 2, 3):
        for delta in (-.08, -.04, .04, .08):
            g = list(B)
            g[j] += delta
            out = unique([*out, g], 4)
    if len(out) != 16:
        raise AssertionError('Expected 4 controls + 12 local perturbations')
    qlo, qhi = -math.log1p(-.05), -math.log1p(-.90)
    for mode in ('physical', 'log_remaining_noise'):
        count, index = 0, 17
        while count < 12:
            u = sorted(radical_inverse(index, base) for base in (2, 3, 5))
            index += 1
            nodes = ([.05 + .85 * v for v in u] if mode == 'physical'
                     else [-math.expm1(-(qlo + (qhi - qlo) * v)) for v in u])
            new = unique([*out, (0., *nodes, T)], 4)
            if len(new) > len(out):
                out = new
                count += 1
            if index > 100000:
                raise RuntimeError('Fixed candidate quota not filled')
    assert len(out) == 40
    return out


def coarse_two(baseline: Sequence[float]) -> list[tuple[float, ...]]:
    if not valid(baseline, 2):
        raise ValueError('Resolve actual finite_two grid first')
    mids = (.10, .20, .30, .40, .475, .5271291955, .5814685447,
            .65, .7125, .80, .90)
    out = unique([baseline, *((0., t, T) for t in mids)], 2)
    assert len(out) <= 12
    return out


def refine_four(anchors: Iterable[Sequence[float]]) -> list[tuple[float, ...]]:
    out = []
    for anchor in unique(anchors, 4):
        for j in (1, 2, 3):
            for delta in (-.02, .02):
                g = list(anchor)
                g[j] += delta
                out.append(g)
    return unique(out, 4)


def refine_two(anchors: Iterable[Sequence[float]]) -> list[tuple[float, ...]]:
    return unique([(0., a[1] + delta, T) for a in unique(anchors, 2)
                   for delta in (-.05, -.025, .025, .05)], 2)


def perturb_for_robustness(grid: Sequence[float]) -> list[tuple[float, ...]]:
    nfe, out = len(grid) - 1, []
    for j in range(1, nfe):
        for delta in (-.01, .01):
            g = list(grid)
            g[j] += delta
            out.append(g)
    return unique(out, nfe)
