"""Pure functions for optimizer-step clocks, weights and gradient calibration."""
from __future__ import annotations
import math
from statistics import median
from typing import Sequence


def linear_knots(k: float, knots: Sequence[Sequence[float]]) -> float:
    if k <= knots[0][0]:
        return float(knots[0][1])
    for (a, va), (b, vb) in zip(knots, knots[1:]):
        if a <= k < b:
            return float(va + (vb - va) * (k - a) / (b - a))
    return float(knots[-1][1])


def stage_at(stages: list[dict], k: int) -> dict:
    for stage in stages:
        if stage['start_step'] <= k < stage['end_step']:
            return stage
    raise ValueError(f"No stage for completed_updates={k}; do not train beyond 50000")


def rho_map(sampler: dict, k: int) -> float:
    s = stage_at(sampler['map']['stages'], k)
    q = (k - s['start_step']) / (s['end_step'] - s['start_step'])
    return s['rho_map_start'] + q * (s['rho_map_end'] - s['rho_map_start'])


def hard_weight(sampler: dict, k: int) -> float:
    h = sampler['map']['hard_weight']
    if k < h['start_step']:
        return 1.0  # No hard class may exist before this step.
    return linear_knots(k, [[h['start_step'],h['start_value']],[h['end_step'],h['end_value']]])


def learning_rates(train: dict, sampler: dict, k: int) -> dict[str,float]:
    # First actual optimizer update uses a small positive learning rate.
    base = linear_knots(k + 1, train['optimizer']['lr_schedule'])
    gate = min(1.0, rho_map(sampler,k) / train['objective']['rho_map_final'])
    result = {'legacy':base, 'finite_G':base * gate}
    if train['line'] == 'F':
        result['finite_B'] = base * .1 * gate
    return result


def ema_decay(j: int, max_decay: float) -> float:
    if j < 1:
        raise ValueError("EMA update expects j>=1 completed updates")
    return min(max_decay, 1 - 1/(j+1))


def calibrated_scale(local_norms: Sequence[float], raw_map_norms: Sequence[float]) -> float:
    if len(local_norms) != len(raw_map_norms) or not local_norms:
        raise ValueError("Nonempty matched probe norms required")
    ratios=[]
    for a,b in zip(local_norms,raw_map_norms):
        if not (math.isfinite(a) and math.isfinite(b)) or min(a,b)<0:
            raise FloatingPointError("Invalid calibration norm")
        ratios.append(a/max(b,.1*a,1e-12))
    return min(10.,median(ratios))


def safety_recalibrate(c: float, rho: float, local_norm: float, weighted_map_norm: float,
                       factor: float=1.5) -> float:
    if not all(math.isfinite(x) and x>=0 for x in (c,rho,local_norm,weighted_map_norm)):
        raise FloatingPointError("Invalid safety-controller state")
    if rho == 0 or weighted_map_norm == 0:
        return c
    # rho cancels; the hard-class weights remain in weighted_map_norm.
    return min(c, factor * local_norm / max(weighted_map_norm,1e-12))
