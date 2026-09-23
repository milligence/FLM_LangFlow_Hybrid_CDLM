"""Exact mathematical reference for the contracted losses.

Preserve input dtype (FP64 unit tests; FP32 production posterior arithmetic).
The repo adapter supplies logits and connected d(logits)/d(eta). These routines
must NOT be called inside no_grad for the student branch.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def _check_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}; no nan_to_num recovery allowed")


def eta_view(eta: Tensor, like: Tensor) -> Tensor:
    if eta.ndim == 1:
        eta = eta[:, None, None]
    if eta.shape != (like.shape[0], 1, 1):
        raise ValueError("eta must be [B] or [B,1,1], one time per sequence")
    return eta


def sequence_mean(token_loss: Tensor, weights: Tensor | None = None) -> Tensor:
    """token_loss=[B,L]. Divide weighted sequence sum by B, NEVER sum(weights)."""
    if token_loss.ndim != 2:
        raise ValueError("Expected [batch,length] token loss")
    per_seq = token_loss.mean(dim=1)
    if weights is not None:
        if weights.shape != per_seq.shape:
            raise ValueError("Weights must be [B]")
        per_seq = per_seq * weights
    return per_seq.mean()


def probability_mse(logits: Tensor, y: Tensor) -> Tensor:
    p = logits.softmax(-1)
    # Algebraically equivalent to 1/2 sum((p-onehot)^2), no giant onehot required.
    loss = 0.5 * (p.square().sum(-1) - 2 * p.gather(-1, y[..., None]).squeeze(-1) + 1)
    return sequence_mean(loss)


def softmax_eta_jvp(logits: Tensor, dlogits: Tensor) -> tuple[Tensor, Tensor]:
    p = logits.softmax(-1)
    return p, p * (dlogits - (p * dlogits).sum(-1, keepdim=True))


def f_outputs(logits: Tensor, dlogits: Tensor, b_raw: Tensor, db_raw: Tensor,
              eta: Tensor) -> tuple[Tensor, Tensor]:
    eta = eta_view(eta, logits)
    p, dp = softmax_eta_jvp(logits, dlogits)
    b = b_raw - b_raw.mean(-1, keepdim=True)
    db = db_raw - db_raw.mean(-1, keepdim=True)
    return p + eta * b, dp + b + eta * db


def f_residual(a: Tensor, da: Tensor, eta: Tensor, teacher_p: Tensor) -> Tensor:
    eta = eta_view(eta, a)
    out = ((1 - eta) * a + eta * (1 - eta) * da
           + eta * a.detach() - teacher_p.detach())
    _check_finite("F residual", out)
    return out


def f_map_loss(a: Tensor, da: Tensor, eta: Tensor, teacher_p: Tensor,
               weights: Tensor | None = None) -> Tensor:
    residual = f_residual(a, da, eta, teacher_p)
    return sequence_mean(0.5 * residual.square().sum(-1), weights)


def log_softplus(z: Tensor) -> Tensor:
    """Stable log(softplus(z)); asymptotic branches differ below dtype tolerance.

    This is NOT a probability floor or a change in the calibration fixed point.
    Masked indexing prevents an unused branch from constructing overflow/NaN.
    """
    _check_finite("log_softplus input", z)
    out = torch.empty_like(z)
    low, high = z < -40, z > 40
    mid = ~(low | high)
    out[low] = z[low]  # relative neglected correction <= exp(-40)/2
    out[mid] = F.softplus(z[mid], threshold=50).log()
    # softplus(z)=z+log1p(exp(-z)); retain the tiny correction for FP64 tests.
    out[high] = z[high].log() + torch.log1p(torch.log1p(torch.exp(-z[high])) / z[high])
    return out


def teacher_log_calibration(log_r: Tensor, log_teacher: Tensor, kappa: float) -> Tensor:
    """No ratios of underflowed probabilities. Entire target is detached.

    a=log(pT/R/kappa).  For a>log(40),
    log(kappa*softplus(exp(a))) = log(kappa)+a to machine precision.
    This avoids ever constructing pT/R or exp(a) in its unsafe region.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    with torch.no_grad():
        lr, lt = log_r.detach(), log_teacher.detach()
        _check_finite("teacher log probabilities", lt)
        _check_finite("student log probabilities", lr)
        a = lt - lr - math.log(kappa)
        high = a > math.log(40)
        h = torch.empty_like(a)
        h[high] = math.log(kappa) + a[high]
        # exp(a) is <=40 here, softplus(exp(a)) >= log(2).
        h[~high] = math.log(kappa) + F.softplus(a[~high].exp(), threshold=50).log()
        log_num = lr + h
        result = log_num - log_num.logsumexp(-1, keepdim=True)
        _check_finite("calibrated teacher", result)
        return result


def p_map_terms(logits: Tensor, dlogits: Tensor, eta: Tensor, teacher_logits: Tensor,
                kappa: float = .02, negative_mass_weight: float = .1,
                weights: Tensor | None = None) -> dict[str, Tensor]:
    eta = eta_view(eta, logits)
    _check_finite("student logits", logits)
    _check_finite("student logit tangent", dlogits)
    log_r = logits.log_softmax(-1)
    r = log_r.exp()
    v = eta * (1 - eta) * dlogits
    mu = (r * v).sum(-1, keepdim=True)
    g = 1 + v - mu
    q_raw = r * g
    log_num_s = log_r + math.log(kappa) + log_softplus(g / kappa)
    log_qs = log_num_s - log_num_s.logsumexp(-1, keepdim=True)
    log_pt = teacher_logits.detach().log_softmax(-1)
    log_qt = teacher_log_calibration(log_r, log_pt, kappa)
    qt = log_qt.exp()
    ce_token = -(qt * log_qs).sum(-1)
    kl_token = (qt * (log_qt - log_qs)).sum(-1)
    negative_mass = (-q_raw).clamp_min(0).sum(-1)  # Only the specified ReLU(-Q).
    total_token = ce_token + negative_mass_weight * negative_mass.square()
    _check_finite("P token loss", total_token)
    return {
        "loss": sequence_mean(total_token, weights),
        "ce": sequence_mean(ce_token, weights),
        "kl_unweighted": kl_token.mean(),
        "q_raw": q_raw, "negative_mass": negative_mass,
        "log_qs": log_qs, "log_qt": log_qt,
        "r": r, "teacher_p": log_pt.exp(),
        "raw_residual": q_raw - log_pt.exp(),
    }
