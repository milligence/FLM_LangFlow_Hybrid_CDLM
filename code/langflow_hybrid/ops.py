"""Pure tensor operations used by the LangFlow-FLM hybrid."""

import math

import torch
import torch.nn.functional as F


def learned_gumbel_gamma(q, loc, scale, cutoff=1e-5):
    """Map low-discrepancy uniform quantiles to a differentiable Gumbel."""
    q = q.float().clamp(float(cutoff), 1.0 - float(cutoff))
    loc = torch.as_tensor(loc, device=q.device, dtype=torch.float32)
    scale = torch.as_tensor(scale, device=q.device, dtype=torch.float32)
    return loc - scale * torch.log(-torch.log(q))


def gumbel_entropy_curve(gamma, entropy, loc, scale):
    """LangFlow information curve whose derivative is Gumbel-shaped."""
    gamma = gamma.float()
    entropy = torch.as_tensor(
        entropy, device=gamma.device, dtype=torch.float32)
    loc = torch.as_tensor(loc, device=gamma.device, dtype=torch.float32)
    scale = torch.as_tensor(scale, device=gamma.device, dtype=torch.float32)
    return entropy * torch.exp(-torch.exp(-(gamma - loc) / scale))


def langflow_gumbel_gamma(q, loc=4.723, scale=0.852, cutoff=1e-5):
    """Map uniform quantiles to the frozen LangFlow Gumbel gamma path."""
    return learned_gumbel_gamma(q, loc, scale, cutoff)


def langflow_alpha_sigma(gamma):
    """Return VP alpha and sigma with alpha**2 + sigma**2 == 1."""
    alpha = torch.sigmoid(-gamma.float()).sqrt()
    sigma = torch.sigmoid(gamma.float()).sqrt()
    return alpha, sigma


def langflow_corrupt_embedding(clean_embedding, gamma, noise):
    """Apply the VP corruption path to an embedding-state tensor."""
    if clean_embedding.shape != noise.shape:
        raise ValueError('clean_embedding and noise must have matching shapes.')
    alpha, sigma = langflow_alpha_sigma(gamma)
    alpha = alpha[:, None, None].to(clean_embedding.dtype)
    sigma = sigma[:, None, None].to(clean_embedding.dtype)
    return alpha * clean_embedding + sigma * noise


def flm_linear_gamma(t, eps=1e-5):
    """Encode the FLM linear path ``alpha=t, sigma=1-t`` as log-SNR gamma."""
    t = t.float().clamp(float(eps), 1.0 - float(eps))
    return 2.0 * (torch.log1p(-t) - torch.log(t))


def flm_model_time_condition(u, t, coordinate, eps=1e-5):
    """Return the sole A/C model difference for the shared FLM path."""
    if coordinate == 'tau':
        return u.float()
    if coordinate == 'log_nsr':
        return flm_linear_gamma(t, eps)
    raise ValueError(
        'Task1 model_time_condition must be tau or log_nsr, got '
        f'{coordinate!r}.')


def flm_linear_alpha_sigma(gamma):
    """Recover the unnormalised FLM linear interpolation coefficients."""
    half_gamma = gamma.float() / 2.0
    return torch.sigmoid(-half_gamma), torch.sigmoid(half_gamma)


def flm_corrupt_vocab(
        clean_tokens, t, vocab_size, noise=None, *, return_target=True,
        return_noise=True):
    """Apply FLM Gaussian interpolation in one-hot vocabulary space.

    The training path does not need the dense one-hot target after corruption.
    ``return_target=False`` therefore forms ``t * one_hot + (1-t) * noise``
    with one indexed add and skips that vocabulary-sized allocation.  When the
    random noise is internal and ``return_noise=False``, its buffer becomes the
    noisy state in place instead of retaining a second vocabulary-sized tensor.
    The defaults preserve the helper's existing three-tensor API for callers
    that inspect either source explicitly.
    """
    if clean_tokens.ndim != 2:
        raise ValueError(
            'clean_tokens must have shape [B, L], got '
            f'{tuple(clean_tokens.shape)}.')
    if t.ndim != 1 or t.shape[0] != clean_tokens.shape[0]:
        raise ValueError(
            't must have shape [B] matching clean_tokens, got '
            f'{tuple(t.shape)}.')
    state_shape = (*clean_tokens.shape, int(vocab_size))
    sampled_internally = noise is None
    if sampled_internally:
        noise = torch.randn(
            state_shape, device=clean_tokens.device, dtype=torch.float32)
    if noise.shape != state_shape:
        raise ValueError(
            'Vocabulary noise must match the one-hot target shape, got '
            f'{tuple(noise.shape)} and {state_shape}.')
    coefficient = t.float()[:, None, None]
    if sampled_internally and not return_noise:
        state = noise
        state.mul_(1.0 - coefficient)
        noise = None
    else:
        state = (1.0 - coefficient) * noise.float()
    state.scatter_add_(
        -1,
        clean_tokens.unsqueeze(-1),
        coefficient.expand(-1, clean_tokens.shape[1], 1))
    target = None
    if return_target:
        target = torch.zeros(
            state_shape, device=clean_tokens.device, dtype=torch.float32)
        target.scatter_(-1, clean_tokens.unsqueeze(-1), 1.0)
    return state, target, noise if return_noise else None


def flm_vocab_gaussian_bias(state, t, weight=1.0, eps=1e-5):
    """Matched class likelihood for ``N(t e_k, (1-t)^2 I)``.

    This operates directly on the physical vocabulary state.  It deliberately
    does not use the learned input projection or any classifier/codebook.
    """
    if state.ndim != 3:
        raise ValueError(
            'Vocabulary state must have shape [B, L, V], got '
            f'{tuple(state.shape)}.')
    if t.ndim != 1 or t.shape[0] != state.shape[0]:
        raise ValueError(
            't must have shape [B] matching state, got '
            f'{tuple(t.shape)}.')
    safe_t = t.float().clamp(0.0, 1.0 - float(eps))
    coefficient = (
        float(weight) * safe_t
        / (1.0 - safe_t).square().clamp_min(float(eps) ** 2))
    return coefficient[:, None, None].to(state.dtype) * state


def flm_linear_euler_update(state, predicted_clean, t_current, t_next,
                            eps=1e-5):
    """Euler step for the FLM linear Gaussian-to-data probability flow."""
    if state.shape != predicted_clean.shape:
        raise ValueError('state and predicted_clean must have matching shapes.')
    current = torch.as_tensor(
        t_current, device=state.device, dtype=torch.float32)
    following = torch.as_tensor(
        t_next, device=state.device, dtype=torch.float32)
    while current.ndim < state.ndim:
        current = current.unsqueeze(-1)
        following = following.unsqueeze(-1)
    velocity = ((predicted_clean.float() - state.float())
                / (1.0 - current).clamp_min(float(eps)))
    return (state.float() + (following - current) * velocity).to(state.dtype)


def detached_self_conditioning_embedding(probabilities, embedding_layer):
    """Convert a posterior to its expected embedding without target gradients."""
    return embedding_layer(probabilities).detach()


def finite_sc_gate(delta, d50):
    """Continuous finite-jump SC gate with ``b(d50) == 0.5``."""
    d50 = float(d50)
    if not math.isfinite(d50) or d50 <= 0.0:
        raise ValueError(f'd50 must be finite and positive, got {d50!r}.')
    delta = torch.as_tensor(delta).float()
    if bool(torch.any(delta < 0.0)):
        raise ValueError('finite SC gaps must be non-negative.')
    alpha = math.log(2.0) / d50
    return -torch.expm1(-alpha * delta)


def finite_sc_update(carried_state, proposal, delta, d50):
    """Mix an SC proposal into the carried state using float32 arithmetic."""
    if carried_state.shape != proposal.shape:
        raise ValueError(
            'SC carried state and proposal must have matching shapes, got '
            f'{tuple(carried_state.shape)} and {tuple(proposal.shape)}.')
    gate = finite_sc_gate(delta, d50).to(carried_state.device)
    while gate.ndim < carried_state.ndim:
        gate = gate.unsqueeze(-1)
    carried = carried_state.float()
    proposal = proposal.float()
    return carried + gate * (proposal - carried)


def candidate_bias_codebook(codebook, gradient_mode):
    """Select the Gaussian-bias codebook without changing its forward value."""
    if gradient_mode == 'detach_candidate':
        return codebook.detach()
    if gradient_mode in {'all', 'frozen'}:
        return codebook
    raise ValueError(f'Unsupported codebook_gradient_mode={gradient_mode!r}.')


def probability_prediction_loss(logits, target_tokens, loss_type):
    """Compute only the per-token objective required by the optimizer."""
    float_logits = logits.float()
    if loss_type == 'cross_entropy':
        target_logit = float_logits.gather(
            -1, target_tokens.unsqueeze(-1)).squeeze(-1)
        return torch.logsumexp(float_logits, dim=-1) - target_logit
    if loss_type == 'softmax_probability_mse':
        probabilities = F.softmax(float_logits, dim=-1)
        target_probability = probabilities.gather(
            -1, target_tokens.unsqueeze(-1)).squeeze(-1)
        return (
            probabilities.square().sum(dim=-1)
            + 1.0 - 2.0 * target_probability)
    raise ValueError(f'Unsupported probability loss {loss_type!r}.')


def langflow_token_bias_weight(step, warmup_steps=5000):
    """Linear token-bias warmup, expressed independently for unit tests."""
    if warmup_steps <= 0:
        return 1.0
    return min(max(float(step) / float(warmup_steps), 0.0), 1.0)


def langflow_euler_edm_update(z_gamma, predicted_clean_embedding,
                              gamma_current, gamma_next):
    """Apply one Euler-EDM update used by the released LangFlow sampler."""
    current = z_gamma.float() * torch.exp(
        (F.softplus(gamma_current.float())
         - F.softplus(gamma_next.float())) / 2.0)
    endpoint = (torch.sigmoid(-gamma_next.float()).sqrt()
                * predicted_clean_embedding.float())
    mix = torch.exp((gamma_next.float() - gamma_current.float()) / 2.0)
    updated = torch.lerp(endpoint, current, mix)
    return updated.to(z_gamma.dtype)
