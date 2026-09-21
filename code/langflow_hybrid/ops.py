"""Pure tensor operations used by the LangFlow-FLM hybrid."""

import torch
import torch.nn.functional as F


def langflow_gumbel_gamma(q, loc=4.723, scale=0.852, cutoff=1e-5):
    """Map uniform quantiles to the frozen LangFlow Gumbel gamma path."""
    q = q.float().clamp(float(cutoff), 1.0 - float(cutoff))
    return float(loc) - float(scale) * torch.log(-torch.log(q))


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


def candidate_bias_codebook(codebook, gradient_mode):
    """Select the Gaussian-bias codebook without changing its forward value."""
    if gradient_mode == 'detach_candidate':
        return codebook.detach()
    if gradient_mode in {'all', 'frozen'}:
        return codebook
    raise ValueError(f'Unsupported codebook_gradient_mode={gradient_mode!r}.')


def probability_prediction_metrics(
        logits, target_tokens, include_distribution_stats=True):
    """Return the registered per-token probability metrics.

    The Brier implementation is algebraically identical to constructing a
    vocabulary-sized one-hot target, but avoids that extra allocation.
    """
    float_logits = logits.float()
    target_logit = float_logits.gather(
        -1, target_tokens.unsqueeze(-1)).squeeze(-1)
    target_log_prob = target_logit - torch.logsumexp(float_logits, dim=-1)
    probs = F.softmax(float_logits, dim=-1)
    target_prob = probs.gather(
        -1, target_tokens.unsqueeze(-1)).squeeze(-1)
    probability_square_sum = probs.square().sum(dim=-1)
    raw_brier = probability_square_sum + 1.0 - 2.0 * target_prob
    max_probability, predicted_tokens = probs.max(dim=-1)
    top_values, top_indices = probs.topk(k=2, dim=-1)
    best_other = torch.where(
        top_indices[..., 0] == target_tokens,
        top_values[..., 1],
        top_values[..., 0])
    result = {
        'token_ce': -target_log_prob,
        'raw_brier': raw_brier,
        'target_probability': target_prob,
        'target_margin': target_prob - best_other,
        'top1_accuracy': (predicted_tokens == target_tokens).float(),
        'max_probability': max_probability,
        'high_confidence_wrong': (
            (predicted_tokens != target_tokens) & (max_probability >= 0.9)
        ).float(),
    }
    if include_distribution_stats:
        top100 = probs.topk(k=min(100, probs.shape[-1]), dim=-1).indices
        top10_count = min(10, top100.shape[-1])
        mse_target_gradient = (
            probs.shape[-1] * target_prob
            * (2.0 * target_prob - 1.0 - probability_square_sum))
        ce_target_gradient = 1.0 - target_prob
        result['posterior_entropy'] = -(
            probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        ).sum(dim=-1)
        result.update({
            'top10_true_inclusion': (
                top100[..., :top10_count] == target_tokens.unsqueeze(-1)
            ).any(dim=-1).float(),
            'top100_true_inclusion': (
                top100 == target_tokens.unsqueeze(-1)
            ).any(dim=-1).float(),
            'probability_square_sum': probability_square_sum,
            'abs_mse_target_logit_gradient': mse_target_gradient.abs(),
            'ce_target_gradient_magnitude': ce_target_gradient,
            'target_gradient_ratio': (
                mse_target_gradient.abs() / (ce_target_gradient + 1e-12)),
            'target_probability_lt_1e_6': (target_prob < 1e-6).float(),
            'target_probability_lt_1e_5': (target_prob < 1e-5).float(),
            'target_probability_lt_1e_4': (target_prob < 1e-4).float(),
            'target_probability_lt_uniform': (
                target_prob < (1.0 / probs.shape[-1])).float(),
        })
    return result


def langflow_token_bias_weight(step, warmup_steps=5000):
    """Linear token-bias warmup, expressed independently for unit tests."""
    if warmup_steps <= 0:
        return 1.0
    return min(max(float(step) / float(warmup_steps), 0.0), 1.0)


def langflow_euler_edm_update(z_gamma, predicted_clean_embedding,
                              gamma_current, gamma_next):
    """Apply one Euler-EDM update used by the released LangFlow sampler."""
    current = z_gamma.double() * torch.exp(
        (F.softplus(gamma_current.double())
         - F.softplus(gamma_next.double())) / 2.0)
    endpoint = (torch.sigmoid(-gamma_next.double()).sqrt()
                * predicted_clean_embedding.double())
    mix = torch.exp((gamma_next.double() - gamma_current.double()) / 2.0)
    updated = torch.lerp(endpoint, current, mix)
    return updated.to(z_gamma.dtype)
