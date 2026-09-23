import copy

import torch
from omegaconf import OmegaConf

from models.dit import DIT
from task1_posterior_tvm import posterior_time_features
from task1_tvm_ce import (
    calibrated_relative_distributions,
    tvm_quantity_from_logits_jvp,
)


def test_bmm_jvp_attention_matches_reference_softcap():
    from models.dit import DDiTBlock

    torch.manual_seed(20260921)
    block = DDiTBlock(
        dim=16, n_heads=2, adaLN=False, dropout=0.0,
        attention_jvp_backend='reference')
    q = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    k = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 7, 8, dtype=torch.float64)
    reference = block.custom_sdpa(q, k, v, softcap=50.0)
    candidate = block.bmm_sdpa(q, k, v, softcap=50.0)
    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)


def _config(finite_sc):
    return OmegaConf.create({
        'model': {
            'hidden_size': 32,
            'cond_dim': 16,
            'n_blocks': 1,
            'n_heads': 4,
            'dropout': 0.0,
            'qk_norm': True,
            'scale_by_sigma': True,
            'tie_word_embeddings': False,
        },
        'algo': {
            'causal_attention': False,
            'embedding_state': False,
            'self_conditioning': False,
            'codebook_gradient_mode': 'all',
            'classification_prototype_mode': 'direct_vocab_state',
            'double_temb': False,
            'learnable_loss_weighting': False,
            'posterior_tvm_time_conditioning': True,
            'finite_map_only_sc': finite_sc,
            'finite_map_sc_hidden': 12,
        },
        'is_di4c': False,
        'is_di4c_deterministic': True,
    })


def _forward(model, state, r, eta, cache=None, valid=None):
    embedded = model.vocab_embed(state)
    return model(
        embedded, r, inputs_are_embeddings=True, use_jvp_attn=True,
        posterior_time_features=posterior_time_features(r, eta),
        finite_sc_cache=cache,
        finite_sc_previous_eta=torch.full_like(r, 0.3),
        finite_sc_valid=valid,
        finite_sc_eta=eta)


def test_posterior_tvm_math_jvp_pairing_and_strict_local_sc_invariance():
    torch.manual_seed(123)
    model_a = DIT(_config(False), vocab_size=13)
    torch.manual_seed(123)
    model_b = DIT(_config(True), vocab_size=13)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    common_b = {
        key: value for key, value in state_b.items()
        if not key.startswith('finite_map_sc_projector.')}
    assert state_a.keys() == common_b.keys()
    assert all(torch.equal(state_a[key], common_b[key]) for key in state_a)

    with torch.no_grad():
        model_b.output_layer.linear.weight.normal_(0.0, 0.1)
        model_b.finite_map_sc_projector.projection[-1].weight.normal_(0.0, 0.1)
        model_b.finite_map_sc_projector.projection[-1].bias.normal_(0.0, 0.1)
    state = torch.randn(2, 5, 13)
    r = torch.tensor([0.2, 0.7])
    eta_zero = torch.zeros(2)
    random_cache = torch.randn(2, 5, 32)
    valid = torch.ones(2)
    local_warm = _forward(
        model_b, state, r, eta_zero, random_cache, valid)
    local_cold = _forward(
        model_b, state, r, eta_zero, torch.zeros_like(random_cache),
        torch.zeros_like(valid))
    torch.testing.assert_close(local_warm, local_cold, rtol=0.0, atol=0.0)

    eta = torch.tensor([0.25, 0.6], requires_grad=True)
    logits, tangent = torch.func.jvp(
        lambda value: _forward(
            model_b, state, r, value, random_cache, valid),
        (eta,), (eta * (1.0 - eta),))
    tangent.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model_b.posterior_tvm_time_conditioner.parameters())

    toy_eta = torch.tensor([0.17, 0.63], dtype=torch.float64)
    toy_direction = toy_eta * (1.0 - toy_eta)
    toy_logits = lambda value: torch.stack(
        (torch.sin(value), value.square(), torch.exp(-value)), dim=-1)
    _, toy_jvp = torch.func.jvp(
        toy_logits, (toy_eta,), (toy_direction,))
    delta = 1e-6
    toy_finite_difference = (
        toy_logits(toy_eta + delta * toy_direction)
        - toy_logits(toy_eta - delta * toy_direction)) / (2.0 * delta)
    relative_error = (
        (toy_jvp - toy_finite_difference).abs()
        / toy_finite_difference.abs().clamp_min(1e-12)).max()
    assert float(relative_error) < 1e-4

    probability, _, gate, quantity = tvm_quantity_from_logits_jvp(
        logits, tangent)
    torch.testing.assert_close(
        quantity.sum(dim=-1), torch.ones_like(quantity[..., 0]),
        rtol=1e-5, atol=1e-5)

    fixed_logits = torch.randn(2, 3, 13)
    fixed_probability = torch.softmax(fixed_logits, dim=-1)
    fixed_gate = torch.ones_like(fixed_probability)
    student_log, teacher = calibrated_relative_distributions(
        fixed_probability, fixed_gate, fixed_probability, 0.02)
    kl = (
        teacher * (teacher.clamp_min(1e-30).log() - student_log)
    ).sum(dim=-1)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0.0)
    assert not teacher.requires_grad
