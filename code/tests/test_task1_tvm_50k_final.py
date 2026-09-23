"""Focused correctness tests for the final F/P repository binding."""

import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import yaml
from omegaconf import OmegaConf

from models.dit import DIT
from task1_tvm_50k_final import Task1TVM50KFinal
from task1_tvm_50k_final import (
    _f_map_tokens,
    _p_map_tokens,
    _softmax_jvp,
)


def _config(line='F', finite=True):
    return OmegaConf.create({
        'model': {
            'hidden_size': 32, 'cond_dim': 16, 'n_blocks': 1,
            'n_heads': 4, 'dropout': 0.0, 'qk_norm': False,
            'scale_by_sigma': True, 'tie_word_embeddings': False,
        },
        'algo': {
            'causal_attention': False, 'embedding_state': False,
            'self_conditioning': True, 'codebook_gradient_mode': 'all',
            'classification_prototype_mode': 'direct_vocab_state',
            'double_temb': False, 'learnable_loss_weighting': False,
            'finite_map_only_sc': False,
            'task1_finite_time_conditioning': finite,
            'task1_finite_init_seed': 17 if line == 'F' else 23,
            'task1_tvm_final_line': line,
        },
        'is_di4c': False, 'is_di4c_deterministic': True,
    })


def _activate(model):
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation.weight.normal_(0.0, 0.05)
            block.adaLN_modulation.bias.normal_(0.0, 0.05)
        model.output_layer.adaLN_modulation.weight.normal_(0.0, 0.05)
        model.output_layer.adaLN_modulation.bias.normal_(0.0, 0.05)
        model.output_layer.linear.weight.normal_(0.0, 0.05)
        model.finite_time_conditioner.output.weight.normal_(0.0, 0.05)
        model.finite_time_conditioner.output.bias.normal_(0.0, 0.05)
        if model.finite_correction_head is not None:
            model.finite_correction_head.weight.normal_(0.0, 0.05)
            model.finite_correction_head.bias.normal_(0.0, 0.05)


def _activate_finite_only(model):
    """Perturb only G/B so diagonal parity keeps shared weights identical."""
    with torch.no_grad():
        model.finite_time_conditioner.output.weight.normal_(0.0, 0.05)
        model.finite_time_conditioner.output.bias.normal_(0.0, 0.05)
        if model.finite_correction_head is not None:
            model.finite_correction_head.weight.normal_(0.0, 0.05)
            model.finite_correction_head.bias.normal_(0.0, 0.05)


def test_f_and_p_share_byte_identical_legacy_initialization():
    torch.manual_seed(20260921)
    model_f = DIT(_config('F'), vocab_size=19)
    torch.manual_seed(20260921)
    model_p = DIT(_config('P'), vocab_size=19)
    common_f = {
        name: value for name, value in model_f.state_dict().items()
        if not name.startswith(('finite_time_conditioner.',
                                'finite_correction_head.'))}
    common_p = {
        name: value for name, value in model_p.state_dict().items()
        if not name.startswith('finite_time_conditioner.')}
    assert common_f.keys() == common_p.keys()
    assert all(torch.equal(common_f[name], common_p[name]) for name in common_f)


def test_eta_zero_preserves_legacy_logits_after_nonzero_finite_perturbation():
    torch.manual_seed(7)
    legacy = DIT(_config('P', finite=False), vocab_size=19).eval()
    torch.manual_seed(7)
    finite = DIT(_config('F', finite=True), vocab_size=19).eval()
    finite.load_state_dict({
        **finite.state_dict(),
        **legacy.state_dict(),
    })
    _activate_finite_only(finite)
    state = torch.randn(2, 5, 32)
    sigma = torch.tensor([0.2, 0.7])
    legacy_logits = legacy(
        state, sigma, inputs_are_embeddings=True, use_jvp_attn=True)
    finite_logits = finite(
        state, sigma, inputs_are_embeddings=True, use_jvp_attn=True,
        finite_time_features=torch.stack(
            (torch.tensor([0.1, 0.6]), torch.zeros(2)), dim=-1))
    torch.testing.assert_close(finite_logits, legacy_logits, rtol=0.0, atol=0.0)


def test_explicit_full_model_eta_jvp_and_f_head_match_torch_func():
    torch.manual_seed(11)
    model = DIT(_config('F'), vocab_size=19).eval()
    _activate(model)
    state = torch.randn(2, 5, 32)
    sigma = torch.tensor([0.2, 0.7])
    r = torch.tensor([0.0, 0.7])
    eta = torch.tensor([0.3, 0.2])
    tangent = torch.stack((torch.zeros_like(eta), torch.ones_like(eta)), -1)
    zeros = torch.zeros_like(state)

    def reference(value):
        return model(
            state, sigma, inputs_are_embeddings=True, use_jvp_attn=True,
            finite_time_features=value, return_output_features=True)

    reference_primal, reference_tangent = torch.func.jvp(
        reference, (torch.stack((r, eta), -1),), (tangent,))
    layer_primal, layer_tangent = model.forward_with_jvp(
        state, zeros, sigma, sigma_jvp=torch.zeros_like(sigma),
        inputs_are_embeddings=True,
        finite_time_features=torch.stack((r, eta), -1),
        finite_time_features_jvp=tangent, return_output_features=True)
    torch.testing.assert_close(
        layer_primal[0], reference_primal[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        layer_tangent[0], reference_tangent[0], rtol=3e-2, atol=3e-2)
    head = model.finite_correction_head
    layer_b = head(layer_primal[1])
    layer_db = torch.nn.functional.linear(layer_tangent[1], head.weight)
    reference_b, reference_db = torch.func.jvp(
        head, (reference_primal[1],), (reference_tangent[1],))
    torch.testing.assert_close(layer_b, reference_b, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(layer_db, reference_db, rtol=3e-2, atol=3e-2)


def test_f_stop_gradient_residual_has_connected_mixed_derivative():
    torch.manual_seed(31)
    logits = torch.randn(2, 3, 11, requires_grad=True)
    tangent = torch.randn_like(logits, requires_grad=True)
    correction = torch.randn_like(logits, requires_grad=True)
    correction_tangent = torch.randn_like(logits, requires_grad=True)
    eta = torch.tensor([0.2, 0.7])
    teacher = torch.softmax(torch.randn_like(logits), -1)
    tokens, value = _f_map_tokens(
        logits, tangent, correction, correction_tangent, eta, teacher)
    loss = tokens.mean()
    gradients = torch.autograd.grad(
        loss, (logits, tangent, correction, correction_tangent))
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in gradients)
    probability, probability_tangent = _softmax_jvp(logits, tangent)
    centered = correction - correction.mean(-1, keepdim=True)
    centered_tangent = correction_tangent - correction_tangent.mean(
        -1, keepdim=True)
    torch.testing.assert_close(
        value, probability + eta[:, None, None] * centered)
    assert torch.isfinite(probability_tangent + centered
                          + eta[:, None, None] * centered_tangent).all()


def test_p_log_domain_calibration_is_finite_and_fixed_at_zero_velocity():
    logits = torch.tensor(
        [[[-1.0e4, 0.0, 1.0e4], [1.0e4, -1.0e4, 0.0]]],
        requires_grad=True)
    tangent = torch.zeros_like(logits, requires_grad=True)
    eta = torch.tensor([0.4])
    teacher_log = torch.log_softmax(logits.detach(), dim=-1)
    tokens, reference = _p_map_tokens(
        logits, tangent, eta, teacher_log, 0.02, 0.1)
    assert torch.isfinite(tokens).all()
    torch.testing.assert_close(
        reference.sum(-1), torch.ones_like(reference[..., 0]),
        rtol=0.0, atol=2e-5)
    gradients = torch.autograd.grad(tokens.mean(), (logits, tangent))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert not teacher_log.requires_grad


def _sampler_harness(batch_size):
    contract = os.environ.get('TASK1_TVM_CONTRACT_DIR')
    if not contract:
        import pytest
        pytest.skip('TASK1_TVM_CONTRACT_DIR is required for contract tests.')
    root = Path(contract)
    model = Task1TVM50KFinal.__new__(Task1TVM50KFinal)
    torch.nn.Module.__init__(model)
    model._device = torch.device('cpu')
    model.register_buffer('_device_anchor', torch.zeros(()))
    model.register_buffer('canonical_k', torch.ones((), dtype=torch.long))
    model.register_buffer(
        'canonical_previous_k', torch.ones((), dtype=torch.long))
    model.register_buffer(
        'canonical_transition_start', torch.full((), -1, dtype=torch.long))
    model.config = SimpleNamespace(
        loader=SimpleNamespace(batch_size=batch_size))
    model.plan_seed = 20260921
    model.map_size = 96
    model.flm_time_eps = 1e-5
    model.line = 'F'
    model._sampler_profile = 'pre32'
    model.sampler_contract = yaml.safe_load(
        (root / 'sampler_f.yaml').read_text())
    model.train_contract = yaml.safe_load(
        (root / 'train_f.yaml').read_text())
    import utils
    model.lut_a2g, model.lut_g2a = utils.build_luts(K=50257)
    return model


def _collect_plan(model, step):
    size = int(model.config.loader.batch_size)
    chunks = [model.sample_step_plan(step, index)
              for index in range(256 // size)]
    local_t = torch.cat([value['local_t'] for value in chunks])
    local_sc = torch.cat([value['local_sc'] for value in chunks])
    map_mask = torch.cat([value['map_mask'] for value in chunks])
    rows = []
    for chunk in chunks:
        rows.extend(chunk['map_rows'])
    return local_t, local_sc, map_mask, rows


def test_global_plan_is_invariant_to_physical_microbatch_and_has_exact_quotas():
    plan8 = _collect_plan(_sampler_harness(8), 10000)
    plan4 = _collect_plan(_sampler_harness(4), 10000)
    for left, right in zip(plan8[:3], plan4[:3]):
        assert torch.equal(left, right)
    assert plan8[3] == plan4[3]
    assert int(plan8[2].sum()) == 96
    rows = [row for row in plan8[3] if row is not None]
    counts = {}
    for row in rows:
        counts[row['class']] = counts.get(row['class'], 0) + 1
        assert 0.0 <= row['r'] < row['s'] <= 0.95 + 1e-7
    assert counts == {'S': 12, 'M': 24, 'L': 24, 'D': 24, 'Z': 6, 'H': 6}
    deployment = [row for row in rows if row['class'] == 'D']
    assert sum(row['jittered'] for row in deployment) == 12
    assert sum(not row['jittered'] for row in deployment) == 12
    assert sum(row['r'] == 0.0 for row in rows) >= 18


def test_f_patch_adds_only_eight_separate_deployment_rows_after_32k():
    model = _sampler_harness(8)
    model._sampler_profile = 'uniform'
    rows = [row for row in _collect_plan(model, 32000)[3]
            if row is not None]
    counts = {}
    for row in rows:
        counts[row['class']] = counts.get(row['class'], 0) + 1
    assert counts == {
        'S': 12, 'M': 24, 'L': 16, 'D': 24,
        'D_patch': 8, 'Z': 6, 'H': 6,
    }
    original = [row for row in rows if row['class'] == 'D']
    added = [row for row in rows if row['class'] == 'D_patch']
    assert sum(row['grid'] == 'g0' for row in original) == 12
    assert sum(row['grid'] == 'gu' for row in original) == 12
    assert {row['grid'] for row in added} == {'gstar'}
    assert {row['source'] for row in added} == {'analytical_data'}
    assert {row['interval'] for row in added} == {0, 1, 2, 3}
    for interval in range(4):
        interval_rows = [row for row in added if row['interval'] == interval]
        assert len(interval_rows) == 2
        assert sum(row['jittered'] for row in interval_rows) == 1


def test_f_patch_prefix_rollout_is_detached_and_restricted_to_added_rows():
    model = Task1TVM50KFinal.__new__(Task1TVM50KFinal)
    torch.nn.Module.__init__(model)
    model.backbone = torch.nn.Linear(1, 1)
    model._device = torch.device('cpu')
    calls = []

    def query(state, r, eta):
        calls.append((float(r[0]), float(eta[0])))
        return state + 1.0

    model._finite_generation_query = query
    initial = torch.randn(1, 2, 3, requires_grad=True)
    row = {
        'class': 'D_patch', 'interval': 3,
        'nodes': [0.0, 0.2375, 0.475, 0.7125, 0.95],
    }
    result = model._detached_prefix_rollout_source(initial, row)
    assert len(calls) == 3
    assert not result.requires_grad
    with pytest.raises(RuntimeError, match='restricted'):
        model._detached_prefix_rollout_source(
            initial, {**row, 'class': 'D'})


def test_split_accumulation_and_resume_preserve_tiny_optimizer_state():
    def run(stop, checkpoint=None):
        torch.manual_seed(73)
        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        optimizer = torch.optim.AdamW(
            [parameter], lr=3e-4, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=0.0)
        target = parameter.detach().clone()
        start = 0
        if checkpoint is not None:
            parameter.data.copy_(checkpoint['parameter'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            target.copy_(checkpoint['target'])
            torch.random.set_rng_state(checkpoint['rng'])
            start = checkpoint['step']
        for step in range(start, stop):
            optimizer.zero_grad(set_to_none=True)
            full = torch.randn(256, 2)
            map_mask = torch.randperm(256)[:96]
            closure = map_mask[:32]
            for chunk in full.chunk(32):
                ((chunk @ parameter).square().mean() / 32.0).backward()
            ((full[map_mask] @ parameter).square().mean() * 0.3).backward()
            ((full[closure] @ parameter).square().mean() * 0.1).backward()
            optimizer.step()
            update = step + 1
            beta = min(0.99, 1.0 - 1.0 / (update + 1.0))
            target.mul_(beta).add_(parameter.detach(), alpha=1.0 - beta)
        return {
            'parameter': parameter.detach().clone(),
            'optimizer': copy.deepcopy(optimizer.state_dict()),
            'target': target.clone(), 'rng': torch.random.get_rng_state(),
            'step': stop,
        }

    uninterrupted = run(8)
    resumed = run(8, run(4))
    torch.testing.assert_close(
        resumed['parameter'], uninterrupted['parameter'], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        resumed['target'], uninterrupted['target'], rtol=0.0, atol=0.0)
    assert resumed['optimizer']['state'].keys() == uninterrupted['optimizer']['state'].keys()
    for key in resumed['optimizer']['state']:
        for name, value in resumed['optimizer']['state'][key].items():
            reference = uninterrupted['optimizer']['state'][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value, reference)
            else:
                assert value == reference


def test_post_step_adamw_moments_reconstruct_exact_zero_decay_update():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=3e-4, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=0.0)
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([0.7, -0.4], dtype=torch.float64)
    optimizer.step()
    state = optimizer.state[parameter]
    step = float(state['step'])
    beta1, beta2 = optimizer.param_groups[0]['betas']
    denominator = (
        state['exp_avg_sq'].sqrt() / math.sqrt(1.0 - beta2 ** step)
        + optimizer.param_groups[0]['eps'])
    delta = -(optimizer.param_groups[0]['lr'] / (1.0 - beta1 ** step)) \
        * state['exp_avg'] / denominator
    torch.testing.assert_close(parameter.detach() - before, delta)
