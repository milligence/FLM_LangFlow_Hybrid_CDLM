import inspect
import types
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo import FLMBase
from models.dit import DIT
from task1_tvm_ce import (
    Task1TVMJointJ0,
    Task1TVMSCRepair,
    calibrated_relative_distributions,
    finite_map_update,
    finite_sc_regression_state,
    gap_mode_probabilities,
    integer_quota_counts,
    sample_global_interval_batch,
    tvm_quantity_from_logits_jvp,
)
from langflow_hybrid.ops import finite_sc_gate, finite_sc_update


def _tiny_dit_config(name, double_temb, finite_sc_residual_head=False):
    return OmegaConf.create({
        'is_di4c': False,
        'algo': {
            'name': name,
            'causal_attention': False,
            'embedding_state': False,
            'self_conditioning': True,
            'normalize_embeddings': False,
            'codebook_gradient_mode': 'all',
            'classification_prototype_mode': 'direct_vocab_state',
            'double_temb': double_temb,
            'learnable_loss_weighting': False,
            'finite_sc_residual_head': finite_sc_residual_head,
            'finite_sc_residual_hidden': 4,
        },
        'model': {
            'hidden_size': 8,
            'cond_dim': 8,
            'n_blocks': 1,
            'n_heads': 1,
            'dropout': 0.0,
            'scale_by_sigma': True,
        },
    })


def test_tvm_quantity_preserves_mass_and_calibration_fixed_point():
    torch.manual_seed(17)
    logits = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    weighted_jvp = 0.15 * torch.randn_like(logits)
    probabilities, _, gate, quantity = tvm_quantity_from_logits_jvp(
        logits, weighted_jvp)

    assert torch.allclose(
        quantity.sum(dim=-1), torch.ones_like(quantity[..., 0]),
        atol=2e-6, rtol=0.0)
    assert torch.all(quantity > 0.0)

    student_log_probabilities, teacher_target = (
        calibrated_relative_distributions(
            probabilities, gate, quantity.detach(), kappa=0.02))
    loss = -(teacher_target * student_log_probabilities).sum(dim=-1).mean()
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient.abs().max().item() < 2e-6


def test_finite_map_update_supports_one_and_multiple_intervals():
    state = torch.tensor([[[2.0, -1.0]]])
    probability = torch.tensor([[[0.25, 0.75]]])
    r = torch.tensor([0.0])

    one_step = finite_map_update(state, probability, r, torch.tensor([1.0]))
    half_step = finite_map_update(state, probability, r, torch.tensor([0.5]))

    assert torch.equal(one_step, probability)
    assert torch.equal(half_step, 0.5 * state + 0.5 * probability)


def test_gap_curriculum_interpolates_without_an_inference_grid():
    assert gap_mode_probabilities(0.0) == (0.15, 0.45, 0.30, 0.10)
    assert gap_mode_probabilities(1.0) == (0.10, 0.20, 0.30, 0.40)
    middle = gap_mode_probabilities(0.5)
    torch.testing.assert_close(
        torch.tensor(middle),
        torch.tensor((0.125, 0.325, 0.30, 0.25)))
    assert sum(middle) == 1.0


def test_global_batch_quotas_are_exact_and_reproducible():
    assert integer_quota_counts(
        gap_mode_probabilities(0.0), 128) == (19, 58, 38, 13)
    assert integer_quota_counts(
        gap_mode_probabilities(1.0), 128) == (13, 26, 38, 51)

    first = sample_global_interval_batch(
        128, 511 / 512, 45, 1.0, torch.device('cpu'), seed=41)
    second = sample_global_interval_batch(
        128, 511 / 512, 45, 1.0, torch.device('cpu'), seed=41)
    for left, right in zip(first[:5], second[:5]):
        torch.testing.assert_close(left, right)
    r, s, diagonal, _, mode, counts = first
    assert int((r == 0).sum()) == 45
    assert counts == (13, 26, 38, 51)
    assert tuple(int((mode == index).sum()) for index in range(4)) == counts
    assert torch.equal(diagonal, mode == 0)
    assert torch.all(s >= r)
    assert torch.all(s <= 511 / 512)


def test_checkpoint_copy_zero_initializes_only_eta_injection_output():
    torch.manual_seed(23)
    teacher = DIT(
        _tiny_dit_config('langflow_flm_hybrid', False), vocab_size=13)
    student = DIT(
        _tiny_dit_config('task1_tvm_ce', True), vocab_size=13)
    owner = type('Owner', (), {'backbone': student})()
    FLMBase._copy_teacher_weights_to_student(owner, teacher.state_dict())

    assert torch.count_nonzero(
        student.sigma_map_prime.mlp[0].weight).item() > 0
    assert torch.count_nonzero(
        student.sigma_map_prime.mlp[2].weight).item() == 0
    assert torch.count_nonzero(
        student.sigma_map_prime.mlp[2].bias).item() == 0

    state = torch.randn(2, 4, 13)
    r_condition = torch.tensor([0.2, 0.7])
    teacher.eval()
    student.eval()
    teacher_logits = teacher(state, r_condition, use_jvp_attn=True)
    student_short = student(
        state, r_condition, torch.tensor([0.1, 0.1]), use_jvp_attn=True)
    student_long = student(
        state, r_condition, torch.tensor([0.9, 0.9]), use_jvp_attn=True)
    torch.testing.assert_close(student_short, teacher_logits)
    torch.testing.assert_close(student_long, teacher_logits)


def test_finite_sc_gate_has_exact_zero_and_no_epsilon_floor():
    gaps = torch.tensor([0.0, 1e-12, 1e-6, 1e-3])
    gate = finite_sc_gate(gaps, 1e-3)

    assert gate[0].item() == 0.0
    assert gate[1].item() > 0.0
    assert gate[1].item() < gate[2].item() < gate[3].item()
    torch.testing.assert_close(gate[-1], torch.tensor(0.5), atol=1e-7, rtol=0)
    assert 'epsilon' not in inspect.getsource(finite_sc_gate).lower()


def test_finite_sc_update_converges_to_carried_state_as_gap_vanishes():
    carried = torch.tensor([[[1.0, -2.0]]])
    proposal = torch.tensor([[[5.0, 6.0]]])
    at_zero = finite_sc_update(carried, proposal, torch.tensor([0.0]), 1e-3)
    near = finite_sc_update(carried, proposal, torch.tensor([1e-9]), 1e-3)
    farther = finite_sc_update(carried, proposal, torch.tensor([1e-5]), 1e-3)

    torch.testing.assert_close(at_zero, carried, atol=0.0, rtol=0.0)
    assert torch.linalg.vector_norm(near - carried) < torch.linalg.vector_norm(
        farther - carried)


def test_residual_head_is_zero_initialized_and_numerically_zero():
    model = DIT(
        _tiny_dit_config(
            'task1_tvm_sc_repair', True, finite_sc_residual_head=True),
        vocab_size=13)
    hidden = torch.randn(2, 3, 8)

    assert torch.count_nonzero(
        model.sc_residual_head[-1].weight).item() == 0
    assert torch.count_nonzero(
        model.sc_residual_head[-1].bias).item() == 0
    torch.testing.assert_close(
        model.sc_residual_head(hidden), torch.zeros_like(hidden))


def test_sc_coordinate_is_a_frozen_buffer_not_an_optimizer_parameter():
    init_source = inspect.getsource(Task1TVMSCRepair.__init__)
    optimizer_source = inspect.getsource(Task1TVMSCRepair.configure_optimizers)

    assert "register_buffer(\n            'sc_coordinate'" in init_source
    assert 'self._get_optimizer_parameters()' in optimizer_source
    assert 'sc_coordinate' not in optimizer_source


def test_student_ema_load_allows_rebuilt_rotary_buffer_only():
    source = inspect.getsource(Task1TVMSCRepair._load_student_ema_initialization)

    assert "'rotary_emb.inv_freq'" in source
    assert 'unexpected_missing = set(incompatible.missing_keys) - allowed_missing' in source
    assert 'incompatible.unexpected_keys' in source


def test_step0_checkpoint_waits_for_train_dataloader_setup():
    source = inspect.getsource(Task1TVMSCRepair.on_train_start)

    assert 'super().on_train_start()' in source
    assert 'self.trainer.save_checkpoint(path, weights_only=False)' in source
    assert 'on_fit_start' not in Task1TVMSCRepair.__dict__


def test_gap_gradient_proxy_is_disabled_during_validation():
    owner = types.SimpleNamespace(sc_d50=1e-3)
    loss = torch.ones(2)
    hidden = torch.ones(2, 3, 4)
    delta = torch.tensor([1e-4, 1e-3])

    with torch.no_grad():
        Task1TVMSCRepair._log_sc_gap_gradient_proxies(
            owner, loss, hidden, delta)


def test_sc_diagnostics_report_loaded_checkpoint_step():
    source = inspect.getsource(Task1TVMSCRepair.run_sc_diagnostics)

    assert "'_loaded_checkpoint_global_step'" in source


def test_gradient_probe_offloads_full_gradients_before_next_loss():
    source = inspect.getsource(Task1TVMSCRepair._gradient_geometry)

    assert 'gradient.detach().float().cpu()' in source
    assert 'del raw_gradients' in source


def test_terminal_teacher_query_detaches_both_text_and_sc_endpoints():
    loss_source = inspect.getsource(Task1TVMSCRepair.loss)

    assert 'endpoint.detach(), s_off, endpoint_h.detach()' in loss_source
    assert 'endpoint.detach(), s[index], endpoint_h.detach()' in loss_source


def test_text_jvp_path_retains_parameter_backpropagation():
    weight = torch.randn(5, 7, requires_grad=True)
    state = torch.randn(2, 3, 5)
    eta = torch.tensor([0.2, 0.7])

    def logits_at_eta(value):
        return state @ weight + value[:, None, None] * (state @ weight.square())

    logits, weighted_jvp = torch.func.jvp(
        logits_at_eta, (eta,), (eta * (1.0 - eta),))
    probabilities, _, gate, _ = tvm_quantity_from_logits_jvp(
        logits, weighted_jvp)
    student_log, target = calibrated_relative_distributions(
        probabilities, gate, probabilities.detach(), kappa=0.02)
    loss = -(target * student_log).sum(dim=-1).mean()
    gradient = torch.autograd.grad(loss, weight)[0]

    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_sc_regression_detaches_R_E0_but_trains_residual():
    probabilities = torch.softmax(
        torch.randn(2, 3, 5), dim=-1).requires_grad_()
    coordinate = torch.randn(5, 4, requires_grad=True)
    residual = torch.randn(2, 3, 4, requires_grad=True)
    carried = torch.randn(2, 3, 4)
    output = finite_sc_regression_state(
        probabilities, coordinate, residual, carried,
        torch.tensor([1e-4, 1e-3]), 1e-3)
    output.sum().backward()

    assert probabilities.grad is None
    assert coordinate.grad is None
    assert residual.grad is not None
    assert residual.grad.abs().sum().item() > 0.0


def test_all_inference_resolutions_use_one_finite_step_per_interval():
    source = inspect.getsource(Task1TVMSCRepair.generate_samples)

    assert 'torch.func.jvp' not in source
    assert source.count('self.finite_step(') == 1
    assert 'self.last_sampling_nfe = expected_steps' in source
    for nfe in (1, 2, 4, 16, 512):
        grid = [index / nfe for index in range(nfe + 1)]
        assert len(tuple(zip(grid, grid[1:]))) == nfe


def test_reference_gate_candidates_are_independently_reproducible():
    delta = torch.linspace(0.0, 1.0, 1025)
    for d50 in (1e-4, 1e-3, 1e-2):
        first = finite_sc_gate(delta, d50)
        second = finite_sc_gate(delta.clone(), d50)
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_joint_j0_contract_and_local_time_mapping():
    root = Path(__file__).parents[1]
    config = OmegaConf.load(root / 'configs/algo/task1_tvm_joint_j0.yaml')
    assert config.name == 'task1_tvm_joint_j0'
    assert config.tvm_sc_pair_weight == 0.03
    assert config.tvm_local_weight == 0.15
    assert config.tvm_local_warmup_steps == 250
    assert config.tvm_local_sc_probability == 0.25
    assert config.tvm_local_time_sampling == 'v1_m_tau25_global256'
    assert config.tvm_endpoint_teacher_sc_mode == 'predicted_finite_state'
    assert config.tvm_sc_gradient_probe_interval == 100
    assert config.tvm_sc_backbone_lr == 5e-5
    assert config.tvm_sc_input_projection_lr == 5e-5
    assert config.tvm_sc_residual_head_lr == 5e-5

    owner = types.SimpleNamespace(
        sc_pair_weight=0.03,
        local_weight=0.15,
        local_warmup_steps=250,
        global_step=125,
        _task1_lut_anchor_physical_t=[0.0, 1.0 / 128.0],
        _t_to_tau=lambda value: value,
        _task1_physical_time=lambda value: value,
    )
    assert Task1TVMJointJ0._sc_loss_weights(owner) == 0.03
    assert Task1TVMJointJ0._local_lambda(owner) == 0.075
    group_id = torch.arange(7)
    random = torch.full((7,), 0.5)
    tau, physical_t, returned_group = (
        Task1TVMJointJ0._local_times_from_groups(
            owner, group_id, random))
    torch.testing.assert_close(
        tau,
        torch.tensor([
            0.0, 1.0 / 256.0, 1.0 / 256.0,
            (1.0 / 128.0 + 0.1) / 2.0,
            0.25, 0.55, 0.85,
        ]))
    torch.testing.assert_close(physical_t, tau)
    assert torch.equal(returned_group, group_id)

    init_source = inspect.getsource(
        Task1TVMJointJ0._load_student_ema_initialization)
    assert '_extract_ema_state_dict' in init_source
    assert 'sc_residual_head' not in init_source
    loss_source = inspect.getsource(Task1TVMJointJ0.loss)
    assert 'super().loss' in loss_source
    assert 'local_lambda * local_ce' in loss_source
    assert 'weighted_local' in inspect.getsource(
        Task1TVMJointJ0._gradient_probe_extra_losses)
