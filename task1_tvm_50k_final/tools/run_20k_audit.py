#!/usr/bin/env python3
"""Read-only checkpoint diagnostics for TASK1-TVM-50K-FINAL.

This tool lives outside the training repository source manifest.  It composes
the existing Hydra model config, loads a full-state checkpoint, and writes one
atomic JSON artifact without mutating optimizer, checkpoint, or training RNG.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import yaml


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _compose_model(args, bindings, audit):
    repo = Path(bindings['repo_root']).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ['FLM_TOKENIZER_PATH'] = str(Path(bindings['data']['tokenizer']).resolve())
    os.environ['FLM_PACKED_DATA_DIR'] = str(Path(bindings['data']['owt_train']).resolve())
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    from hydra import compose, initialize_config_dir
    import algorithm_registry
    import dataloader

    output_parent = args.output.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    overrides = [
        f'algo=task1_tvm_50k_final_{args.line}',
        'model=small_128',
        'data=openwebtext_327m_packed',
        'mode=sample_eval',
        f'algo.task1_tvm_contract_dir={args.contract_dir.resolve()}',
        f'algo.task1_eval_weight_role={args.model_type}',
        'algo.task1_compile_finite_jvp=false',
        'algo.task1_compile_calibration_kernels=false',
        f'data.packed_dir={Path(bindings["data"]["owt_train"]).resolve()}',
        f'data.cache_dir={Path(bindings["data"]["owt_train"]).resolve()}',
        f'data.tokenizer_name_or_path={Path(bindings["data"]["tokenizer"]).resolve()}',
        'loader.global_batch_size=256',
        'loader.batch_size=32',
        'loader.eval_global_batch_size=32',
        'loader.eval_batch_size=32',
        'trainer.devices=1',
        'trainer.num_nodes=1',
        'trainer.accumulate_grad_batches=8',
        'strategy=single_device',
        f'eval.checkpoint_path={args.checkpoint.resolve()}',
        f'eval.gen_ppl_eval_model_name_or_path={Path(bindings["data"]["evaluator"]).resolve()}',
        f'checkpointing.save_dir={output_parent}',
        f'hydra.run.dir={output_parent}',
    ]
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
        config = compose(config_name='config', overrides=overrides)
    tokenizer = dataloader.get_tokenizer(config)
    model_class = algorithm_registry.get_algorithm_class(config.algo.name)
    model = model_class.load_from_checkpoint(
        str(args.checkpoint.resolve()), tokenizer=tokenizer,
        config=config, weights_only=False, map_location='cpu')
    model._loaded_checkpoint_global_step = int(args.step)
    model = model.to('cuda')
    model._eval_mode()
    model.eval()
    model.training_target_ema.eval()
    return model, repo


def _load_tokens(repo: Path, packed_dir: Path, count: int) -> torch.Tensor:
    sys.path.insert(0, str(repo))
    from packed_dataset import PackedTokenDataset
    dataset = PackedTokenDataset(str(packed_dir), 'validation')
    rows = []
    for index in range(count):
        item = dataset[index]
        value = item['input_ids'] if isinstance(item, dict) else item
        rows.append(torch.as_tensor(value, dtype=torch.long))
    return torch.stack(rows).cpu()


def _physical_bin(audit: dict, value: float) -> str:
    for item in audit['heldout_local']['physical_bins']:
        low, high = float(item['low']), float(item['high'])
        if low <= value < high or (bool(item['right_closed']) and low <= value <= high):
            return str(item['name'])
    raise ValueError(f'Physical t={value} is outside the audit bins.')


@torch.no_grad()
def _local_curve(model, tokens: torch.Tensor, audit: dict, step: int) -> list[dict]:
    specification = audit['heldout_local']
    batch_size = int(specification['batch_size'])
    rows = []
    for node_index, physical_t in enumerate(specification['physical_t_nodes']):
        totals = {
            'loss': 0.0, 'top1_prob': 0.0, 'top1_accuracy': 0.0,
            'posterior_entropy': 0.0, 'margin_top1_top2': 0.0,
            'CE_nats': 0.0, 'target_probability': 0.0,
            'logit_margin_top1_top2': 0.0,
        }
        token_count = 0
        tau_value = None
        for start in range(0, tokens.shape[0], batch_size):
            clean = tokens[start:start + batch_size].to(model.device)
            count = clean.shape[0]
            t = torch.full((count,), float(physical_t), device=model.device)
            global_indices = torch.arange(start, start + count, device=model.device)
            noise = model._noise_rows(
                count, clean.shape[1], 'audit20k_local_noise',
                int(specification['noise_seed']) + node_index, global_indices)
            state = model.corrupt_vocab_state(clean, t, noise=noise)
            logits, log_probability, probability = model.local_posterior(
                state, t, None, dropout_enabled=False)
            if model.line == 'F':
                token_loss = 0.5 * (
                    probability.square().sum(dim=-1)
                    - 2.0 * probability.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
                    + 1.0)
            else:
                token_loss = -log_probability.gather(
                    -1, clean.unsqueeze(-1)).squeeze(-1)
            top2 = probability.topk(2, dim=-1).values
            logit_top2 = logits.topk(2, dim=-1).values
            target_probability = probability.gather(
                -1, clean.unsqueeze(-1)).squeeze(-1)
            target_log_probability = log_probability.gather(
                -1, clean.unsqueeze(-1)).squeeze(-1)
            total_here = clean.numel()
            totals['loss'] += float(token_loss.sum())
            totals['top1_prob'] += float(top2[..., 0].sum())
            totals['top1_accuracy'] += float((probability.argmax(dim=-1) == clean).sum())
            totals['posterior_entropy'] += float((-(probability * log_probability).sum(dim=-1)).sum())
            totals['margin_top1_top2'] += float((top2[..., 0] - top2[..., 1]).sum())
            totals['CE_nats'] += float((-target_log_probability).sum())
            totals['target_probability'] += float(target_probability.sum())
            totals['logit_margin_top1_top2'] += float(
                (logit_top2[..., 0] - logit_top2[..., 1]).sum())
            token_count += total_here
            if tau_value is None:
                tau_value = float(model._t_to_tau(t[:1].float())[0])
        rows.append({
            'physical_t': float(physical_t),
            'tau': tau_value,
            'noise_bin': _physical_bin(audit, float(physical_t)),
            **{name: value / token_count for name, value in totals.items()},
        })
    return rows


def _map_probe_rows(model, audit: dict, step: int) -> dict[str, list[dict]]:
    specification = audit['map_validation']
    if bool(specification.get('patch_fixed_data_source', False)):
        needed = int(specification['continuous_rows_per_bucket'])
        output = {name: [] for name in ('S', 'M', 'L', 'H')}
        repeat = 0
        while any(len(values) < needed for values in output.values()):
            rows = model._map_rows(
                30000, model._generator('patch_fixed_map_pair', 30000, repeat))
            for row in rows:
                label = str(row['class'])
                if label in output and len(output[label]) < needed:
                    output[label].append({
                        **row, 'source': 'analytical_data', 'closure': False})
            repeat += 1
            if repeat > 64:
                raise RuntimeError('Could not fill patch fixed map buckets.')
        per_interval = int(specification['deployment_rows_per_interval'])
        for grid_name, grid in specification['deployment_grids'].items():
            for interval, (r, s) in enumerate(zip(grid, grid[1:])):
                bucket = f'{grid_name}_interval_{interval}'
                output[bucket] = [model._row(
                    'D', float(r), float(s), interval, grid_name, False,
                    30000, nodes=[float(value) for value in grid],
                    source='analytical_data') for _ in range(per_interval)]
        return output
    needed = int(specification['sequences_per_bucket'])
    class_to_bucket = {str(value): str(key) for key, value in specification['buckets'].items()}
    output = {bucket: [] for bucket in specification['buckets']}
    repeat = 0
    while any(len(value) < needed for value in output.values()):
        rows = model._map_rows(
            step, model._generator('audit20k_map_pair', step, repeat))
        for row in rows:
            bucket = class_to_bucket.get(str(row['class']))
            if bucket is not None and len(output[bucket]) < needed:
                output[bucket].append(row)
        repeat += 1
        if repeat > 64:
            raise RuntimeError('Could not fill the fixed map audit buckets.')
    return output


@torch.no_grad()
def _map_metrics(model, tokens: torch.Tensor, audit: dict, step: int) -> tuple[list[dict], float]:
    from task1_tvm_50k_final import (
        _student_log_calibration, _teacher_log_calibration)

    specification = audit['map_validation']
    microbatch = int(specification['microbatch'])
    probes = _map_probe_rows(model, audit, step)
    results = []
    tangent_total = 0.0
    tangent_count = 0
    token_cursor = 0
    for bucket_index, (bucket, rows) in enumerate(probes.items()):
        totals: dict[str, float] = {}
        token_count = 0
        for start in range(0, len(rows), microbatch):
            chunk = rows[start:start + microbatch]
            clean = tokens[
                token_cursor + start:token_cursor + start + len(chunk)].to(
                    model.device)
            r = torch.tensor([row['r'] for row in chunk], device=model.device)
            s = torch.tensor([row['s'] for row in chunk], device=model.device)
            eta = torch.tensor([row['eta'] for row in chunk], device=model.device)
            indices = torch.arange(
                token_cursor + start,
                token_cursor + start + len(chunk), device=model.device)
            noise = model._noise_rows(
                len(chunk), clean.shape[1], 'audit20k_map_noise',
                int(specification['noise_seed']), indices)
            state = model.corrupt_vocab_state(clean, r, noise=noise)
            output = model.finite_with_eta_jvp(state, r, eta)
            eta_view = eta[:, None, None]
            if model.line == 'F':
                value, value_tangent = model._f_objective_terms(
                    output['logits'], output['dlogits'], output['b_raw'],
                    output['db_raw'], eta)
                endpoint = (1.0 - eta_view) * state + eta_view * value
                target = model.canonical_target(endpoint.detach(), s)
                predicted_velocity = (
                    -state + value + eta_view * value_tangent
                ) / (1.0 - r[:, None, None])
                target_velocity = (
                    target.prob - endpoint.detach()
                ) / (1.0 - s[:, None, None])
                residual = (1.0 - s[:, None, None]) * (
                    predicted_velocity - target_velocity)
                q_value = (
                    value + eta_view * (1.0 - eta_view) * value_tangent)
                correction = output['b_raw'].float()
                correction = correction - correction.mean(
                    dim=-1, keepdim=True)
                pred_norm = predicted_velocity.norm(dim=-1)
                target_norm = target_velocity.norm(dim=-1)
                cosine = F.cosine_similarity(
                    predicted_velocity, target_velocity, dim=-1, eps=1e-12)
                values = {
                    'map_residual': residual.norm(dim=-1),
                    'velocity_norm': pred_norm,
                    'target_velocity_norm': target_norm,
                    'cosine_velocity': cosine,
                    'signal_residual_E_squared_vocab_sum': residual.square().sum(dim=-1),
                    'raw_terminal_residual_squared_vocab_sum': (
                        residual / (1.0 - s[:, None, None])).square().sum(dim=-1),
                    'signal_Q_vs_pT_cosine': F.cosine_similarity(
                        q_value, target.prob, dim=-1, eps=1e-12),
                    'd_eta_A_norm': value_tangent.norm(dim=-1),
                    'eta_times_1_minus_eta_times_d_eta_A_norm': (
                        eta_view * (1.0 - eta_view) * value_tangent).norm(dim=-1),
                    'eta_B_norm': (eta_view * correction).norm(dim=-1),
                    'eta_squared_B_norm': (eta_view.square() * correction).norm(dim=-1),
                    'signed_A_negative_mass': (-value).clamp_min(0.0).sum(dim=-1),
                    'signed_Q_negative_mass': (-q_value).clamp_min(0.0).sum(dim=-1),
                    'A_mass_abs_error': (value.sum(dim=-1) - 1.0).abs(),
                    'Q_mass_abs_error': (q_value.sum(dim=-1) - 1.0).abs(),
                }
                tangent = value_tangent.norm(dim=-1)
            else:
                log_reference = F.log_softmax(output['logits'].float(), dim=-1)
                reference = log_reference.exp()
                velocity = eta_view * (1.0 - eta_view) * output['dlogits'].float()
                mean = (reference * velocity).sum(dim=-1, keepdim=True)
                gate = 1.0 + velocity - mean
                raw_q = reference * gate
                endpoint = (1.0 - eta_view) * state + eta_view * reference
                target = model.canonical_target(endpoint.detach(), s)
                student_log = _student_log_calibration(
                    log_reference, gate, float(model.kappa))
                teacher_log = _teacher_log_calibration(
                    log_reference, target.log_prob, float(model.kappa))
                student = student_log.exp()
                teacher = teacher_log.exp()
                values = {
                    'KL(qT||qS)': (teacher * (teacher_log - student_log)).sum(dim=-1),
                    'Q_minus_teacher_L1': (raw_q - target.prob).abs().sum(dim=-1),
                    'negative_mass': (-raw_q).clamp_min(0.0).sum(dim=-1),
                    'student_entropy': -(student * student_log).sum(dim=-1),
                    'teacher_entropy': -(teacher * teacher_log).sum(dim=-1),
                    'calibration_strength': (student - raw_q).abs().sum(dim=-1),
                }
                tangent = output['dlogits'].float().norm(dim=-1)
            count = clean.numel()
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.sum())
            tangent_total += float(tangent.sum())
            tangent_count += tangent.numel()
            token_count += count
        result = {
            'bucket': bucket,
            **{name: value / token_count for name, value in totals.items()},
            'samples': len(rows),
        }
        if model.line == 'F':
            result['F_Q_KL'] = None
            result['F_Q_KL_reason'] = (
                'signed_A_and_Q_are_not_categorical_distributions')
        results.append(result)
        token_cursor += len(rows)
    return results, tangent_total / max(1, tangent_count)


@torch.no_grad()
def _sc_metrics(model, tokens: torch.Tensor, audit: dict, step: int) -> list[dict]:
    specification = audit['sc_validation']
    batch_size = int(specification['batch_size'])
    maximum_k = max(int(value) for value in specification['K'])
    totals = {int(k): {'kl': 0.0, 'cold': 0.0, 'refined': 0.0, 'count': 0}
              for k in specification['K']}
    for node_index, physical_t in enumerate(specification['physical_t_nodes']):
        for start in range(0, tokens.shape[0], batch_size):
            clean = tokens[start:start + batch_size].to(model.device)
            count = clean.shape[0]
            t = torch.full((count,), float(physical_t), device=model.device)
            indices = torch.arange(start, start + count, device=model.device)
            noise = model._noise_rows(
                count, clean.shape[1], 'audit20k_sc_noise',
                int(specification['noise_seed']) + node_index, indices)
            state = model.corrupt_vocab_state(clean, t, noise=noise)
            chain = model._target_chain(state, t, maximum_k, bias_step=step)
            cold_probability, cold_log = chain[0]
            if model.line == 'F':
                cold_loss = 0.5 * (
                    cold_probability.square().sum(dim=-1)
                    - 2.0 * cold_probability.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
                    + 1.0)
            else:
                cold_loss = -cold_log.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
            for k in totals:
                probability, log_probability = chain[k - 1]
                if model.line == 'F':
                    refined_loss = 0.5 * (
                        probability.square().sum(dim=-1)
                        - 2.0 * probability.gather(-1, clean.unsqueeze(-1)).squeeze(-1)
                        + 1.0)
                else:
                    refined_loss = -log_probability.gather(
                        -1, clean.unsqueeze(-1)).squeeze(-1)
                kl = (probability * (log_probability - cold_log)).sum(dim=-1)
                totals[k]['kl'] += float(kl.sum())
                totals[k]['cold'] += float(cold_loss.sum())
                totals[k]['refined'] += float(refined_loss.sum())
                totals[k]['count'] += clean.numel()
    return [{
        'K': k,
        'cold_vs_refined_KL': values['kl'] / values['count'],
        'cold_loss': values['cold'] / values['count'],
        'refined_loss': values['refined'] / values['count'],
        'generation_metric': None,
    } for k, values in sorted(totals.items())]


def _controller_state(model, step: int) -> dict:
    calibrations = [event for event in model._calibration_events
                    if int(event.get('completed_updates', -1)) <= step]
    audits = [event for event in model._gradient_audit_events
              if int(event.get('completed_updates', -1)) <= step]
    latest_calibration = calibrations[-1] if calibrations else None
    latest_audit = audits[-1] if audits else None
    result = {
        'canonical_K': int(model.canonical_k),
        'canonical_previous_K': int(model.canonical_previous_k),
        'canonical_transition_start': int(model.canonical_transition_start),
        'C_map': float(model.map_calibration),
        'C_closure': float(model.closure_calibration),
        'latest_calibration': latest_calibration,
        'latest_gradient_audit': latest_audit,
        'gpu_wall_seconds': float(model.gpu_wall_seconds),
        'cumulative_tokens': int(model.cumulative_tokens),
    }
    if latest_calibration and latest_calibration.get('packs'):
        local = sorted(float(item['local_norm']) for item in latest_calibration['packs'])
        raw_map = sorted(float(item['raw_map_norm']) for item in latest_calibration['packs'])
        middle = len(local) // 2
        result['local_grad_norm_median'] = local[middle]
        result['map_grad_norm_raw_median'] = raw_map[middle]
    return result


def _relative_l2(left, right) -> float:
    delta_square = 0.0
    base_square = 0.0
    for first, second in zip(left, right):
        first_value = first.detach().float()
        second_value = second.detach().to(device=first_value.device, dtype=torch.float32)
        delta_square += float((first_value - second_value).square().sum())
        base_square += float(first_value.square().sum())
    return math.sqrt(delta_square) / max(math.sqrt(base_square), 1e-12)


def _parameter_diagnostics(model) -> dict:
    online = list(model.backbone.parameters())
    target = list(model.training_target_ema.parameters())
    parameter_norm = math.sqrt(sum(float(value.detach().float().square().sum()) for value in online))
    result = {
        'parameter_norm': parameter_norm,
        'target_ema_relative_l2_lag': _relative_l2(online, target),
        'eval_ema_relative_l2_lag': None,
    }
    if model.ema is not None:
        trainable = [value for value in model._get_parameters() if value.requires_grad]
        shadows = list(model.ema.shadow_params)
        if len(trainable) == len(shadows):
            result['eval_ema_relative_l2_lag'] = _relative_l2(trainable, shadows)
    return result


def _accumulated_gradients(scalars, parameters):
    accumulated = [None] * len(parameters)
    for scalar, multiplier in scalars:
        gradients = torch.autograd.grad(
            scalar * float(multiplier), parameters,
            retain_graph=False, allow_unused=True)
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            value = gradient.detach().float()
            if accumulated[index] is None:
                accumulated[index] = value
            else:
                accumulated[index].add_(value)
    return accumulated


def _gradient_cosine_probe(model, step: int, seed: int) -> dict:
    model.setup(None)
    if model._calibration_tokens_cpu is None:
        from packed_dataset import PackedTokenDataset
        dataset = PackedTokenDataset(model.config.data.packed_dir, 'train')
        offset = int(model.plan_seed % max(1, len(dataset) - 128))
        rows = []
        for index in range(offset, offset + 128):
            item = dataset[index]
            value = item['input_ids'] if isinstance(item, dict) else item
            rows.append(torch.as_tensor(value, dtype=torch.long))
        model._calibration_tokens_cpu = torch.stack(rows).cpu()
    named = model._shared_calibration_parameters()
    parameters = tuple(parameter for _, parameter in named)
    pack_index = 0
    pack_tokens = model._calibration_tokens_cpu[:32].to(model.device)
    valid = torch.ones_like(pack_tokens, dtype=torch.long)
    prior_mode = model.backbone.training
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    model.backbone.train()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        time_plan = model._local_times(
            step, model._generator('calibration_local_time', step, pack_index))[:32]
        sc_enabled = bool(torch.rand(
            (), device=model.device,
            generator=model._generator(
                'calibration_local_sc', step, pack_index)) < 0.25)
        local_chunk = int(model.config.loader.batch_size)

        def local_scalars():
            for begin in range(0, 32, local_chunk):
                end = min(begin + local_chunk, 32)
                count = end - begin
                plan = {
                    'global_indices': torch.arange(
                        begin, end, device=model.device),
                    'local_t': time_plan[begin:end],
                    'local_sc': torch.full(
                        (count,), sc_enabled, device=model.device,
                        dtype=torch.bool),
                }
                scalar, _ = model._local_objective(
                    pack_tokens[begin:end], valid[begin:end], plan, step,
                    noise_stream='calibration_local_noise')
                yield scalar, count / 32.0

        local = _accumulated_gradients(local_scalars(), parameters)
        local_cpu = [
            value.cpu() if value is not None else None for value in local]
        del local

        all_rows = model._map_rows(
            step, model._generator('calibration_map_pair', step))
        if len(all_rows) != 96:
            raise RuntimeError('Gradient cosine probe requires an active 96-row map stage.')
        rows = [dict(row, closure=False) for row in all_rows[:24]]

        def map_scalars():
            for begin in range(0, 24, model.map_microbatch):
                end = min(begin + model.map_microbatch, 24)
                count = end - begin
                plan = {
                    'global_indices': torch.arange(
                        begin, end, device=model.device),
                    'map_mask': torch.ones(
                        count, device=model.device, dtype=torch.bool),
                    'map_rows': rows[begin:end],
                }
                scalar, selected, _, _ = model._map_objective(
                    pack_tokens[begin:end], plan, step,
                    noise_stream='calibration_map_noise')
                if selected != count:
                    raise RuntimeError('Gradient cosine map probe lost rows.')
                yield scalar, count / 24.0

        mapped = _accumulated_gradients(map_scalars(), parameters)
        dot = 0.0
        local_square = 0.0
        map_square = 0.0
        for local_value, map_value in zip(local_cpu, mapped):
            if local_value is not None:
                local_square += float(local_value.double().square().sum())
            if map_value is not None:
                map_cpu = map_value.cpu()
                map_square += float(map_cpu.double().square().sum())
                if local_value is not None:
                    dot += float((local_value.double() * map_cpu.double()).sum())
        local_norm = math.sqrt(local_square)
        map_norm = math.sqrt(map_square)
        return {
            'grad_cosine': dot / max(local_norm * map_norm, 1.0e-30),
            'probe_local_grad_norm': local_norm,
            'probe_map_grad_norm': map_norm,
            'probe_pack': pack_index,
            'class_weighting': 'contracted map class weights',
        }
    finally:
        model.backbone.train(prior_mode)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def _activation_probe(model, tokens: torch.Tensor, step: int, audit: dict) -> dict:
    specification = audit['optimization_validation']
    count = int(specification['activation_sequences'])
    clean = tokens[:count].to(model.device)
    per_block = [dict(square=0.0, count=0) for _ in model.backbone.blocks]
    handles = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            value = value.detach().float()
            per_block[index]['square'] += float(value.square().sum())
            per_block[index]['count'] += value.numel()
        return hook

    for index, block in enumerate(model.backbone.blocks):
        handles.append(block.register_forward_hook(make_hook(index)))
    try:
        for node_index, physical_t in enumerate(
                specification['activation_physical_t_nodes']):
            t = torch.full((count,), float(physical_t), device=model.device)
            indices = torch.arange(count, device=model.device)
            noise = model._noise_rows(
                count, clean.shape[1], 'f2k_activation_noise',
                int(specification['noise_seed']) + node_index, indices)
            state = model.corrupt_vocab_state(clean, t, noise=noise)
            model.local_posterior(state, t, None, dropout_enabled=False)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    total_square = 0.0
    total_count = 0
    for index, values in enumerate(per_block):
        rms = math.sqrt(values['square'] / max(values['count'], 1))
        rows.append({'block': index, 'activation_rms': rms})
        total_square += values['square']
        total_count += values['count']
    return {
        'activation_norm': math.sqrt(total_square / max(total_count, 1)),
        'definition': 'RMS hidden activation over all transformer blocks and fixed physical-t nodes',
        'per_block': rows,
        'completed_updates': int(step),
    }


def _update_diagnostic(run_root: Path | None, step: int) -> dict:
    if run_root is None:
        return {}
    path = run_root / 'f2k_optimizer_update_events.jsonl'
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding='utf-8').splitlines()):
        event = json.loads(line)
        if int(event.get('completed_updates', -1)) == step:
            return event
    return {}


@torch.no_grad()
def _checkpoint_update_diagnostic(model, checkpoint: Path, step: int) -> dict:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    states = payload.get('optimizer_states', [])
    if not states:
        return {}
    optimizer = model.configure_optimizers()
    optimizer.load_state_dict(states[0])
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    update_square = 0.0
    before_square = 0.0
    parameter_count = 0
    group_update_square = {}
    group_parameter_count = {}
    group_lrs = {}
    for group in optimizer.param_groups:
        name = str(group.get('name', 'unnamed'))
        beta1, beta2 = (float(value) for value in group['betas'])
        epsilon = float(group['eps'])
        learning_rate = float(group['lr'])
        if float(group.get('weight_decay', 0.0)) != 0.0:
            raise RuntimeError('Patch update diagnostic requires zero weight decay.')
        group_lrs[name] = learning_rate
        group_update_square[name] = 0.0
        group_parameter_count[name] = 0
        for parameter in group['params']:
            if id(parameter) not in backbone_ids:
                continue
            state = optimizer.state.get(parameter, {})
            if 'exp_avg' not in state or 'exp_avg_sq' not in state:
                continue
            optimizer_step = state.get('step', step)
            if isinstance(optimizer_step, torch.Tensor):
                optimizer_step = float(optimizer_step)
            bias1 = 1.0 - beta1 ** float(optimizer_step)
            bias2 = 1.0 - beta2 ** float(optimizer_step)
            exp_avg = state['exp_avg'].to(parameter.device).float()
            exp_avg_sq = state['exp_avg_sq'].to(parameter.device).float()
            delta = -(learning_rate / bias1) * exp_avg / (
                exp_avg_sq.sqrt() / math.sqrt(bias2) + epsilon)
            before = parameter.detach().float() - delta
            square = float(delta.double().square().sum())
            update_square += square
            before_square += float(before.double().square().sum())
            parameter_count += parameter.numel()
            group_update_square[name] += square
            group_parameter_count[name] += parameter.numel()
    update_norm = math.sqrt(update_square)
    parameter_norm_before = math.sqrt(before_square)
    return {
        'completed_updates': int(step),
        'parameter_count': int(parameter_count),
        'parameter_norm_before': parameter_norm_before,
        'update_norm': update_norm,
        'update_rms': update_norm / math.sqrt(max(parameter_count, 1)),
        'update_weight_ratio': update_norm / max(
            parameter_norm_before, 1.0e-6 * math.sqrt(max(parameter_count, 1))),
        'group_learning_rates': group_lrs,
        'group_update_norms': {
            name: math.sqrt(value) for name, value in group_update_square.items()},
        'group_update_rms': {
            name: math.sqrt(group_update_square[name]
                            / max(group_parameter_count[name], 1))
            for name in group_update_square},
        'definition': 'exact AdamW delta reconstructed from checkpoint post-step moments; weight_decay=0',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--audit-config', type=Path, required=True)
    parser.add_argument('--line', choices=('f', 'p'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--model-type', choices=('online', 'target_ema', 'eval_ema'), required=True)
    parser.add_argument('--scope', choices=('full', 'trajectory'), default='full')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    bindings = _read_json(args.bindings)
    audit = _read_yaml(args.audit_config)
    model, repo = _compose_model(args, bindings, audit)
    local_count = int(audit['heldout_local']['first_sequences'])
    validation_path = (
        bindings['data'].get('owt_validation')
        or bindings['data']['owt_train'])
    tokens = _load_tokens(repo, Path(validation_path), local_count)
    payload = {
        'audit_id': audit['audit_id'],
        'line': args.line.upper(),
        'step': int(args.step),
        'model_type': args.model_type,
        'checkpoint': str(args.checkpoint.resolve()),
        'time_direction': audit['time_direction'],
        'local_curve': _local_curve(model, tokens, audit, int(args.step)),
        'controller': _controller_state(model, int(args.step)),
    }
    if args.scope == 'full':
        map_specification = audit['map_validation']
        if bool(map_specification.get('patch_fixed_data_source', False)):
            map_count = (
                4 * int(map_specification['continuous_rows_per_bucket'])
                + 4 * len(map_specification['deployment_grids'])
                * int(map_specification['deployment_rows_per_interval']))
        else:
            map_count = (
                int(map_specification['sequences_per_bucket'])
                * len(map_specification['buckets']))
        map_tokens = tokens[:map_count]
        payload['map_metrics'], tangent_norm = _map_metrics(
            model, map_tokens, audit, int(args.step))
        payload['JVP_tangent_norm'] = tangent_norm
        if args.model_type == str(audit['sc_validation']['model_type']):
            sc_tokens = tokens[:int(audit['sc_validation']['first_sequences'])]
            payload['SC_metrics'] = _sc_metrics(
                model, sc_tokens, audit, int(args.step))
        else:
            payload['SC_metrics'] = []
        if args.model_type == 'online':
            payload['parameter_diagnostics'] = _parameter_diagnostics(model)
            probe_step = int(args.step)
            terminal_step = min(
                int(model.sampler_contract['local']['stages'][-1]['end_step']),
                int(model.sampler_contract['map']['stages'][-1]['end_step']))
            if probe_step == terminal_step:
                # The terminal checkpoint has no next update; probe the last
                # contracted sampling stage using its final valid step.
                probe_step -= 1
            optimization = _gradient_cosine_probe(
                model, probe_step,
                int(audit['optimization_validation']['noise_seed']))
            optimization['sampling_stage_completed_updates'] = probe_step
            optimization.update(_activation_probe(
                model, tokens, int(args.step), audit))
            optimization.update(_update_diagnostic(
                args.run_root, int(args.step)))
            if optimization.get('update_weight_ratio') is None:
                optimization.update(_checkpoint_update_diagnostic(
                    model, args.checkpoint, int(args.step)))
            payload['optimization_diagnostics'] = optimization
    _atomic_json(args.output, payload)


if __name__ == '__main__':
    main()
