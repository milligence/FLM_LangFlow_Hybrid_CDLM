#!/usr/bin/env python3
"""Exact Posterior-TVM engineering benchmark and equivalence harness."""

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

import lightning as L
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

import algorithm_registry
import dataloader
from packed_dataset import PackedTokenDataset
from scripts.profile_tvm_vs_multistep import (
    BatchSource,
    optimizer_objects,
    profiler_trace,
    quantile,
    write_csv,
)
from task1_tvm_ce import (
    calibrated_relative_distributions,
    finite_map_update,
    tvm_quantity_from_logits_jvp,
)


VARIANTS = {
    'reference': ('reference', 'reference', 'no_grad'),
    'bmm': ('bmm', 'reference', 'no_grad'),
    'detached_metrics': ('reference', 'detached', 'no_grad'),
    'inference_query': ('reference', 'reference', 'inference_clone'),
    'bmm_detached': ('bmm', 'detached', 'no_grad'),
    'exact_combo': ('bmm', 'detached', 'inference_clone'),
}


def config_for(variant, microbatch, output_dir):
    if 256 % microbatch:
        raise ValueError('microbatch must divide scientific global batch 256.')
    attention, metrics, target = VARIANTS[variant]
    with initialize_config_dir(
            version_base=None, config_dir=str(SOURCE_ROOT / 'configs')):
        return compose(config_name='config', overrides=[
            'mode=train', 'seed=1', 'data=openwebtext_327m_packed',
            'model=small_128_posterior_tvm',
            'algo=task1_posterior_tvm_a', 'strategy=single_device',
            'loader.global_batch_size=256', f'loader.batch_size={microbatch}',
            'loader.eval_global_batch_size=2', 'loader.eval_batch_size=2',
            'loader.num_workers=4', 'trainer.devices=1',
            f'trainer.accumulate_grad_batches={256 // microbatch}',
            'training.ema=0.999', 'training.loss_precision=float32',
            'optim.lr=3e-4', 'optim.beta1=0.9', 'optim.beta2=0.95',
            'optim.eps=1e-8', 'optim.weight_decay=0.01',
            f'model.attention_jvp_backend={attention}',
            f'algo.posterior_tvm_metrics_backend={metrics}',
            f'algo.posterior_tvm_target_query_mode={target}',
            f'checkpointing.save_dir={output_dir}',
        ])


def make_model(variant, microbatch, output_dir):
    config = config_for(variant, microbatch, output_dir)
    L.seed_everything(1)
    tokenizer = dataloader.get_tokenizer(config)
    model_class = algorithm_registry.get_algorithm_class(config.algo.name)
    model = model_class(config, tokenizer=tokenizer).to('cuda')
    model.log = lambda *unused_args, **unused_kwargs: None
    trainer = SimpleNamespace(
        global_step=3000, accumulate_grad_batches=256 // microbatch,
        is_global_zero=True)
    model._trainer = trainer
    model.train()
    model.training_target_ema.eval()
    model.ema.move_shadow_params_to_device('cuda')
    return config, model, trainer


def set_variant(model, variant):
    attention, metrics, target = VARIANTS[variant]
    for block in model.backbone.blocks:
        block.attention_jvp_backend = attention
    model.metrics_backend = metrics
    model.target_query_mode = target


def reference_inputs(model, tokens, valid, microbatch):
    model.config.loader.global_batch_size = microbatch
    model.config.loader.batch_size = microbatch
    model._trainer.accumulate_grad_batches = 1
    model.map_subset_size = microbatch
    plan = model._global_plan(0)
    return tokens[:microbatch], valid[:microbatch], plan


def capture_exact_path(model, tokens, valid, plan):
    local_shape = (*tokens.shape, model.vocab_size)
    local_noise = model._noise(local_shape, 'local_noise', 0)
    local_state = model.corrupt_vocab_state(
        tokens, plan['local_t'], noise=local_noise)
    local_logits = model._logits(
        model.backbone, local_state, plan['local_t'],
        torch.zeros_like(plan['local_t']))

    map_tokens, state, r, cache, previous_eta, cache_valid = model._map_state(
        tokens, plan, 0)
    s = plan['s'][plan['map_mask']]
    eta = ((s - r) / (1.0 - r).clamp_min(1e-8)).clamp(0.0, 1.0)

    def logits_at_eta(value):
        return model._logits(
            model.backbone, state, r, value, cache=cache,
            previous_eta=previous_eta, valid=cache_valid,
            use_jvp_attn=True)

    map_logits, tangent = torch.func.jvp(
        logits_at_eta, (eta,), (eta * (1.0 - eta),))
    probability, _, gate, quantity = tvm_quantity_from_logits_jvp(
        map_logits, tangent)
    endpoint = finite_map_update(state, probability, r, s)
    terminal = model._target_probabilities(
        endpoint.detach(), s, torch.zeros_like(s)).detach()
    student_log, calibrated_target = calibrated_relative_distributions(
        probability, gate, terminal, model.kappa)
    calibrated_ce = -(calibrated_target * student_log).sum(dim=-1)
    negative_mass = F.relu(-quantity).sum(dim=-1)
    map_loss = (calibrated_ce + 0.1 * negative_mass.square()).mean()
    return {
        'local_logits': local_logits,
        'map_logits': map_logits,
        'jvp_tangent': tangent,
        'R': probability,
        'Q': quantity,
        'qS': student_log.exp(),
        'qT': calibrated_target,
        'map_loss': map_loss,
    }


def tensor_error(reference, candidate):
    difference = (candidate.float() - reference.float()).reshape(-1)
    reference_flat = reference.float().reshape(-1)
    denominator = reference_flat.norm().clamp_min(1e-30)
    return {
        'max_abs_error': float(difference.abs().max()),
        'relative_l2_error': float(difference.norm() / denominator),
    }


def gradient_snapshot(model, loss):
    gradients = torch.autograd.grad(
        loss, model._shared_parameters(), allow_unused=True)
    return tuple(
        None if gradient is None else gradient.detach().float().cpu()
        for gradient in gradients)


def compare_gradients(reference, candidate):
    dot = ref_sq = candidate_sq = difference_sq = 0.0
    for ref, cand in zip(reference, candidate):
        if ref is None and cand is None:
            continue
        if ref is None or cand is None:
            return {'grad_cosine': 0.0, 'relative_grad_norm_difference': math.inf}
        dot += float(torch.sum(ref.double() * cand.double()))
        ref_sq += float(torch.sum(ref.double().square()))
        candidate_sq += float(torch.sum(cand.double().square()))
        difference_sq += float(torch.sum((cand.double() - ref.double()).square()))
    cosine = dot / max(math.sqrt(ref_sq * candidate_sq), 1e-300)
    relative = math.sqrt(difference_sq) / max(math.sqrt(ref_sq), 1e-300)
    return {'grad_cosine': cosine, 'relative_grad_norm_difference': relative}


def run_equivalence(output, candidates, microbatch):
    config, model, _ = make_model('reference', 32, output)
    dataset = PackedTokenDataset(config.data.packed_dir, 'train')
    tokens = torch.stack([
        dataset[index]['input_ids'] for index in range(max(256, microbatch))
    ])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    fixed_tokens, fixed_valid, plan = reference_inputs(
        model, tokens.to('cuda'), valid.to('cuda'), microbatch)
    torch.save({
        'tokens': tokens[:microbatch].cpu(),
        'valid': valid[:microbatch].cpu(),
        'plan': {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in plan.items()},
    }, output / 'reference_batch.pt')
    (output / 'reference_rng.json').write_text(json.dumps({
        'model_seed': 1,
        'plan_seed': int(model.plan_seed),
        'global_step': int(model.global_step),
        'accumulation_step': 0,
        'microbatch': microbatch,
        'sequence_length': 128,
        'map_subset': microbatch,
    }, indent=2) + '\n')

    set_variant(model, 'reference')
    reference_outputs = capture_exact_path(
        model, fixed_tokens, fixed_valid, plan)
    reference_cpu = {
        name: value.detach().cpu() for name, value in reference_outputs.items()
    }
    reference_gradients = gradient_snapshot(model, reference_outputs['map_loss'])
    forward_rows = []
    gradient_rows = []
    for candidate_name in candidates:
        set_variant(model, candidate_name)
        candidate_outputs = capture_exact_path(
            model, fixed_tokens, fixed_valid, plan)
        for tensor_name, candidate in candidate_outputs.items():
            errors = tensor_error(reference_cpu[tensor_name], candidate.detach().cpu())
            forward_rows.append({
                'candidate': candidate_name,
                'tensor': tensor_name,
                **errors,
            })
        candidate_gradients = gradient_snapshot(
            model, candidate_outputs['map_loss'])
        gradient_rows.append({
            'candidate': candidate_name,
            **compare_gradients(reference_gradients, candidate_gradients),
        })
        del candidate_outputs, candidate_gradients
        torch.cuda.empty_cache()
    write_csv(output / 'equivalence_forward.csv', forward_rows)
    write_csv(output / 'equivalence_grad.csv', gradient_rows)


def benchmark_variant(output, variant, microbatch, warmup, measured):
    config, model, trainer = make_model(variant, microbatch, output)
    source = BatchSource(config, microbatch, cached=True)
    optimizer, scheduler = optimizer_objects(model)
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(warmup + measured):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        finite = True
        for accumulation_step in range(trainer.accumulate_grad_batches):
            tokens, valid = source.next()
            tokens = tokens.to('cuda', non_blocking=True)
            valid = valid.to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model._loss(
                    tokens, valid,
                    current_accumulation_step=accumulation_step,
                    train_mode=True).loss
            finite = finite and bool(torch.isfinite(loss))
            (loss / trainer.accumulate_grad_batches).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.backbone.parameters(), 1.0)
        finite = finite and bool(torch.isfinite(grad_norm))
        optimizer.step()
        scheduler.step()
        model.ema.update(model._get_parameters())
        model._update_training_target_ema()
        trainer.global_step += 1
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if iteration >= warmup:
            durations.append(duration)
        if not finite:
            raise RuntimeError(f'nonfinite event in {variant} iteration {iteration + 1}')
    row = {
        'variant': variant,
        'microbatch': microbatch,
        'accumulation': trainer.accumulate_grad_batches,
        'warmup_steps': warmup,
        'measured_steps': measured,
        'mean_seconds': statistics.mean(durations),
        'p50_seconds': statistics.median(durations),
        'p90_seconds': quantile(durations, 0.90),
        'p99_seconds': quantile(durations, 0.99),
        'tokens_per_second': 256 * 128 / statistics.mean(durations),
        'map_samples_per_second': 96 / statistics.mean(durations),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'status': 'passed',
    }
    path = output / 'patch_benchmark_rows'
    path.mkdir(exist_ok=True)
    write_csv(path / f'{variant}_mb{microbatch}.csv', [row])
    print(json.dumps(row), flush=True)


def stability_run(output, variant, steps, microbatch):
    config, model, trainer = make_model(variant, microbatch, output)
    source = BatchSource(config, microbatch, cached=False)
    optimizer, scheduler = optimizer_objects(model)
    latest_logs = {}

    def capture_log(name, value, **unused_kwargs):
        if torch.is_tensor(value):
            latest_logs[name] = float(value.detach().float().mean().cpu())

    model.log = capture_log
    rows = []
    last_gpu_utilization = None
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        latest_logs.clear()
        losses = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        for accumulation_step in range(trainer.accumulate_grad_batches):
            tokens, valid = source.next()
            tokens = tokens.to('cuda', non_blocking=True)
            valid = valid.to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model._loss(
                    tokens, valid,
                    current_accumulation_step=accumulation_step,
                    train_mode=True).loss
            losses.append(float(loss.detach().cpu()))
            (loss / trainer.accumulate_grad_batches).backward()
        gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.backbone.parameters()
            if parameter.grad is not None)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.backbone.parameters(), 1.0)
        loss_finite = all(math.isfinite(value) for value in losses)
        if not loss_finite or not gradients_finite or not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f'nonfinite stability event at step {step}')
        optimizer.step()
        scheduler.step()
        model.ema.update(model._get_parameters())
        model._update_training_target_ema()
        trainer.global_step += 1
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started
        if step == 1 or step % 10 == 0:
            utilization = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'], check=False,
                capture_output=True, text=True)
            if utilization.returncode == 0:
                try:
                    last_gpu_utilization = int(
                        utilization.stdout.strip().splitlines()[0])
                except (IndexError, ValueError):
                    last_gpu_utilization = None
        recent_seconds = [row['seconds'] for row in rows[-49:]]
        rolling_median = statistics.median(recent_seconds + [step_seconds])
        rows.append({
            'step': step,
            'seconds': step_seconds,
            'rolling_median_50_seconds': rolling_median,
            'loss': statistics.mean(losses),
            'local_ce': latest_logs.get('posterior_tvm/local_ce'),
            'map_loss': latest_logs.get('posterior_tvm/map_loss'),
            'map_kl': latest_logs.get('posterior_tvm/map/calibrated_kl_mean'),
            'raw_residual_l1': latest_logs.get('posterior_tvm/map/raw_l1_mean'),
            'gradient_norm': float(grad_norm.detach().cpu()),
            'nonfinite_count': 0,
            'gpu_utilization_percent': last_gpu_utilization,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        })
        if step % 10 == 0:
            print(json.dumps(rows[-1]), flush=True)
    write_csv(output / 'final_1000step_stability.csv', rows)
    checkpoint = output / 'stability_resume_checkpoint.pt'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'evaluation_ema': model.ema.state_dict(),
        'global_step': trainer.global_step,
        'variant': variant,
    }, checkpoint)
    del model, optimizer, scheduler, source
    gc.collect()
    torch.cuda.empty_cache()

    resumed_config, resumed, resumed_trainer = make_model(
        variant, microbatch, output)
    resumed_optimizer, resumed_scheduler = optimizer_objects(resumed)
    payload = torch.load(checkpoint, map_location='cuda', weights_only=False)
    resumed.load_state_dict(payload['model'])
    resumed_optimizer.load_state_dict(payload['optimizer'])
    resumed_scheduler.load_state_dict(payload['scheduler'])
    resumed.ema.load_state_dict(payload['evaluation_ema'])
    resumed_trainer.global_step = int(payload['global_step'])
    loaded_global_step = resumed_trainer.global_step
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    resumed_source = BatchSource(resumed_config, microbatch, cached=False)
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_losses = []
    for accumulation_step in range(resumed_trainer.accumulate_grad_batches):
        tokens, valid = resumed_source.next()
        tokens = tokens.to('cuda', non_blocking=True)
        valid = valid.to('cuda', non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            resumed_loss = resumed._loss(
                tokens, valid,
                current_accumulation_step=accumulation_step,
                train_mode=True).loss
        resumed_losses.append(float(resumed_loss.detach().cpu()))
        (resumed_loss / resumed_trainer.accumulate_grad_batches).backward()
    resumed_gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in resumed.backbone.parameters()
        if parameter.grad is not None)
    resumed_grad_norm = torch.nn.utils.clip_grad_norm_(
        resumed.backbone.parameters(), 1.0)
    resume_step_finite = (
        all(math.isfinite(value) for value in resumed_losses)
        and resumed_gradients_finite
        and bool(torch.isfinite(resumed_grad_norm)))
    if not resume_step_finite:
        raise RuntimeError('nonfinite event in post-resume optimizer step')
    resumed_optimizer.step()
    resumed_scheduler.step()
    resumed.ema.update(resumed._get_parameters())
    resumed._update_training_target_ema()
    resumed_trainer.global_step += 1
    (output / 'stability_resume_check.json').write_text(json.dumps({
        'checkpoint': str(checkpoint),
        'loaded_global_step': loaded_global_step,
        'expected_global_step': 3000 + steps,
        'model_state_loaded': True,
        'optimizer_state_loaded': True,
        'scheduler_state_loaded': True,
        'evaluation_ema_state_loaded': True,
        'post_resume_optimizer_step_finite': resume_step_finite,
        'post_resume_global_step': resumed_trainer.global_step,
    }, indent=2) + '\n')


def run_final_profile(output, variant, microbatch):
    config, model, trainer = make_model(variant, microbatch, output)
    source = BatchSource(config, microbatch, cached=True)
    profiler_trace(
        model, trainer, source, variant,
        output / 'final_profiler_trace.json', output / 'final_top_ops.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    subparsers = parser.add_subparsers(dest='command', required=True)
    equivalence = subparsers.add_parser('equivalence')
    equivalence.add_argument(
        '--candidates', nargs='+', default=list(VARIANTS)[1:])
    equivalence.add_argument('--microbatch', type=int, default=2)
    benchmark = subparsers.add_parser('benchmark')
    benchmark.add_argument('--variant', choices=VARIANTS, required=True)
    benchmark.add_argument('--microbatch', type=int, default=32)
    benchmark.add_argument('--warmup', type=int, default=30)
    benchmark.add_argument('--measured', type=int, default=100)
    stability = subparsers.add_parser('stability')
    stability.add_argument('--variant', choices=VARIANTS, required=True)
    stability.add_argument('--steps', type=int, default=1000)
    stability.add_argument('--microbatch', type=int, default=32)
    profile = subparsers.add_parser('profile')
    profile.add_argument('--variant', choices=VARIANTS, required=True)
    profile.add_argument('--microbatch', type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == 'equivalence':
        run_equivalence(output, args.candidates, args.microbatch)
    elif args.command == 'benchmark':
        benchmark_variant(
            output, args.variant, args.microbatch, args.warmup, args.measured)
    elif args.command == 'stability':
        stability_run(output, args.variant, args.steps, args.microbatch)
    elif args.command == 'profile':
        run_final_profile(output, args.variant, args.microbatch)


if __name__ == '__main__':
    main()
