#!/usr/bin/env python3
"""Short matched local-only throughput probe for one microbatch size."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

import lightning as L
import torch
from hydra import compose, initialize_config_dir

import algorithm_registry
import dataloader
from packed_dataset import PackedTokenDataset


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def config_for(microbatch, output_dir):
    accumulation = 256 // microbatch
    with initialize_config_dir(
            version_base=None, config_dir=str(SOURCE_ROOT / 'configs')):
        return compose(config_name='config', overrides=[
            'mode=train', 'seed=1', 'data=openwebtext_327m_packed',
            'model=small_128_posterior_tvm',
            'algo=task1_posterior_tvm_local_only',
            'strategy=single_device', 'loader.global_batch_size=256',
            f'loader.batch_size={microbatch}',
            'loader.eval_global_batch_size=2', 'loader.eval_batch_size=2',
            'loader.num_workers=0', 'trainer.devices=1',
            f'trainer.accumulate_grad_batches={accumulation}',
            'training.ema=0.999', 'training.loss_precision=float32',
            'optim.lr=3e-4', 'optim.beta1=0.9', 'optim.beta2=0.95',
            'optim.eps=1e-8', 'optim.weight_decay=0.01',
            f'checkpointing.save_dir={output_dir}',
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--measure', type=int, default=30)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if 256 % args.microbatch:
        raise ValueError('microbatch must divide global batch 256')

    config = config_for(args.microbatch, Path(args.output).parent)
    L.seed_everything(1)
    tokenizer = dataloader.get_tokenizer(config)
    model_class = algorithm_registry.get_algorithm_class(config.algo.name)
    model = model_class(config, tokenizer=tokenizer).to('cuda')
    model.log = lambda *unused_args, **unused_kwargs: None
    trainer = SimpleNamespace(
        global_step=3000, accumulate_grad_batches=256 // args.microbatch,
        is_global_zero=True)
    model._trainer = trainer
    model.train()
    model.training_target_ema.eval()
    if model.ema is None:
        raise RuntimeError('Eval EMA is required')
    model.ema.move_shadow_params_to_device('cuda')
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config[0][0]
    scheduler = optimizer_config[1][0]['scheduler']

    dataset = PackedTokenDataset(config.data.packed_dir, 'train')
    tokens = torch.stack([
        dataset[index]['input_ids'] for index in range(256)
    ]).pin_memory()
    valid = torch.ones_like(tokens, dtype=torch.bool).pin_memory()

    total_steps = args.warmup + args.measure
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(total_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for accumulation_step in range(trainer.accumulate_grad_batches):
            start = accumulation_step * args.microbatch
            stop = start + args.microbatch
            batch_tokens = tokens[start:stop].to('cuda', non_blocking=True)
            batch_valid = valid[start:stop].to('cuda', non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model._loss(
                    batch_tokens, batch_valid,
                    current_accumulation_step=accumulation_step,
                    train_mode=True).loss
            (loss / trainer.accumulate_grad_batches).backward()
        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.ema.update(model._get_parameters())
        trainer.global_step += 1
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if index >= args.warmup:
            durations.append(duration)
        print(json.dumps({
            'microbatch': args.microbatch, 'iteration': index + 1,
            'duration_seconds': duration,
        }), flush=True)

    median = statistics.median(durations)
    result = {
        'status': 'completed', 'microbatch': args.microbatch,
        'accumulation': trainer.accumulate_grad_batches,
        'global_batch': 256, 'warmup_steps': args.warmup,
        'measured_steps': args.measure,
        'seconds_per_step_mean': statistics.mean(durations),
        'seconds_per_step_p50': median,
        'seconds_per_step_p90': percentile(durations, 0.90),
        'seconds_per_step_p99': percentile(durations, 0.99),
        'tokens_per_second': 256 * 128 / statistics.mean(durations),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'device': torch.cuda.get_device_name(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
