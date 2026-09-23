#!/usr/bin/env python3
"""Run one declared 20k generation audit job from a full-state checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--line', choices=('f', 'p'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-type', default='eval_ema')
    parser.add_argument('--mode', choices=(
        'finite', 'legacy_rolling_T1', 'matched_rolling_T095',
        'matched_cold_T095', 'canonical_fixed_budget_T095'), required=True)
    parser.add_argument('--nfe', type=int, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--grid', default='[0.0,0.95]')
    parser.add_argument('--canonical-k', type=int, choices=(1, 2, 4))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples % args.batch_size:
        raise ValueError('samples must be divisible by batch-size')

    bindings = json.loads(args.bindings.read_text(encoding='utf-8'))
    repo = Path(bindings['repo_root']).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ['FLM_TOKENIZER_PATH'] = str(Path(bindings['data']['tokenizer']).resolve())
    os.environ['FLM_PACKED_DATA_DIR'] = str(Path(bindings['data']['owt_train']).resolve())
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'

    from hydra import compose, initialize_config_dir
    import algorithm_registry
    import dataloader
    import main as flm_main
    import utils

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overrides = [
        f'algo=task1_tvm_50k_final_{args.line}',
        'model=small_128', 'data=openwebtext_327m_packed',
        'mode=sample_eval',
        f'algo.task1_tvm_contract_dir={args.contract_dir.resolve()}',
        f'algo.task1_eval_weight_role={args.model_type}',
        f'algo.task1_eval_mode={args.mode}',
        f'algo.task1_eval_physical_grid={args.grid}',
        f'seed={args.seed}',
        f'data.packed_dir={Path(bindings["data"]["owt_train"]).resolve()}',
        f'data.cache_dir={Path(bindings["data"]["owt_train"]).resolve()}',
        f'data.tokenizer_name_or_path={Path(bindings["data"]["tokenizer"]).resolve()}',
        'loader.global_batch_size=256', 'loader.batch_size=32',
        f'loader.eval_global_batch_size={args.batch_size}',
        f'loader.eval_batch_size={args.batch_size}',
        'trainer.devices=1', 'trainer.num_nodes=1',
        'trainer.accumulate_grad_batches=8', 'strategy=single_device',
        f'sampling.num_sample_batches={args.samples // args.batch_size}',
        f'sampling.steps=[{args.nfe}]',
        f'sampling.task1_initial_noise_seed={args.seed}',
        'sampling.task1_initial_noise_schedule=base_seed_plus_sample_index',
        'sampling.temperature=1.0',
        f'eval.checkpoint_path={args.checkpoint.resolve()}',
        f'eval.disable_ema={str(args.model_type != "eval_ema").lower()}',
        'eval.compute_generative_perplexity=true',
        f'eval.gen_ppl_eval_model_name_or_path={Path(bindings["data"]["evaluator"]).resolve()}',
        'eval.gen_ppl_protocol_id=owt128-gpt2large-genppl-v1',
        'eval.gen_ppl_comparable=true',
        f'eval.gen_ppl_formal_sample_count={args.samples}',
        f'eval.generated_samples_path={args.output.resolve()}',
        f'checkpointing.save_dir={args.output.parent.resolve()}',
        f'hydra.run.dir={args.output.parent.resolve()}',
    ]
    if args.canonical_k is not None:
        overrides.append(
            f'+algo.task1_eval_canonical_k={args.canonical_k}')
    with initialize_config_dir(version_base=None, config_dir=str(repo / 'configs')):
        config = compose(config_name='config', overrides=overrides)
    tokenizer = dataloader.get_tokenizer(config)
    model_class = algorithm_registry.get_algorithm_class(config.algo.name)
    flm_main._generate_samples(
        model_class, config, utils.get_logger(__name__), tokenizer)


if __name__ == '__main__':
    main()
