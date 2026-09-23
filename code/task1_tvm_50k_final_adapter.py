"""Thin argv adapter from the delivered contract to the existing Hydra app."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


CONTRACT_VERSION = 'task1-tvm-50k-final-v1.0.0'


def _read_yaml(path):
    with path.open('r', encoding='utf-8') as handle:
        value = yaml.safe_load(handle)
    if value.get('contract_version') != CONTRACT_VERSION:
        raise RuntimeError(f'Wrong contract version in {path}.')
    return value


def _training_artifacts(output):
    candidates = (
        output / '.hydra', output / 'checkpoints',
        output / 'run_manifest.json', output / 'eval_queue.jsonl')
    return [path for path in candidates if path.exists()]


def _resolved_contract(
        contract_dir, output, line, train, local_microbatch, map_microbatch):
    resolved = json.loads(json.dumps(train))
    resolved['model']['self_condition']['logical_group_size'] = 32
    resolved['model']['self_condition']['cache_prepass_dropout'] = 0.1
    resolved['resolved_repository_architecture'] = {
        'layernorm_eps': 1.0e-5,
        'safe_time': 'legacy flm_vocab_gaussian_bias clamp to flm_time_eps',
        'qk_rmsnorm': False,
        'attention_softcap': 50.0,
        'jvp_backend': 'explicit layer JVP; compile stable backbone only',
    }
    resolved['runtime_physical_microbatch'] = {
        'local': int(local_microbatch), 'map': int(map_microbatch),
        'global_batch_unchanged': int(train['global_batch']),
        'map_batch_unchanged': int(train['objective']['map_batch']),
    }
    target = output / f'resolved_train_{line}.yaml'
    temporary = target.with_suffix('.yaml.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _hydra_train(args, bindings, train):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = _training_artifacts(output)
    last = output / 'checkpoints' / 'last.ckpt'
    if args.resume == 'never' and artifacts:
        raise RuntimeError(
            'resume=never requires a fresh output directory; found: '
            + ', '.join(str(path) for path in artifacts))
    if args.resume == 'auto':
        if last.is_file():
            checkpoint = last
        elif artifacts:
            raise RuntimeError(
                'Run artifacts exist but no valid rolling full-state checkpoint.')
        else:
            checkpoint = None
    elif args.resume == 'never':
        checkpoint = None
    else:
        checkpoint = Path(args.resume).resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing explicit checkpoint {checkpoint}.')

    line = args.line.lower()
    milestones = train['checkpoint']['save_steps']
    data = bindings['data']
    packed = Path(data['owt_train']).resolve()
    tokenizer = Path(data['tokenizer']).resolve()
    evaluator = Path(
        data.get('evaluator')
        or bindings['legacy']['gpt2large_evaluator']).resolve()
    if (not packed.is_dir() or not tokenizer.exists()
            or not evaluator.is_dir()):
        raise RuntimeError(
            'Resolved packed data/tokenizer/evaluator binding is missing.')
    local_candidates = [int(value) for value in train['oom']['local_candidates']]
    map_candidates = [int(value) for value in train['oom']['map_candidates']]
    local_microbatch = int(train['microbatch'])
    map_microbatch = int(train['map_microbatch'])

    def command_for(local_size, map_size, resume_checkpoint):
        accumulation = int(train['global_batch']) // int(local_size)
        if accumulation * int(local_size) != int(train['global_batch']):
            raise RuntimeError('OOM local microbatch does not divide global batch.')
        return [
        sys.executable, 'main.py', '--config-name=config',
        f'algo=task1_tvm_50k_final_{line}',
        f'+experiment.variant=task1_tvm_50k_final_{line}',
        'model=small_128', 'data=openwebtext_327m_packed',
        f'algo.task1_tvm_contract_dir={args.contract_dir.resolve()}',
        f'algo.task1_map_microbatch_override={int(map_size)}',
        f'seed={int(train["seed"])}',
        f'sampling.task1_initial_noise_seed={int(train["seed"])}',
        f'loader.global_batch_size={int(train["global_batch"])}',
        f'loader.batch_size={int(local_size)}',
        f'loader.eval_batch_size={int(local_size)}',
        f'data.packed_dir={packed}', f'data.cache_dir={packed}',
        f'data.tokenizer_name_or_path={tokenizer}',
        f'eval.gen_ppl_eval_model_name_or_path={evaluator}',
        f'trainer.max_steps={int(train["total_optimizer_steps"])}',
        f'trainer.accumulate_grad_batches={accumulation}',
        'strategy=single_device',
        f'trainer.gradient_clip_val={float(train["optimizer"]["grad_clip"])}',
        'trainer.precision=bf16',
        f'trainer.val_check_interval={500 * accumulation}',
        'trainer.log_every_n_steps=10',
        'trainer.limit_val_batches=16',
        'trainer.num_sanity_val_steps=0',
        f'optim.lr={max(float(point[1]) for point in train["optimizer"]["lr_schedule"])}',
        f'optim.beta1={float(train["optimizer"]["betas"][0])}',
        f'optim.beta2={float(train["optimizer"]["betas"][1])}',
        f'optim.eps={float(train["optimizer"]["eps"])}',
        f'optim.weight_decay={float(train["optimizer"]["weight_decay"])}',
        f'training.ema={float(train["ema"]["eval_decay"])}',
        f'training.finetune_path=',
        f'checkpointing.save_dir={output}',
        f'checkpointing.resume_from_ckpt={str(resume_checkpoint is not None).lower()}',
        f'checkpointing.resume_ckpt_path={resume_checkpoint or last}',
        f'hydra.run.dir={output}',
        'callbacks.checkpoint_every_n_steps.save_top_k=0',
        'callbacks.checkpoint_monitor.save_top_k=0',
        'callbacks.checkpoint_monitor.save_last=false',
        '+callbacks.milestone_checkpoints._target_=experiment_callbacks.MilestoneCheckpointCallback',
        f'+callbacks.milestone_checkpoints.directory={output / "checkpoints"}',
        f'+callbacks.milestone_checkpoints.queue_path={output / "eval_queue.jsonl"}',
        f'+callbacks.milestone_checkpoints.global_batch_size={int(train["global_batch"])}',
        f'+callbacks.milestone_checkpoints.sequence_length={int(train["seq_len"])}',
        f'+callbacks.milestone_checkpoints.milestones={milestones}',
        '+callbacks.milestone_checkpoints.save_weights_only=false',
        '+callbacks.milestone_checkpoints.atomic_milestones=true',
        '+callbacks.milestone_checkpoints.allow_overwrite_milestones=false',
        '+callbacks.milestone_checkpoints.write_sha256=true',
        '+callbacks.milestone_checkpoints.zero_grad_before_save=true',
        f'+callbacks.milestone_checkpoints.last_every_n_steps={int(train["checkpoint"]["rolling_full_state_every_steps"])}',
        '+callbacks.milestone_checkpoints.last_filename=last.ckpt',
        f'+callbacks.milestone_checkpoints.last_keep={int(train["checkpoint"]["rolling_keep_last"])}',
        '+callbacks.milestone_checkpoints.stop_marker_path=',
        '+callbacks.milestone_checkpoints.planned_stop_marker_path=',
        f'+callbacks.milestone_checkpoints.transfer_queue_directory={output / "transfer_queue"}',
        f'+callbacks.milestone_checkpoints.run_directory={output}',
        '+callbacks.milestone_checkpoints.launcher_log_path=',
        f'+callbacks.milestone_checkpoints.arm_id={line.upper()}',
        '+callbacks.throughput._target_=experiment_callbacks.OptimizerStepTimerCallback',
        f'+callbacks.throughput.output_path={output / "throughput.jsonl"}',
        f'+callbacks.throughput.global_batch_size={int(train["global_batch"])}',
        f'+callbacks.throughput.sequence_length={int(train["seq_len"])}',
        '+callbacks.throughput.dataset_tokens=0',
        '+callbacks.throughput.every_n_steps=10',
        '+callbacks.cuda_peak._target_=experiment_callbacks.CUDAPeakMemoryCallback',
        ]

    def newest_committed_checkpoint():
        if last.is_file():
            return last
        candidates = sorted((output / 'checkpoints').glob('step_*.ckpt'))
        return candidates[-1] if candidates else None

    attempt = 0
    while True:
        _resolved_contract(
            args.contract_dir, output, args.line, train,
            local_microbatch, map_microbatch)
        command = command_for(local_microbatch, map_microbatch, checkpoint)
        log_path = output / 'logs' / f'adapter_train_attempt_{attempt:02d}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tail = []
        with log_path.open('w', encoding='utf-8', buffering=1) as handle:
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            assert process.stdout is not None
            for line_text in process.stdout:
                sys.stdout.write(line_text)
                sys.stdout.flush()
                handle.write(line_text)
                tail.append(line_text)
                tail = tail[-200:]
            return_code = process.wait()
        if return_code == 0:
            return 0
        joined = ''.join(tail).lower()
        if not ('out of memory' in joined or 'cuda oom' in joined):
            return return_code
        checkpoint = newest_committed_checkpoint()
        if checkpoint is None:
            raise RuntimeError(
                'CUDA OOM occurred before a committed full-state checkpoint; '
                'cannot replay safely.')
        map_failure = any(marker in joined for marker in (
            'finite_with_eta_jvp', '_map_objective',
            'calibration_map', 'calibration_closure'))
        old_local, old_map = local_microbatch, map_microbatch
        if map_failure and map_candidates.index(map_microbatch) + 1 < len(map_candidates):
            map_microbatch = map_candidates[map_candidates.index(map_microbatch) + 1]
        elif local_candidates.index(local_microbatch) + 1 < len(local_candidates):
            local_microbatch = local_candidates[
                local_candidates.index(local_microbatch) + 1]
        elif map_candidates.index(map_microbatch) + 1 < len(map_candidates):
            map_microbatch = map_candidates[map_candidates.index(map_microbatch) + 1]
        else:
            raise RuntimeError(
                'CUDA OOM persists at the smallest contracted microbatches.')
        event = {
            'attempt': attempt, 'checkpoint': str(checkpoint),
            'local_microbatch_before': old_local,
            'local_microbatch_after': local_microbatch,
            'map_microbatch_before': old_map,
            'map_microbatch_after': map_microbatch,
            'global_batch': int(train['global_batch']),
            'map_batch': int(train['objective']['map_batch']),
            'reason': 'cuda_oom_full_update_replay',
        }
        with (output / 'oom_replay_events.jsonl').open(
                'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, sort_keys=True) + '\n')
        attempt += 1


def _precheck(args, bindings):
    """Run the correctness-only repository preflight and bind its report."""
    target = args.output / 'repository_precheck.json'
    command = [
        sys.executable, '-m', 'task1_tvm_50k_precheck',
        '--contract-dir', str(args.contract_dir.resolve()),
        '--bindings', str(args.bindings.resolve()),
        '--output', str(target.resolve()),
    ]
    return_code = subprocess.call(
        command, cwd=Path(__file__).resolve().parent)
    if return_code == 0:
        bindings['precheck_report'] = str(target.resolve())
        temporary = args.bindings.with_suffix(args.bindings.suffix + '.tmp')
        temporary.write_text(
            json.dumps(bindings, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, args.bindings)
    print(target)
    return return_code


def _run_eval_job(args, bindings, train, checkpoint, job):
    output = args.output.resolve()
    job_dir = output / 'evaluation' / job['name']
    job_dir.mkdir(parents=True, exist_ok=True)
    result = job_dir / 'samples.json'
    if result.is_file():
        return 0
    data = bindings['data']
    evaluator = Path(data.get('evaluator') or bindings['legacy'][
        'gpt2large_evaluator']).resolve()
    tokenizer = Path(data['tokenizer']).resolve()
    packed = Path(data['owt_train']).resolve()
    batch = int(job.get('batch_size', 4))
    samples = int(job['samples'])
    if samples % batch:
        raise RuntimeError('Evaluation samples must divide eval batch size.')
    role = job['weight']
    disable_ema = role != 'eval_ema'
    command = [
        sys.executable, 'main.py', '--config-name=config',
        f'algo=task1_tvm_50k_final_{args.line}',
        'model=small_128', 'data=openwebtext_327m_packed',
        'mode=sample_eval',
        f'algo.task1_tvm_contract_dir={args.contract_dir.resolve()}',
        f'algo.task1_eval_weight_role={role}',
        f'algo.task1_eval_mode={job["mode"]}',
        f'algo.task1_eval_physical_grid={job.get("grid", [0.0, 0.95])}',
        f'seed={int(train["seed"])}',
        f'data.packed_dir={packed}', f'data.cache_dir={packed}',
        f'data.tokenizer_name_or_path={tokenizer}',
        f'loader.global_batch_size={int(train["global_batch"])}',
        f'loader.eval_batch_size={batch}',
        f'sampling.num_sample_batches={samples // batch}',
        f'sampling.steps=[{int(job["nfe"])}]',
        f'sampling.task1_initial_noise_seed={int(job["seed"])}',
        'sampling.task1_initial_noise_schedule=base_seed_plus_sample_index',
        'sampling.temperature=1.0',
        f'eval.checkpoint_path={checkpoint}',
        f'eval.disable_ema={str(disable_ema).lower()}',
        'eval.compute_generative_perplexity=true',
        f'eval.gen_ppl_eval_model_name_or_path={evaluator}',
        'eval.gen_ppl_protocol_id=owt128-gpt2large-genppl-v1',
        'eval.gen_ppl_comparable=true',
        f'eval.gen_ppl_formal_sample_count={samples}',
        f'eval.generated_samples_path={result}',
        f'checkpointing.save_dir={job_dir}',
        f'hydra.run.dir={job_dir}',
    ]
    log_path = job_dir / 'evaluation.log'
    with log_path.open('w', encoding='utf-8', buffering=1) as handle:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line_text in process.stdout:
            sys.stdout.write(line_text)
            sys.stdout.flush()
            handle.write(line_text)
        return process.wait()


def _evaluation_jobs(evaluation, step):
    trend = int(evaluation['trend_samples'])
    formal = int(evaluation['formal_samples'])
    seed = int(evaluation['seed'])
    formal_step = step in {int(value) for value in evaluation['formal_steps']}
    jobs = []

    def add(label, weight, mode, nfe, samples, grid=None):
        jobs.append({
            'name': f'step_{step:06d}__{weight}__{label}__n{samples}',
            'weight': weight, 'mode': mode, 'nfe': nfe,
            'samples': samples, 'grid': grid, 'seed': seed,
        })

    for grid_name, grid in evaluation['finite']['grids'].items():
        for weight in ('online', 'target_ema', 'eval_ema'):
            samples = formal if formal_step and weight == 'eval_ema' else trend
            add(f'finite_{grid_name}', weight, 'finite', len(grid) - 1,
                samples, grid)
    for weight in ('online', 'target_ema', 'eval_ema'):
        add('legacy_rolling_T1', weight, 'legacy_rolling_T1', 128, trend)
    if formal_step:
        add('legacy_rolling_T1_512', 'eval_ema',
            'legacy_rolling_T1', 512, formal)
        add('legacy_rolling_T1_512', 'online',
            'legacy_rolling_T1', 512, trend)
        add('legacy_rolling_T1_512', 'target_ema',
            'legacy_rolling_T1', 512, trend)
        add('matched_rolling_T095', 'eval_ema',
            'matched_rolling_T095', 512, trend)
        add('matched_cold_T095', 'eval_ema',
            'matched_cold_T095', 128, trend)
    elif step in {5000, 15000, 40000}:
        add('legacy_rolling_T1_512', 'eval_ema',
            'legacy_rolling_T1', 512, trend)
    if step in {6000, 10000, 20000, 30000, 50000}:
        add('canonical_fixed_budget_T095', 'target_ema',
            'canonical_fixed_budget_T095', 128, trend)
    return jobs


def _evaluate(args, bindings, train):
    evaluation = _read_yaml(args.contract_dir / 'eval.yaml')
    declared = [int(value) for value in evaluation['checkpoint_steps']]
    requested = (
        declared if not args.steps else
        [int(value) for value in args.steps.split(',') if value])
    unknown = sorted(set(requested) - set(declared))
    if unknown:
        raise RuntimeError(f'Undeclared evaluation checkpoints: {unknown}.')
    for step in requested:
        checkpoint = args.output / 'checkpoints' / f'step_{step:06d}.ckpt'
        if not checkpoint.is_file():
            raise RuntimeError(f'Missing full-state checkpoint {checkpoint}.')
        for job in _evaluation_jobs(evaluation, step):
            return_code = _run_eval_job(
                args, bindings, train, checkpoint, job)
            if return_code:
                return return_code
    if args.line == 'f':
        return _evaluate_baseline(args, bindings, evaluation)
    return 0


def _evaluate_baseline(args, bindings, evaluation):
    checkpoint_value = bindings['legacy'].get('mse50k_eval_checkpoint')
    if not checkpoint_value:
        raise RuntimeError('Missing bound MSE50k baseline checkpoint.')
    checkpoint = Path(checkpoint_value).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f'Missing baseline checkpoint {checkpoint}.')
    output = args.output / 'evaluation' / 'baseline_mse50k'
    output.mkdir(parents=True, exist_ok=True)
    data = bindings['data']
    evaluator = Path(data.get('evaluator') or bindings['legacy'][
        'gpt2large_evaluator']).resolve()
    tokenizer = Path(data['tokenizer']).resolve()
    packed = Path(data['owt_train']).resolve()
    samples = int(evaluation['formal_samples'])
    batch = 4
    grid_path = output / 'matched_t095_grid.json'
    if not grid_path.exists():
        query = [0.95 * index / 512.0 for index in range(512)]
        grid_path.write_text(json.dumps({
            'endpoint_physical_t': 0.95,
            'tau_pairing': 'official inverse LUT',
            'schedules': {'matched_t095': {'t_query': query}},
        }, indent=2) + '\n', encoding='utf-8')
    jobs = (
        ('legacy_rolling_T1_512',
         'physical_time_uniform_official_inverse_lut', None, None),
        ('matched_rolling_T095_512',
         'custom_physical_time_nodes', grid_path, 'matched_t095'),
    )
    for name, grid_mode, custom_path, schedule in jobs:
        job_dir = output / name
        result = job_dir / 'samples.json'
        if result.is_file():
            continue
        job_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, 'main.py', '--config-name=config',
            'algo=langflow_hybrid_task1_a', 'model=small_128',
            'data=openwebtext_327m_packed', 'mode=sample_eval',
            f'data.packed_dir={packed}', f'data.cache_dir={packed}',
            f'data.tokenizer_name_or_path={tokenizer}',
            f'loader.eval_batch_size={batch}',
            f'sampling.num_sample_batches={samples // batch}',
            'sampling.steps=[512]',
            f'sampling.task1_time_grid={grid_mode}',
            f'sampling.task1_initial_noise_seed={int(evaluation["seed"])}',
            'sampling.task1_initial_noise_schedule=base_seed_plus_sample_index',
            'sampling.temperature=1.0',
            f'eval.checkpoint_path={checkpoint}', 'eval.disable_ema=false',
            'eval.compute_generative_perplexity=true',
            f'eval.gen_ppl_eval_model_name_or_path={evaluator}',
            'eval.gen_ppl_protocol_id=owt128-gpt2large-genppl-v1',
            'eval.gen_ppl_comparable=true',
            f'eval.gen_ppl_formal_sample_count={samples}',
            f'eval.generated_samples_path={result}',
            f'checkpointing.save_dir={job_dir}',
            f'hydra.run.dir={job_dir}',
        ]
        if custom_path is not None:
            command.extend((
                f'sampling.task1_custom_time_grid_path={custom_path}',
                f'sampling.task1_custom_time_grid_schedule={schedule}',
            ))
        with (job_dir / 'evaluation.log').open(
                'w', encoding='utf-8', buffering=1) as handle:
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parent,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            assert process.stdout is not None
            for line_text in process.stdout:
                sys.stdout.write(line_text)
                sys.stdout.flush()
                handle.write(line_text)
            return_code = process.wait()
        if return_code:
            return return_code
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'evaluate', 'precheck'))
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--line', choices=('f', 'p'), required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', default='auto')
    parser.add_argument('--steps')
    args = parser.parse_args()
    bindings = json.loads(args.bindings.read_text(encoding='utf-8'))
    train = _read_yaml(args.contract_dir / f'train_{args.line}.yaml')
    if args.action == 'precheck':
        return _precheck(args, bindings)
    if args.action == 'evaluate':
        return _evaluate(args, bindings, train)
    return _hydra_train(args, bindings, train)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f'BLOCKED: {error}', file=sys.stderr)
        raise SystemExit(2)
