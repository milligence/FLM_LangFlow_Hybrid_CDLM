"""Run already-selected grid jobs in a disjoint side lane of the same search."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

def grid_id(grid):
    import hashlib
    return hashlib.sha256(json.dumps(list(grid), separators=(',', ':')).encode()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weight', choices=('online', 'eval_ema'), required=True)
    parser.add_argument('--nfe', type=int, choices=(2, 4), required=True)
    parser.add_argument('--indices', required=True, help='Comma-separated Stage2 pool indices')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    pool = json.loads((root / 'candidates_stage2.json').read_text())[str(args.nfe)]
    indices = [int(i) for i in args.indices.split(',')]
    chosen = [pool[i] for i in indices]
    if len(set(indices)) != len(indices) or len({tuple(g) for g in chosen}) != len(chosen):
        raise RuntimeError('Duplicate side-lane index or grid')
    jobs = []
    for g in chosen:
        name = f'S1__{args.nfe}nfe__{args.weight}__{grid_id(g)}__n128'
        jobs.append({'name': name, 'mode': 'finite', 'weight': args.weight,
                     'nfe': args.nfe, 'grid': g, 'samples': 128,
                     'seed': 314159, 'batch_size': 4})
    print(json.dumps([j['name'] for j in jobs]), flush=True)
    if args.plan_only:
        return
    bindings = json.loads(args.bindings.read_text())
    sys.path.insert(0, bindings['repo_root'])
    from task1_tvm_50k_final_adapter import _run_eval_job
    adapter_args = argparse.Namespace(output=root, contract_dir=args.contract, line='f')
    train = {'seed': 20260921, 'global_batch': 256}
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    for job in jobs:
        samples = root / 'evaluation' / job['name'] / 'samples.json'
        if samples.is_file():
            print('SKIP_COMPLETE', job['name'], flush=True)
            continue
        claim = root / (job['name'] + '.side_claim')
        with claim.open('x') as handle:
            handle.write(str(os.getpid()))
        start = time.perf_counter()
        code = _run_eval_job(adapter_args, bindings, train, args.checkpoint, job)
        elapsed = time.perf_counter() - start
        row = {'name': job['name'], 'exit_code': code, 'seconds': elapsed,
               'nfe': job['nfe'], 'samples': 128, 'seed': job['seed'],
               'grid': job['grid'], 'weight': job['weight'], 'side_lane': True}
        with (root / 'tasks.jsonl').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row) + '\n')
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
        if code or not samples.is_file():
            raise RuntimeError(f'Side-lane job failed: {job["name"]}, exit={code}')
        print(f'SIDE_COMPLETE {job["name"]} seconds={elapsed:.2f}', flush=True)


if __name__ == '__main__':
    main()
