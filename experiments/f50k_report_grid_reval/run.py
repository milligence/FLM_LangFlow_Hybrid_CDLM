"""Read-only F checkpoint evaluations for the report's fixed four-step grid."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time


def jobs(plan: dict, checkpoint_dir: Path, output: Path):
    for step in plan["steps"]:
        checkpoint = checkpoint_dir / f"step_{step:06d}.ckpt"
        sidecar = checkpoint_dir / f"step_{step:06d}.ckpt.sha256"
        for weight in plan["weights"]:
            name = f"step_{step:06d}__{weight}__finite_four_t0575__n128"
            yield {"name": name, "checkpoint": checkpoint, "sidecar": sidecar,
                   "output": output / "evaluation" / name / "samples.json",
                   "weight": weight, "mode": "finite", "nfe": 4,
                   "grid": plan["grid"], "samples": plan["samples"],
                   "seed": plan["seed"], "batch_size": plan["batch_size"]}


def run_one(item: dict, bindings: dict, contract: Path, output: Path):
    sample_path = item["output"]
    if sample_path.is_file():
        payload = json.loads(sample_path.read_text())
        if payload["num_samples"] != 128 or payload["checkpoint_global_step"] != int(item["name"][5:11]):
            raise RuntimeError(f"Existing result has wrong step/sample count: {sample_path}")
        return item["name"], "reused"
    if not item["checkpoint"].is_file() or not item["sidecar"].is_file():
        raise RuntimeError(f"Checkpoint or SHA sidecar missing: {item['checkpoint']}")
    sys.path.insert(0, bindings["repo_root"])
    from task1_tvm_50k_final_adapter import _run_eval_job
    call = argparse.Namespace(output=output, contract_dir=contract, line="f")
    train = {"seed": 20260921, "global_batch": 256}
    start = time.time()
    result = _run_eval_job(call, bindings, train, item["checkpoint"], item)
    if result or not sample_path.is_file():
        raise RuntimeError(f"Evaluation failed: {item['name']} exit={result}")
    payload = json.loads(sample_path.read_text())
    if payload["num_samples"] != 128 or payload["checkpoint_global_step"] != int(item["name"][5:11]):
        raise RuntimeError(f"Result has wrong step/sample count: {item['name']}")
    return item["name"], round(time.time() - start, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["grid"] == [0.0, 0.2375, 0.575, 0.7125, 0.95]
    assert plan["samples"] == 128 and plan["seed"] == 424242
    assert plan["weights"] == ["online", "eval_ema"]
    work = list(jobs(plan, args.checkpoint_dir, args.output))
    print(json.dumps({"jobs": [x["name"] for x in work], "count": len(work)}), flush=True)
    if args.plan_only:
        return
    bindings = json.loads(args.bindings.read_text())
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    os.environ.pop("TASK1_F_SAMPLER_PROFILE", None)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_one, item, bindings, args.contract, args.output): item
                   for item in work}
        for future in as_completed(futures):
            name, duration = future.result()
            print(f"COMPLETE {name} {duration}", flush=True)
    rows = []
    for item in work:
        payload = json.loads(item["output"].read_text())
        quality = payload["sample_quality"]
        ppl = float(payload["generative_ppl"])
        if not math.isfinite(ppl):
            raise RuntimeError(f"Nonfinite PPL: {item['name']}")
        rows.append({"step": int(item["name"][5:11]), "weight": item["weight"],
                     "samples": payload["num_samples"], "ppl": ppl,
                     "entropy": quality["mean_sample_unigram_entropy_nats"],
                     "distinct_1": quality["distinct_1"],
                     "distinct_2": quality["distinct_2"],
                     "raw": str(item["output"].relative_to(args.output))})
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "EVAL_COMPLETE").write_text(f"{len(rows)}/{len(work)}\n")
    print(f"EVAL_COMPLETE {len(rows)}/{len(work)}", flush=True)


if __name__ == "__main__":
    main()
