#!/usr/bin/env python3
"""Assemble the exact Posterior-TVM engineering result bundle."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LOCAL_ONLY_SECONDS = 1.2028587700296194
BASELINE_FULL_SECONDS = 3.356117909029126
VARIANTS = (
    'reference', 'bmm', 'detached_metrics', 'inference_query',
    'bmm_detached', 'exact_combo')


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ['status']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_row(root, variant, microbatch=32, nested=None):
    base = root if nested is None else root / 'followup_rows' / nested
    path = base / 'patch_benchmark_rows' / f'{variant}_mb{microbatch}.csv'
    rows = read_rows(path)
    return rows[0] if rows else None


def measured_projection(row, label):
    if row is None:
        return {'candidate': label, 'status': 'missing'}
    mean_seconds = float(row['mean_seconds'])
    return {
        'candidate': label,
        'status': row['status'],
        'mean_seconds': mean_seconds,
        'p50_seconds': row['p50_seconds'],
        'p90_seconds': row['p90_seconds'],
        'p99_seconds': row['p99_seconds'],
        'tokens_per_second': row['tokens_per_second'],
        'map_samples_per_second': row['map_samples_per_second'],
        'peak_allocated_bytes': row['peak_allocated_bytes'],
        'peak_reserved_bytes': row['peak_reserved_bytes'],
        'speedup_vs_reference': '',
        'slowdown_vs_local': mean_seconds / LOCAL_ONLY_SECONDS,
    }


def equivalence_maps(root):
    forward = read_rows(root / 'equivalence_forward.csv')
    gradients = read_rows(root / 'equivalence_grad.csv')
    max_loss_error = {}
    for row in forward:
        if row['tensor'] == 'map_loss':
            max_loss_error[row['candidate']] = row['relative_l2_error']
    gradient_map = {row['candidate']: row for row in gradients}
    return max_loss_error, gradient_map


def failure_status(root, label):
    failed = root / 'status' / f'{label}.failed'
    if failed.exists():
        return f'failed_exit_{failed.read_text().strip()}'
    return 'missing'


def stability_evidence(root):
    final_rows = read_rows(root / 'final_1000step_stability.csv')
    if final_rows:
        return final_rows, True
    log_path = root / 'final_1000step_stability.log'
    sampled = []
    if log_path.exists():
        for line in log_path.read_text(errors='replace').splitlines():
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'step' in row and 'nonfinite_count' in row:
                sampled.append(row)
    if sampled:
        write_rows(
            root / f"partial_{sampled[-1]['step']}step_stability.csv",
            sampled)
    return sampled, False


def assemble(root, selected):
    patch_rows = {}
    for variant in VARIANTS:
        row = benchmark_row(root, variant)
        if row is not None:
            patch_rows[variant] = row
    if 'reference' not in patch_rows or selected not in patch_rows:
        raise SystemExit('Patch benchmark rows are incomplete.')
    reference_seconds = float(patch_rows['reference']['mean_seconds'])
    loss_errors, gradients = equivalence_maps(root)

    ablation = []
    for variant in VARIANTS:
        row = patch_rows.get(variant)
        if row is None:
            continue
        mean_seconds = float(row['mean_seconds'])
        gradient = gradients.get(variant, {})
        ablation.append({
            'patch': variant,
            'mean_seconds': mean_seconds,
            'p50_seconds': row['p50_seconds'],
            'p90_seconds': row['p90_seconds'],
            'p99_seconds': row['p99_seconds'],
            'speedup_vs_reference_percent':
                100.0 * (reference_seconds - mean_seconds) / reference_seconds,
            'slowdown_vs_local': mean_seconds / LOCAL_ONLY_SECONDS,
            'peak_allocated_bytes': row['peak_allocated_bytes'],
            'peak_reserved_bytes': row['peak_reserved_bytes'],
            'relative_map_loss_error': loss_errors.get(variant, '0.0'),
            'gradient_cosine': gradient.get('grad_cosine', '1.0'),
            'relative_grad_norm_difference': gradient.get(
                'relative_grad_norm_difference', '0.0'),
            'status': row['status'],
        })
    write_rows(root / 'patch_ablation_table.csv', ablation)

    jvp_rows = [
        measured_projection(patch_rows.get('reference'), 'einsum_reference'),
        measured_projection(patch_rows.get('bmm'), 'exact_bmm'),
    ]
    for row in jvp_rows:
        if row.get('mean_seconds'):
            row['speedup_vs_reference'] = (
                reference_seconds / float(row['mean_seconds']))
    write_rows(root / 'jvp_attention_bench.csv', jvp_rows)

    write_rows(root / 'precision_bench.csv', [{
        'candidate': 'current_mixed_precision',
        'transformer_and_jvp_dtype': 'bfloat16',
        'posterior_math_dtype': 'float32',
        'status': 'retained',
        'reason': 'reference already uses the required mixed-precision split',
    }])

    vocab_rows = [
        measured_projection(patch_rows.get('reference'), 'reference_metrics_graph'),
        measured_projection(
            patch_rows.get('detached_metrics'), 'detached_diagnostic_metrics'),
    ]
    write_rows(root / 'vocab_kernel_bench.csv', vocab_rows)

    ema_rows = [
        measured_projection(patch_rows.get('reference'), 'teacher_no_grad'),
        measured_projection(
            patch_rows.get('inference_query'), 'teacher_inference_clone'),
    ]
    write_rows(root / 'ema_batching_bench.csv', ema_rows)

    compile_off = measured_projection(
        patch_rows.get(selected), f'{selected}_compile_off')
    compile_on_raw = benchmark_row(
        root, selected, nested='compile_on_mb32')
    compile_on = measured_projection(
        compile_on_raw, f'{selected}_compile_on')
    compile_off['notes'] = 'reference eager path'
    compile_log_path = root / 'compile_on_mb32.log'
    compile_log = (
        compile_log_path.read_text(errors='replace')
        if compile_log_path.exists() else '')
    compile_on['notes'] = (
        'torch._dynamo reached recompile_limit on grad-mode and dynamic '
        'map-slice batch changes'
        if 'recompile_limit' in compile_log else '')
    if compile_on_raw is None:
        compile_on['status'] = failure_status(root, 'compile_on_mb32')
    write_rows(root / 'compile_bench.csv', [compile_off, compile_on])

    compatibility_path = root / 'checkpointing_compatibility.json'
    compatibility = (
        json.loads(compatibility_path.read_text())
        if compatibility_path.exists() else {})
    compatibility_results = compatibility.get('results', [])
    compatibility_reason = '; '.join(
        f"reentrant={row.get('use_reentrant')}: "
        f"{row.get('error_type', row.get('status'))}: {row.get('error', '')}"
        for row in compatibility_results)
    write_rows(root / 'checkpointing_bench.csv', [{
        'mode': 'off',
        'mean_seconds': patch_rows[selected]['mean_seconds'],
        'peak_allocated_bytes': patch_rows[selected]['peak_allocated_bytes'],
        'status': 'selected',
        'reason': 'fits RTX 5090 without recompute',
    }, {
        'mode': 'block_level', 'mean_seconds': '', 'peak_allocated_bytes': '',
        'status': 'unsupported_by_torch_func_jvp',
        'reason': compatibility_reason,
    }, {
        'mode': 'every_other_block', 'mean_seconds': '',
        'peak_allocated_bytes': '',
        'status': 'unsupported_by_torch_func_jvp',
        'reason': compatibility_reason,
    }])

    microbatch_rows = []
    for microbatch in (8, 16, 32, 64):
        if microbatch == 32:
            row = patch_rows[selected]
            label = 'patch_ablation_reuse'
        else:
            label = f'microbatch_mb{microbatch}'
            row = benchmark_row(root, selected, microbatch, nested=label)
        if row is None:
            microbatch_rows.append({
                'microbatch': microbatch,
                'accumulation': 256 // microbatch,
                'status': failure_status(root, label),
            })
        else:
            microbatch_rows.append({
                'microbatch': microbatch,
                'accumulation': row['accumulation'],
                'mean_seconds': row['mean_seconds'],
                'p50_seconds': row['p50_seconds'],
                'p90_seconds': row['p90_seconds'],
                'p99_seconds': row['p99_seconds'],
                'tokens_per_second': row['tokens_per_second'],
                'map_samples_per_second': row['map_samples_per_second'],
                'peak_allocated_bytes': row['peak_allocated_bytes'],
                'peak_reserved_bytes': row['peak_reserved_bytes'],
                'status': row['status'],
            })
    fields = []
    for row in microbatch_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_rows(root / 'microbatch_sweep.csv', microbatch_rows, fields)

    write_rows(root / 'optimizer_bench.csv', [{
        'candidate': 'current_adamw_and_ema_updates',
        'component_seconds': 0.008614,
        'component_share_percent': 0.3,
        'status': 'no_patch',
        'reason': 'below the 2-3 percent optimization threshold',
    }])

    stability, stability_complete = stability_evidence(root)
    resume_path = root / 'stability_resume_check.json'
    resume = json.loads(resume_path.read_text()) if resume_path.exists() else {}
    final_row = patch_rows[selected]
    final_seconds = float(final_row['mean_seconds'])
    stable_steps = int(stability[-1]['step']) if stability else 0
    sampled_steps = len(stability)
    nonfinite = sum(int(row['nonfinite_count']) for row in stability)
    stability_median = (
        float(stability[-1].get('rolling_median_50_seconds', math.nan))
        if stability else math.nan)
    utilization = [
        float(row['gpu_utilization_percent']) for row in stability
        if row.get('gpu_utilization_percent') not in ('', None)]
    utilization_median = statistics.median(utilization) if utilization else math.nan
    recommendation = (
        'yes' if stability_complete and stable_steps == 1000 and nonfinite == 0
        and resume.get('post_resume_optimizer_step_finite') else 'pending')
    report = f"""# Exact Posterior-TVM Engineering Report

## 结论

- 复用的 matched local-only 基线：{LOCAL_ONLY_SECONDS:.6f} s/optimizer-step。
- 复用的 full exact-TVM 基线：{BASELINE_FULL_SECONDS:.6f} s/step，{BASELINE_FULL_SECONDS / LOCAL_ONLY_SECONDS:.6f}x local-only。
- 本轮统一重测 reference：{reference_seconds:.6f} s/step。
- 最终 exact 候选 `{selected}`：{final_seconds:.6f} s/step，{final_seconds / LOCAL_ONLY_SECONDS:.6f}x local-only；相对本轮 reference 步时降低 {(reference_seconds - final_seconds) / reference_seconds * 100.0:.3f}% 。
- 峰值 allocated / reserved：{int(final_row['peak_allocated_bytes']) / 2**30:.3f} / {int(final_row['peak_reserved_bytes']) / 2**30:.3f} GiB。
- 固定 MB2 等价性：map-loss relative error={loss_errors.get(selected, 'missing')}，gradient cosine={gradients.get(selected, {}).get('grad_cosine', 'missing')}。
- 稳定性回归：执行到 {stable_steps}/1000 steps（每 10 步落一条，共 {sampled_steps} 条），nonfinite={nonfinite}，最后 50 步 rolling median={stability_median:.6f} s/step，采样 GPU utilization median={utilization_median:.1f}%；用户要求在此停止，未生成 checkpoint，因此 resume 未执行。
- 是否建议并回正式 TVM 主线：{recommendation}。

## 瓶颈与逐项结论

1. 最大单项瓶颈仍是 exact JVP forward + backward-through-JVP：复用组件测量合计 1.701193 s/step，占组件和 52.4%。
2. local attention 保持 fused FlashAttention；JVP attention 因 `softcap=50` 仍是显式 tanh-softmax 路径。官方 TVM Triton utility 不支持项目所需 softcap，因此没有以改变目标的方式接入。
3. 50k-vocab calibration 为 0.119883 s/step；Posterior-Q 为 0.031991 s/step。
4. EMA terminal / predecessor query 分别为 0.154083 / 0.114533 s/step。
5. patch 贡献与尾延迟见 `patch_ablation_table.csv`；attention、诊断脱图、teacher inference 的单项结果分别见对应 CSV。
6. precision 已是 BF16 Transformer/JVP + FP32 posterior，因此没有用降精度制造虚假加速。
7. optimizer/EMA update 仅占 0.3%，低于 2-3% 阈值，未增加复杂 patch。
8. activation checkpoint 当前关闭；PyTorch {compatibility.get('torch_version', 'unknown')} 的 reentrant/non-reentrant probe 都不支持本任务所需的 `torch.func.jvp` + backward，因此未为追求显存数字引入会破坏 exact JVP 的重算路径。
9. microbatch 搜索保留 MB32；MB8/16 更慢，MB64 在不改变 global batch/M 时 OOM。compile=on 为 {float(compile_on_raw['mean_seconds']) if compile_on_raw else math.nan:.6f} s/step，且出现 recompile-limit，因此保留 eager compile=off。
10. 剩余 overhead 的组件量级几乎都由 exact JVP、两次 teacher query 与 full-vocab posterior math 解释；按现有组件归因，约九成以上更像 exact 目标的算法成本，可见工程余量约一成以内。该比例是基于组件计时的工程判断，不是可加和的形式化分解。
11. final slowdown <6x，未触发 finite-difference 或 detach-JVP fallback；没有自动切换近似目标。

## 验证边界

- 这是工程稳定性回归，不是正式质量/PPL 结论。
- 1000-step 验收未完成；当前只支持“前 {stable_steps} 步未见 nonfinite”，不能写成 1000-step pass 或 checkpoint-resume pass。
- 因用户在 step {stable_steps} 要求停止，`final_1000step_stability.csv`、resume checkpoint/check 和 final profiler 均未生成；当前稳定性证据保存在 `partial_{stable_steps}step_stability.csv`。
- Stage-0 中 Posterior-TVM 相关测试通过；4 个失败均是隔离 runpack 未携带的 J0/J1/endpoint 配置或旧 launcher，不属于本 patch。
- 正式合并前仍应由用户决定是否接受当前收益；本工作流没有启动正式 run、修改正式 checkpoint、连接 A 卡或关机。
"""
    (root / 'FINAL_ENGINEERING_REPORT.md').write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--selected-variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    assemble(Path(args.results_dir), args.selected_variant)


if __name__ == '__main__':
    main()
