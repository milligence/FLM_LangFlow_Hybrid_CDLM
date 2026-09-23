#!/usr/bin/env python3
"""Build the registered CE trajectory report from saved run artifacts."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def _number(value):
    if value in (None, ''):
        return None
    return float(value)


def _format(value, digits=6):
    return '—' if value is None else f'{value:.{digits}g}'


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _training_value(rows, csv_step, key):
    values = [
        _number(row.get(key)) for row in rows
        if row.get('step') == str(csv_step) and row.get(key) not in (None, '')]
    return values[-1] if values else None


def _gpu_utilization(path):
    if not path.is_file():
        return {}
    header = None
    by_gpu = {}
    for raw in path.read_text(errors='replace').splitlines():
        fields = raw.lstrip('#').split()
        if raw.startswith('#') and 'gpu' in fields and 'sm' in fields:
            header = fields
            continue
        if raw.startswith('#') or header is None:
            continue
        values = raw.split()
        if len(values) > len(header):
            values = values[-len(header):]
        if len(values) != len(header):
            continue
        record = dict(zip(header, values))
        try:
            gpu = int(record['gpu'])
            sm = float(record['sm'])
            memory = float(record['mem'])
        except (KeyError, ValueError):
            continue
        by_gpu.setdefault(gpu, {'sm': [], 'mem': []})
        by_gpu[gpu]['sm'].append(sm)
        by_gpu[gpu]['mem'].append(memory)
    return {
        gpu: {
            'mean_sm_percent': statistics.fmean(values['sm']),
            'mean_memory_percent': statistics.fmean(values['mem']),
        }
        for gpu, values in by_gpu.items() if values['sm']
    }


def _generation_rows(eval_root):
    rows = []
    if not eval_root.is_dir():
        return rows
    for path in sorted(eval_root.glob('*/samples.json')):
        try:
            step_label = path.parent.name
            step = int(step_label.split('_')[-1])
        except ValueError:
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        quality = payload.get('sample_quality', {})
        rows.append({
            'step': step,
            'ppl': payload.get('generative_ppl'),
            'entropy': quality.get('mean_sample_unigram_entropy_nats'),
            'distinct_1': quality.get('distinct_1'),
            'distinct_2': quality.get('distinct_2'),
            'repeated_4gram': quality.get('repeated_4gram_fraction'),
            'max_token': quality.get('max_single_token_fraction'),
            'special_token': quality.get('special_token_frequency'),
            'samples': payload.get('generated_seqs', []),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    eval_root = args.eval_root or run_dir / 'evaluations'
    output = args.output or run_dir / 'CE_HEALTH_BASELINE_SUMMARY.md'
    manifest = json.loads(
        (run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    config = manifest['resolved_config']
    global_batch = int(config['loader']['global_batch_size'])
    sequence_length = int(config['model']['length'])
    dataset_tokens = int(
        manifest['data_identity']['packed_manifest']['splits']['train']['tokens'])
    packed = manifest['data_identity']['packed_manifest']
    train_split = packed['splits']['train']
    train_path = str(Path(manifest['data_identity']['packed_dir']) / train_split['file'])
    tokens_per_step = global_batch * sequence_length
    steps_per_pass = dataset_tokens / tokens_per_step
    metrics_rows = _read_csv(run_dir / 'local_metrics' / 'metrics.csv')
    validation_rows = [
        row for row in metrics_rows
        if row.get('val/hybrid_token_ce') not in (None, '')]
    throughput = [
        json.loads(line) for line in
        (run_dir / 'throughput.jsonl').read_text(encoding='utf-8').splitlines()
    ] if (run_dir / 'throughput.jsonl').is_file() else []
    generations = _generation_rows(eval_root)

    lines = ['# CE health-baseline summary', '']
    lines.extend([
        '## 实验配置', '',
        f"- run ID：`{run_dir.name}`",
        f"- Git commit：`{manifest['source'].get('commit')}`",
        f"- 模型：`{config['algo']['name']}`，sequence length {sequence_length}",
        f"- optimizer：AdamW，peak LR {config['optim']['lr']}",
        f"- global batch：{global_batch}",
        f"- warmup：{config['lr_scheduler']['num_warmup_steps']} optimizer steps",
        f"- 完成 steps：{manifest.get('optimizer_steps', 0)}",
        f"- tokens seen：{manifest.get('nominal_tokens_seen', 0)}",
        f"- effective data passes：{manifest.get('nominal_tokens_seen', 0) / dataset_tokens:.6f}",
        f"- 训练文件：`{train_path}`",
        f"- 训练文件 bytes：{train_split['bytes']}",
        f"- dtype：`{packed['dtype']}`",
        f"- 精确 token count：{train_split['tokens']}",
        f"- tokenizer：`{manifest['data_identity']['tokenizer']}`；vocab size {packed['vocab_size']}",
        f"- packing：{packed['packing']}",
        f"- BOS/EOS/PAD：{packed.get('bos_token_id')} / {packed.get('eos_token_id')} / {packed.get('pad_token_id')}",
        f"- 可构造 sequences：{train_split['sequences']}",
        f"- tokens/optimizer step：{tokens_per_step}；steps/data pass：{steps_per_pass:.3f}",
        f"- 10k/20k/30k data passes：{10000 / steps_per_pass:.3f} / {20000 / steps_per_pass:.3f} / {30000 / steps_per_pass:.3f}",
        '', '## 计算性能', '',
    ])
    if throughput:
        rates = [row['aggregate_tokens_per_second'] for row in throughput]
        step_times = [row['step_seconds'] for row in throughput]
        mean_rate = statistics.fmean(rates)
        peak_memory = max(row['gpu_memory_reserved_bytes'] for row in throughput)
        peak_allocated = max(
            row['gpu_memory_allocated_bytes'] for row in throughput)
        estimated_steps = int(7200 * mean_rate / tokens_per_step)
        lines.extend([
            f"- total wall time：{manifest.get('wall_seconds', 0):.3f} s",
            f"- average tokens/s：{mean_rate:.3f}",
            f"- peak tokens/s：{max(rates):.3f}",
            f"- average optimizer step time：{statistics.fmean(step_times):.6f} s",
            f"- average dataloader time/step：{statistics.fmean(row['dataloader_seconds_per_step'] for row in throughput):.6f} s",
            f"- average forward time/step：{statistics.fmean(row['forward_seconds_per_step'] for row in throughput):.6f} s",
            f"- average backward+DDP time/step：{statistics.fmean(row['backward_and_ddp_seconds_per_step'] for row in throughput):.6f} s",
            f"- average optimizer time/step：{statistics.fmean(row['optimizer_seconds_per_step'] for row in throughput):.6f} s",
            f"- peak allocated GPU memory：{peak_allocated / 1024 ** 3:.3f} GiB",
            f"- peak reserved GPU memory：{peak_memory / 1024 ** 3:.3f} GiB",
            f"- 以实测平均吞吐估算两小时：{estimated_steps} steps / {estimated_steps * tokens_per_step} tokens / {estimated_steps * tokens_per_step / dataset_tokens:.6f} data passes",
        ])
    utilization = _gpu_utilization(run_dir / 'gpu_dmon.log')
    for gpu, values in sorted(utilization.items()):
        lines.append(
            f"- GPU {gpu} mean SM utilization：{values['mean_sm_percent']:.2f}%；"
            f"mean memory utilization：{values['mean_memory_percent']:.2f}%")

    lines.extend([
        '', '## Posterior learning trajectory', '',
        '| step | tokens seen | data passes | train CE | val CE | raw Brier | p_target | target margin | top-1 | posterior entropy |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in validation_rows:
        csv_step = int(float(row['step']))
        step = csv_step + 1
        tokens = step * global_batch * sequence_length
        lines.append(
            f"| {step} | {tokens} | {tokens / dataset_tokens:.6f} | "
            f"{_format(_training_value(metrics_rows, csv_step, 'train/hybrid_token_ce'))} | "
            f"{_format(_number(row.get('val/hybrid_token_ce')))} | "
            f"{_format(_number(row.get('val/hybrid_raw_brier')))} | "
            f"{_format(_number(row.get('val/hybrid_target_probability')))} | "
            f"{_format(_number(row.get('val/hybrid_target_margin')))} | "
            f"{_format(_number(row.get('val/hybrid_top1_accuracy')))} | "
            f"{_format(_number(row.get('val/hybrid_posterior_entropy')))} |")

    lines.extend(['', '## Gamma-bin diagnostics', ''])
    gamma = manifest.get('gamma_bin_diagnostics', {})
    for step, bins in sorted(gamma.items(), key=lambda item: int(item[0])):
        lines.extend([
            f'### Step {step}', '',
            '| quantile bin | samples | token CE | raw Brier | p_target | margin | top-1 | entropy |',
            '|---|---:|---:|---:|---:|---:|---:|---:|',
        ])
        for label, values in bins.items():
            lines.append(
                f"| {label} | {_format(values.get('sample_count'), 0)} | "
                f"{_format(values.get('token_ce'))} | {_format(values.get('raw_brier'))} | "
                f"{_format(values.get('target_probability'))} | {_format(values.get('target_margin'))} | "
                f"{_format(values.get('top1_accuracy'))} | {_format(values.get('posterior_entropy'))} |")

    lines.extend(['', '## Embedding geometry', ''])
    geometry = manifest.get('embedding_nearest_neighbor_cosine_distance', {})
    if geometry:
        keys = sorted(next(iter(geometry.values())).keys())
        lines.append('| step | ' + ' | '.join(keys) + ' |')
        lines.append('|---|' + '|'.join('---:' for _ in keys) + '|')
        for step, values in geometry.items():
            lines.append(
                f"| {step} | " + ' | '.join(
                    _format(values.get(key)) for key in keys) + ' |')

    lines.extend([
        '', '## Generation trajectory', '',
        '| step | tokens seen | Gen. PPL | entropy | distinct-1 | distinct-2 | repeated 4-gram | max-token fraction | special-token frequency |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in generations:
        tokens = row['step'] * global_batch * sequence_length
        lines.append(
            f"| {row['step']} | {tokens} | {_format(row['ppl'])} | "
            f"{_format(row['entropy'])} | {_format(row['distinct_1'])} | "
            f"{_format(row['distinct_2'])} | {_format(row['repeated_4gram'])} | "
            f"{_format(row['max_token'])} | {_format(row['special_token'])} |")

    lines.extend(['', '## 固定 seed 样本', ''])
    for row in generations:
        lines.append(f"### Step {row['step']}")
        lines.append('')
        for sample in row['samples'][:8]:
            lines.append(f'- {json.dumps(sample, ensure_ascii=False)}')
        lines.append('')

    nonfinite = sum(
        _number(row.get('train/hybrid_nonfinite_count')) or 0
        for row in metrics_rows)
    lines.extend([
        '## 异常', '',
        f'- non-finite count：{int(nonfinite)}',
        f"- run status：`{manifest.get('status')}`",
    ])
    if manifest.get('error'):
        lines.append(f"- error：`{manifest['error']}`")

    lines.extend([
        '', '## 最终判断', '',
        '根据以上完整轨迹回答：CE 是否健康学习；1k 结果是否主要为欠训练；'
        'posterior、generation 与文本是否随 tokens seen 同步改善；哪些 gamma 区间'
        '最困难；当前证据最符合 undertraining、optimization、noise scheduler、SC/'
        'bias、sampling、数据或实现问题中的哪一种。报告完成前不启动调参实验。',
        '',
    ])
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
