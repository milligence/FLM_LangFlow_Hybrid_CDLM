"""Bounded frozen-F50k grid search; calls the existing formal sample evaluator."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

from grid_candidates import (A, B, C, D, coarse_four, coarse_two, key,
                             perturb_for_robustness, refine_four, refine_two,
                             unique, valid)

CHECKPOINT_SHA = '6c4763c6d07188e84ad49411983b0ee6c72b19887fb162296819f484cce6dcfc'
BANKS = {'reproduce': (424242, 128), 'S0': (271828, 32),
         'S1': (314159, 128), 'SR': (161803, 128),
         'S2': (141421, 512)}
WEIGHTS = ('online', 'eval_ema')
QUALITY = ('entropy', 'distinct_1', 'distinct_2', 'repeated_4gram_fraction',
           'max_single_token_fraction', 'special_token_frequency')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def grid_id(grid):
    return hashlib.sha256(json.dumps(list(grid), separators=(',', ':')).encode()).hexdigest()[:12]


def metric(payload):
    scores = payload['per_sample_scores']
    ids = payload['generated_token_ids']
    if len(scores) != payload['num_samples'] or len(ids) != len(scores):
        raise ValueError('Incomplete sample scores or generated token IDs')
    if any(len(row) != 128 for row in ids):
        raise ValueError('Generated sequence length is not 128')
    losses = [float(row['nll']) for row in scores]
    counts = [int(row['scored_tokens']) for row in scores]
    if any(not math.isfinite(v) for v in losses) or any(n <= 0 for n in counts):
        raise ValueError('Nonfinite NLL or zero scored-token sample')
    q = payload['sample_quality']
    result = {'nll': sum(losses) / sum(counts), 'scored_tokens': sum(counts)}
    result['ppl'] = math.exp(result['nll'])
    result['entropy'] = float(q['mean_sample_unigram_entropy_nats'])
    for name in QUALITY[1:]:
        result[name] = float(q[name])
    return result


def guard(value, baseline, strict=True):
    if strict:
        return (value['distinct_2'] >= .98 * baseline['distinct_2']
                and value['distinct_1'] >= .95 * baseline['distinct_1']
                and value['entropy'] >= baseline['entropy'] - .03
                and value['repeated_4gram_fraction'] <= baseline['repeated_4gram_fraction'] + .001
                and value['max_single_token_fraction'] <= baseline['max_single_token_fraction'] + .005
                and value['special_token_frequency'] <= baseline['special_token_frequency'] + .002)
    return (value['distinct_2'] >= .90 * baseline['distinct_2']
            and value['distinct_1'] >= .85 * baseline['distinct_1']
            and value['entropy'] >= baseline['entropy'] - .10
            and value['repeated_4gram_fraction'] <= baseline['repeated_4gram_fraction'] + .002
            and value['max_single_token_fraction'] <= baseline['max_single_token_fraction'] + .01
            and value['special_token_frequency'] <= baseline['special_token_frequency'] + .004)


def coverage_alt(value, baseline):
    return (value['ppl'] <= 1.03 * baseline['ppl']
            and value['distinct_2'] >= baseline['distinct_2'] + .03
            and value['distinct_1'] >= baseline['distinct_1'] + .01
            and value['entropy'] >= baseline['entropy'] - .01
            and value['repeated_4gram_fraction'] <= baseline['repeated_4gram_fraction'] + .001
            and value['max_single_token_fraction'] <= baseline['max_single_token_fraction'] + .005
            and value['special_token_frequency'] <= baseline['special_token_frequency'] + .002)


def paired_bootstrap(candidate, baseline, seed=505050):
    c = candidate['per_sample_scores']
    b = baseline['per_sample_scores']
    if len(c) != len(b):
        raise ValueError('Unpaired sample count')
    rng = random.Random(seed)
    n = len(c)
    diffs = []
    for _ in range(2000):
        ix = [rng.randrange(n) for _ in range(n)]
        c_nll = sum(c[i]['nll'] for i in ix) / sum(c[i]['scored_tokens'] for i in ix)
        b_nll = sum(b[i]['nll'] for i in ix) / sum(b[i]['scored_tokens'] for i in ix)
        diffs.append(c_nll - b_nll)
    diffs.sort()
    return {'delta_nll': metric(candidate)['nll'] - metric(baseline)['nll'],
            'ci95': [diffs[49], diffs[1950]],
            'upper_98_75': diffs[1974], 'resamples': 2000,
            'seed': seed, 'sample_count': n}


def block_metrics(payload):
    from metrics import Metrics
    import torch
    ids = payload['generated_token_ids']
    scores = payload['per_sample_scores']
    if len(ids) != 512:
        raise ValueError('S2 must contain exactly 512 samples')
    blocks = []
    for start in range(0, 512, 128):
        quality = Metrics.compute_sample_quality(
            torch.tensor(ids[start:start + 128], dtype=torch.long), [50256])
        block = {'block': start // 128, 'start_sample_id': start,
                 'nll': sum(s['nll'] for s in scores[start:start + 128]) /
                 sum(s['scored_tokens'] for s in scores[start:start + 128])}
        block['ppl'] = math.exp(block['nll'])
        block['entropy'] = quality['mean_sample_unigram_entropy_nats']
        for name in QUALITY[1:]:
            block[name] = quality[name]
        blocks.append(block)
    return blocks


class Search:
    def __init__(self, args):
        self.args = args
        self.out = args.output.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.checkpoint = args.checkpoint.resolve()
        self.formal = args.formal.resolve()
        self.bindings = json.loads(args.bindings.read_text())
        self.contract = args.contract.resolve()
        self.project = Path(self.bindings['repo_root'])
        sys.path.insert(0, str(self.project))
        from task1_tvm_50k_final_adapter import _run_eval_job
        self._run_eval_job = _run_eval_job
        self.train = {'seed': 20260921, 'global_batch': 256}
        self.rows = []
        self.forward_calls = 0
        self.eval_seconds = 0.0
        self.b2 = self._actual_b2()
        self.bases = {4: B, 2: self.b2}

    def _actual_b2(self):
        path = self.formal / 'step_050000__online__finite_two__n128' / '.hydra' / 'overrides.yaml'
        match = re.search(r'algo\.task1_eval_physical_grid=(\[[^\n]+\])', path.read_text())
        if not match:
            raise RuntimeError('Actual formal finite_two grid unavailable')
        grid = tuple(float(x) for x in ast.literal_eval(match.group(1)))
        if not valid(grid, 2):
            raise RuntimeError(f'Invalid actual B2 grid: {grid}')
        return grid

    def prepare(self):
        if sha_file(self.checkpoint) != CHECKPOINT_SHA:
            raise RuntimeError('Exact50 checkpoint SHA mismatch')
        if self.b2 != (0.0, .5814685447, .95):
            raise RuntimeError(f'B2 differs from declared contract: {self.b2}')
        four, two = coarse_four(), coarse_two(self.b2)
        if len(four) != 40 or len(two) > 12:
            raise RuntimeError('Candidate budget mismatch')
        manifest = {'checkpoint': str(self.checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA,
                    'checkpoint_step': 50000, 'weights': list(WEIGHTS),
                    'source_manifest_sha256': self.bindings['provenance']['source_sha256_manifest'],
                    'protocol_id': 'owt128-gpt2large-genppl-v1',
                    'formal_reference_dir': str(self.formal),
                    'sample_policy': 'base_seed_plus_sample_index',
                    'banks': {name: {'seed': seed, 'samples': samples,
                                     'uid_hash': hashlib.sha256(json.dumps(
                                         [seed + i for i in range(samples)]).encode()).hexdigest()}
                              for name, (seed, samples) in BANKS.items()},
                    'S2_override': '512 samples, four disjoint 128-sample blocks, 3/4 direction',
                    'B2_actual': list(self.b2), 'no_model_or_training_changes': True}
        existing = self.out / 'manifest.json'
        if existing.exists() and json.loads(existing.read_text()) != manifest:
            raise RuntimeError('Existing manifest differs; refusing mixed search')
        dump(existing, manifest)
        dump(self.out / 'candidates_stage1.json', {
            'four': [{'id': grid_id(g), 'grid': g, 'source': (
                'control' if i < 4 else 'local_B' if i < 16 else
                'low_discrepancy_physical' if i < 28 else 'low_discrepancy_log_remaining')}
                     for i, g in enumerate(four)],
            'two': [{'id': grid_id(g), 'grid': g} for g in two]})
        return {4: four, 2: two}

    def job(self, stage, nfe, weight, grid):
        seed, samples = BANKS[stage]
        gid = grid_id(grid)
        name = f'{stage}__{nfe}nfe__{weight}__{gid}__n{samples}'
        path = self.out / 'evaluation' / name / 'samples.json'
        if not path.exists():
            import argparse as ap
            t0 = time.time()
            job = {'name': name, 'mode': 'finite', 'weight': weight, 'nfe': nfe,
                   'grid': list(grid), 'samples': samples, 'seed': seed, 'batch_size': 4}
            adapter_args = ap.Namespace(output=self.out, contract_dir=self.contract, line='f')
            code = self._run_eval_job(adapter_args, self.bindings, self.train, self.checkpoint, job)
            elapsed = time.time() - t0
            with (self.out / 'tasks.jsonl').open('a') as handle:
                handle.write(json.dumps({'name': name, 'exit_code': code, 'seconds': elapsed,
                                         'nfe': nfe, 'samples': samples, 'seed': seed,
                                         'grid': grid, 'weight': weight}) + '\n')
            if code or not path.exists():
                raise RuntimeError(f'Evaluation failed: {name}; exit={code}')
            self.forward_calls += nfe * samples
            self.eval_seconds += elapsed
        payload = json.loads(path.read_text())
        if (payload['checkpoint_global_step'] != 50000
                or payload['weights'] != weight or payload['nfe'] != nfe
                or payload['num_samples'] != samples):
            raise RuntimeError(f'Evaluation payload identity mismatch: {name}')
        m = metric(payload)
        self.rows.append({'stage': stage, 'nfe': nfe, 'weight': weight,
                          'grid_id': gid, 'grid': list(grid), 'samples': samples, **m})
        print(f'COMPLETE {name} ppl={m["ppl"]:.4f} D2={m["distinct_2"]:.4f}', flush=True)
        return payload, m

    def reproduce(self):
        records = []
        for nfe in (4, 2):
            formal_mode = 'four_uniform' if nfe == 4 else 'two'
            for weight in WEIGHTS:
                current, m = self.job('reproduce', nfe, weight, self.bases[nfe])
                formal_path = self.formal / f'step_050000__{weight}__finite_{formal_mode}__n128' / 'samples.json'
                old = json.loads(formal_path.read_text())
                old_m = metric(old)
                exact = current['generated_token_ids'] == old['generated_token_ids']
                numeric = (abs(m['ppl'] / old_m['ppl'] - 1) <= .005
                           and all(abs(m[k] - old_m[k]) <= .005
                                   for k in ('entropy', 'distinct_1', 'distinct_2')))
                row = {'nfe': nfe, 'weight': weight, 'exact_token_ids': exact,
                       'numeric_tolerance_pass': numeric, 'new': m, 'formal': old_m}
                records.append(row)
                if not exact and not numeric:
                    dump(self.out / 'reproduction.json', records)
                    raise RuntimeError('Formal protocol reproduction gate failed')
        dump(self.out / 'reproduction.json', records)

    def stage1(self, initial):
        selected, anchors = {}, {}
        for nfe in (4, 2):
            measures = {}
            for weight in WEIGHTS:
                baseline = self.job('S0', nfe, weight, self.bases[nfe])[1]
                measure = {}
                for g in initial[nfe]:
                    measure[grid_id(g)] = self.job('S0', nfe, weight, g)[1]
                measures[weight] = measure
                qualified = [g for g in initial[nfe] if g != self.bases[nfe]
                             and guard(measure[grid_id(g)], baseline, strict=False)]
                qualified.sort(key=lambda g: (measure[grid_id(g)]['nll'], grid_id(g)))
                anchors[(nfe, weight)] = qualified[:2]
                if nfe == 4:
                    top = qualified[:5]
                    remaining = [g for g in qualified if g not in top]
                    pareto = [g for g in remaining if not any(
                        measure[grid_id(h)]['nll'] <= measure[grid_id(g)]['nll']
                        and measure[grid_id(h)]['distinct_2'] >= measure[grid_id(g)]['distinct_2']
                        and (measure[grid_id(h)]['nll'] < measure[grid_id(g)]['nll']
                             or measure[grid_id(h)]['distinct_2'] > measure[grid_id(g)]['distinct_2'])
                        for h in remaining if h != g)]
                    pareto.sort(key=lambda g: (-measure[grid_id(g)]['distinct_2'],
                                                measure[grid_id(g)]['nll'], grid_id(g)))
                    selected[(nfe, weight)] = [self.bases[nfe], *top, *pareto[:2]]
                else:
                    selected[(nfe, weight)] = initial[nfe]
        pools = {}
        for nfe in (4, 2):
            keep = unique([g for weight in WEIGHTS for g in selected[(nfe, weight)]], nfe)
            chosen_anchors = unique([g for weight in WEIGHTS for g in anchors[(nfe, weight)]], nfe)
            additions = (refine_four(chosen_anchors) if nfe == 4 else refine_two(chosen_anchors))
            pools[nfe] = unique([*keep, *additions], nfe)
            if len(pools[nfe]) > (40 if nfe == 4 else 28):
                raise RuntimeError('Stage2 candidate budget exceeded')
        dump(self.out / 'candidates_stage2.json', {str(n): [list(g) for g in pool]
                                                  for n, pool in pools.items()})
        return pools

    def stage2(self, pools):
        frozen = {}
        for nfe in (4, 2):
            for weight in WEIGHTS:
                base = self.job('S1', nfe, weight, self.bases[nfe])[1]
                measured = {grid_id(g): self.job('S1', nfe, weight, g)[1] for g in pools[nfe]}
                candidates = [g for g in pools[nfe] if g != self.bases[nfe]]
                good = [g for g in candidates if guard(measured[grid_id(g)], base)]
                good.sort(key=lambda g: (measured[grid_id(g)]['nll'], grid_id(g)))
                primary = good[0] if good and measured[grid_id(good[0])]['nll'] < base['nll'] else None
                alts = [g for g in candidates if coverage_alt(measured[grid_id(g)], base)]
                alts.sort(key=lambda g: (-measured[grid_id(g)]['distinct_2'],
                                          measured[grid_id(g)]['nll'], grid_id(g)))
                alternative = next((g for g in alts if g != primary), None)
                frozen[(nfe, weight)] = {'primary': primary, 'alternative': alternative}
        serial = {f'{n}_{w}': {k: list(v) if v else None for k, v in values.items()}
                  for (n, w), values in frozen.items()}
        dump(self.out / 'shortlist_before_S2.json', serial)
        return frozen

    def stage3(self, frozen):
        robustness = {}
        for nfe in (4, 2):
            for weight in WEIGHTS:
                self.job('SR', nfe, weight, self.bases[nfe])
                for role, center in frozen[(nfe, weight)].items():
                    if center is None:
                        continue
                    center_m = self.job('SR', nfe, weight, center)[1]
                    perturbations = perturb_for_robustness(center)
                    checks = []
                    for g in perturbations:
                        m = self.job('SR', nfe, weight, g)[1]
                        checks.append({'grid': list(g), 'pass': (
                            m['ppl'] <= 1.10 * center_m['ppl']
                            and m['distinct_2'] >= .95 * center_m['distinct_2']
                            and m['entropy'] >= center_m['entropy'] - .05
                            and m['repeated_4gram_fraction'] <= center_m['repeated_4gram_fraction'] + .001
                            and m['special_token_frequency'] <= center_m['special_token_frequency'] + .002)})
                    robustness[f'{nfe}_{weight}_{role}'] = {
                        'robust': bool(checks) and all(x['pass'] for x in checks),
                        'boundary_limited': len(checks) < 2 * (nfe - 1),
                        'perturbations': checks}
        dump(self.out / 'robustness.json', robustness)
        return robustness

    def stage4(self, frozen, robustness):
        final = {}
        for nfe in (4, 2):
            for weight in WEIGHTS:
                base_grid = self.bases[nfe]
                baseline, base_m = self.job('S2', nfe, weight, base_grid)
                base_blocks = block_metrics(baseline)
                entries = []
                for role, grid in frozen[(nfe, weight)].items():
                    if grid is None:
                        continue
                    candidate, m = self.job('S2', nfe, weight, grid)
                    blocks = block_metrics(candidate)
                    pairs = paired_bootstrap(candidate, baseline)
                    strict = guard(m, base_m)
                    d2_blocks = sum(c['distinct_2'] >= .95 * b['distinct_2']
                                    for c, b in zip(blocks, base_blocks))
                    mean_d2 = sum(c['distinct_2'] for c in blocks) / 4
                    mean_b2 = sum(b['distinct_2'] for b in base_blocks) / 4
                    if role == 'primary':
                        confirmed = (m['ppl'] <= .95 * base_m['ppl'] and strict
                                     and mean_d2 >= .98 * mean_b2 and d2_blocks >= 3
                                     and pairs['upper_98_75'] < 0
                                     and robustness[f'{nfe}_{weight}_{role}']['robust'])
                    else:
                        direction = sum(c['distinct_2'] > b['distinct_2']
                                        for c, b in zip(blocks, base_blocks))
                        confirmed = (coverage_alt(m, base_m) and direction >= 3
                                     and robustness[f'{nfe}_{weight}_{role}']['robust'])
                    entries.append({'role': role, 'grid': list(grid), 'grid_id': grid_id(grid),
                                    'metrics': m, 'blocks': blocks, 'paired': pairs,
                                    'coverage_guard_pass': strict, 'd2_blocks_95pct': d2_blocks,
                                    'confirmed_metrics': confirmed,
                                    'text_review_status': 'pending_human_review'})
                    self._text_pairs(nfe, weight, baseline, candidate, role)
                final[f'{nfe}_{weight}'] = {'baseline_grid': list(base_grid),
                                           'baseline_metrics': base_m,
                                           'baseline_blocks': base_blocks,
                                           'candidates': entries}
        dump(self.out / 'statistics' / 'final.json', final)
        return final

    def _text_pairs(self, nfe, weight, baseline, candidate, role):
        rng = random.Random(50000 + nfe)
        ids = rng.sample(range(512), 20)
        pairs = []
        for sample_id in ids:
            left_candidate = bool(rng.randrange(2))
            b = baseline['generated_seqs'][sample_id]
            c = candidate['generated_seqs'][sample_id]
            pairs.append({'sample_id': sample_id, 'left': c if left_candidate else b,
                          'right': b if left_candidate else c})
        path = self.out / 'text_pairs' / f'{nfe}_{weight}_{role}'
        dump(path.with_suffix('.blind.json'), {'pairs': pairs, 'review_status': 'pending_human_review'})
        dump(path.with_suffix('.key.json'), {'candidate_left': [bool(p['left'] ==
            candidate['generated_seqs'][p['sample_id']]) for p in pairs]})

    def finish(self, final):
        with (self.out / 'metrics.csv').open('w', newline='') as handle:
            fields = ['stage', 'nfe', 'weight', 'grid_id', 'grid', 'samples',
                      'ppl', 'nll', 'scored_tokens', *QUALITY]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({**row, 'grid': json.dumps(row['grid'])})
        for nfe in (4, 2):
            for weight in WEIGHTS:
                group = final[f'{nfe}_{weight}']
                verified = [x for x in group['candidates'] if x['confirmed_metrics']]
                dump(self.out / 'outputs' / f'{nfe}_{weight}.json', {
                    'checkpoint_step': 50000, 'full_state_sha256': CHECKPOINT_SHA,
                    'weights_type': weight, 'nfe': nfe,
                    'baseline_grid': group['baseline_grid'],
                    'baseline_metrics': group['baseline_metrics'],
                    'candidates': group['candidates'],
                    'status': ('metric_candidate_text_pending' if verified
                               else 'baseline_retained_no_confirmed_gain'),
                    'text_review_status': 'pending_human_review',
                    'final_samples': 512, 'extra_forward_calls_this_process': self.forward_calls})
        lines = ['# F@50k 冻结权重有限节点搜索', '',
                 '仅对 exact-50k 的 online 和 eval EMA 做只读生成；没有训练、修改权重或JVP。',
                 '最终独立 S2 样本量为512（用户将原工单1024改为512），分为4个不重叠128块，方向阈值3/4。',
                 '文本盲评仍待独立人工审阅；指标确认不等同语义质量确认。', '',
                 f'Checkpoint SHA256: `{CHECKPOINT_SHA}`。',
                 f'本次新增 sequence-forward calls: {self.forward_calls}；评测墙钟秒: {self.eval_seconds:.1f}。',
                 '完整候选与逐任务原始数据见 candidates_stage1.json、candidates_stage2.json、tasks.jsonl、metrics.csv 与 evaluation/。',
                 '', '## 复现门槛', '',
                 '| NFE | 权重 | token IDs精确一致 | 数值容差通过 |',
                 '|---:|---|---|---|']
        for r in json.loads((self.out / 'reproduction.json').read_text()):
            lines.append(f'| {r["nfe"]} | {r["weight"]} | {r["exact_token_ids"]} | {r["numeric_tolerance_pass"]} |')
        lines += ['', '## 独立512样本结果', '',
                  '| NFE | 权重 | 类型 | PPL | H | D1 | D2 | repeat4 | 相对baseline PPL | 成对NLL 95% CI | 指标确认 |',
                  '|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|']
        for nfe in (4, 2):
            for weight in WEIGHTS:
                group = final[f'{nfe}_{weight}']
                base = group['baseline_metrics']
                def fmt(role, m, relative, ci, yes):
                    lines.append(f'| {nfe} | {weight} | {role} | {m["ppl"]:.3f} | {m["entropy"]:.4f} | {m["distinct_1"]:.4f} | {m["distinct_2"]:.4f} | {m["repeated_4gram_fraction"]:.5f} | {relative:.4f} | {ci} | {yes} |')
                fmt('baseline', base, 1.0, '—', '—')
                for e in group['candidates']:
                    m = e['metrics']
                    ci = ', '.join(f'{v:.5f}' for v in e['paired']['ci95'])
                    fmt(e['role'], m, m['ppl']/base['ppl'], ci, e['confirmed_metrics'])
        lines += ['', '## 判定边界', '',
                  '单样本NLL及score-token计数、四个128块、2000次配对bootstrap及98.75%单侧上界见 statistics/final.json。',
                  '未通过门槛时保留旧B4/B2。未完成独立人工文本审阅前，不宣称语义质量已验证。',
                  '旧OWT raw参照若非同一512样本协议，只作背景，不代入任何门控。',
                  '结果仅适用于F exact-50k 权重，不外推到其他checkpoint。', '']
        (self.out / 'GRID_SEARCH_FINAL_REPORT.md').write_text('\n'.join(lines))
        dump(self.out / 'SEARCH_COMPLETE.json', {'checkpoint_sha256_after': sha_file(self.checkpoint),
                                                  'extra_forward_calls': self.forward_calls,
                                                  'evaluation_seconds_this_process': self.eval_seconds})
        if json.loads((self.out / 'SEARCH_COMPLETE.json').read_text())['checkpoint_sha256_after'] != CHECKPOINT_SHA:
            raise RuntimeError('Checkpoint changed during frozen search')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--formal', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_DATASETS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    search = Search(args)
    initial = search.prepare()
    if args.prepare_only:
        print('PREPARED 40 four-NFE and <=12 two-NFE candidates', flush=True)
        return
    search.reproduce()
    pools = search.stage1(initial)
    frozen = search.stage2(pools)
    robustness = search.stage3(frozen)
    final = search.stage4(frozen, robustness)
    search.finish(final)
    print('SEARCH_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
