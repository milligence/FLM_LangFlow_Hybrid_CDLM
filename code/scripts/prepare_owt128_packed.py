"""Prepare a bounded, document-disjoint OWT subset without full Arrow caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = 'Skylion007/openwebtext'
REVISION = '433fe0f44ed7894fea29c08b3202aa348ccc6369'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PackedWriter:
    def __init__(self, path, sequences, length=128, special=50256):
        self.path = Path(path)
        self.target = sequences
        self.length = length
        self.special = special
        self.count = 0
        self.pending = []
        self.handle = self.path.with_suffix('.part').open('wb')

    def add(self, ids):
        self.pending.extend(ids)
        self.pending.append(self.special)  # document EOS
        inner = self.length - 2
        count = min(len(self.pending) // inner, self.target - self.count)
        if count:
            rows = np.full((count, self.length), self.special, dtype='<u4')
            rows[:, 1:-1] = np.asarray(self.pending[:count * inner], dtype='<u4').reshape(count, inner)
            rows.tofile(self.handle)
            self.pending = self.pending[count * inner:]
            self.count += count

    def finish(self):
        self.handle.close()
        if self.count != self.target:
            raise RuntimeError(f'incomplete split: {self.count}/{self.target}')
        os.replace(self.path.with_suffix('.part'), self.path)
        return {'file': self.path.name, 'sequences': self.count,
                'tokens': self.count * self.length, 'bytes': self.path.stat().st_size,
                'sha256': sha256(self.path), 'discarded_tail_tokens': len(self.pending)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer-path', default='gpt2')
    parser.add_argument('--train-sequences', type=int, default=2560000)
    parser.add_argument('--validation-sequences', type=int, default=8192)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('completed dataset exists; verify it instead of overwriting')
    source_dir = out / 'source_shards'
    source_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    assert tokenizer.vocab_size == 50257 and tokenizer.eos_token_id == 50256
    tokenizer.save_pretrained(out / 'tokenizer')
    writers = {name: PackedWriter(out / f'{name}.bin', count) for name, count in
               [('validation', args.validation_sequences), ('train', args.train_sequences)]}
    split = 'validation'
    seen = set()
    docs = {'validation': 0, 'train': 0}
    duplicates = 0
    sources = []
    started = time.monotonic()
    done = False
    with (out / 'documents.jsonl').open('w') as doclog:
        for shard in range(80):
            filename = f'{shard:04d}.parquet'
            local = source_dir / filename
            url = f'https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/plain_text/train/{filename}'
            if not local.exists():
                print(f'download {filename}', flush=True)
                subprocess.run(['curl', '-fL', '--connect-timeout', '15', '--max-time', '600',
                                '--speed-limit', '1024', '--speed-time', '60', '--retry', '1',
                                '-o', str(local) + '.part', url], check=True)
                os.replace(str(local) + '.part', local)
            sources.append({'path': f'plain_text/train/{filename}',
                            'bytes': local.stat().st_size, 'sha256': sha256(local)})
            row_id = 0
            for batch in pq.ParquetFile(local).iter_batches(batch_size=256, columns=['text']):
                texts = [re.sub(r'\n\n\n+', '\n\n', t).strip() for t in batch.column(0).to_pylist()]
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                                    return_token_type_ids=False, verbose=False)['input_ids']
                for text, ids in zip(texts, encoded):
                    current_row = row_id
                    row_id += 1
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    if not ids or content_hash in seen:
                        duplicates += 1
                        continue
                    seen.add(content_hash)
                    writers[split].add(ids)
                    docs[split] += 1
                    doclog.write(json.dumps({'split': split, 'shard': filename, 'row': current_row,
                                             'normalized_text_sha256': content_hash}) + '\n')
                    if writers[split].count == writers[split].target:
                        if split == 'validation':
                            split = 'train'  # discard validation tail; never share a document
                        else:
                            done = True
                            break
                print(json.dumps({'train_sequences': writers['train'].count,
                                  'validation_sequences': writers['validation'].count,
                                  'documents': docs, 'seconds': round(time.monotonic() - started, 1)}), flush=True)
                if done:
                    break
            if done:
                break
    splits = {name: writer.finish() for name, writer in writers.items()}
    manifest = {'format': 'owt128-uint32-v1', 'source': SOURCE, 'source_revision': REVISION,
                'source_files': sources, 'sequence_length': 128, 'dtype': '<u4',
                'vocab_size': 50257, 'bos_token_id': 50256, 'eos_token_id': 50256,
                'packing': 'document EOS; continuous 126-token payload wrapped with BOS/EOS; split tails discarded',
                'normalization': 'strip and collapse 3+ newlines to two',
                'split_rule': 'source shard/row order; validation documents first until target; subsequent documents train',
                'deduplication': 'exclude repeated normalized-text SHA256 across both splits',
                'documents': docs, 'skipped_empty_or_duplicate_documents': duplicates,
                'document_manifest_sha256': sha256(out / 'documents.jsonl'),
                'tokenizer_files': {p.name: sha256(p) for p in sorted((out / 'tokenizer').iterdir()) if p.is_file()},
                'splits': splits, 'elapsed_seconds': time.monotonic() - started,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'preparation_script_sha256': sha256(__file__)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
