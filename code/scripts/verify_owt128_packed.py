"""Verify all stored tokens, split membership, file hashes, and loader batches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packed_dataset import PackedTokenDataset
from scripts.prepare_owt128_packed import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = Path(args.directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    seen = set()
    documents = {'train': 0, 'validation': 0}
    for line in (directory / 'documents.jsonl').open():
        row = json.loads(line)
        fingerprint = row['normalized_text_sha256']
        assert fingerprint not in seen, 'duplicate or cross-split document'
        seen.add(fingerprint)
        documents[row['split']] += 1
    assert documents == manifest['documents']
    assert sha256(directory / 'documents.jsonl') == manifest['document_manifest_sha256']
    for name, digest in manifest['tokenizer_files'].items():
        assert sha256(directory / 'tokenizer' / name) == digest
    for name, record in manifest['splits'].items():
        path = directory / record['file']
        assert sha256(path) == record['sha256']
        assert path.stat().st_size == record['bytes'] == record['tokens'] * 4
        data = np.memmap(path, mode='r', dtype='<u4', shape=(record['sequences'], 128))
        for start in range(0, len(data), 8192):
            chunk = data[start:start + 8192]
            assert int(chunk.max()) < 50257
            assert np.all(chunk[:, 0] == 50256) and np.all(chunk[:, -1] == 50256)
        loader = torch.utils.data.DataLoader(PackedTokenDataset(directory, name), batch_size=16)
        batch = next(iter(loader))
        assert batch['input_ids'].dtype == torch.int64
        assert tuple(batch['input_ids'].shape) == (16, 128)
        assert int(batch['attention_mask'].sum()) == 2048
    print(json.dumps({'status': 'ok', 'documents': documents,
                      'tokens': {k: v['tokens'] for k, v in manifest['splits'].items()},
                      'manifest_sha256': sha256(directory / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
