import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from packed_dataset import PackedTokenDataset
from scripts.prepare_owt128_packed import PackedWriter


class PackedTests(unittest.TestCase):
    def test_document_boundaries_and_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.bin'
            writer = PackedWriter(path, 2, length=6, special=9)
            writer.add([1, 2])
            writer.add([3, 4, 5, 6, 7, 8, 0])
            record = writer.finish()
            rows = np.fromfile(path, dtype='<u4').reshape(-1, 6)
            self.assertEqual(rows.tolist(), [[9, 1, 2, 9, 3, 9], [9, 4, 5, 6, 7, 9]])
            self.assertEqual(record['tokens'], 12)
            self.assertEqual(record['discarded_tail_tokens'], 3)

    def test_incomplete_file_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.bin'
            writer = PackedWriter(path, 2)
            writer.add([1])
            with self.assertRaises(RuntimeError):
                writer.finish()
            self.assertFalse(path.exists())

    def test_loader_shape_dtype_mask_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.bin'
            np.full((2, 128), 50256, dtype='<u4').tofile(path)
            (Path(directory) / 'manifest.json').write_text(json.dumps({
                'format': 'owt128-uint32-v1', 'sequence_length': 128,
                'splits': {'train': {'file': 'train.bin', 'sequences': 2}}}))
            dataset = PackedTokenDataset(directory, 'train')
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
            self.assertEqual(tuple(batch['input_ids'].shape), (2, 128))
            self.assertEqual(batch['input_ids'].dtype, torch.int64)
            self.assertEqual(batch['attention_mask'].sum().item(), 256)
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                PackedTokenDataset(directory, 'train')
