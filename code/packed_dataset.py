"""Read prepacked OWT tokens without materializing Arrow or masks on disk."""
import json
from pathlib import Path

import numpy as np
import torch


class PackedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, directory, split):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['format'] != 'owt128-uint32-v1' or manifest['sequence_length'] != 128:
            raise ValueError('unsupported packed dataset format/length')
        self.length = manifest['sequence_length']
        record = manifest['splits'][split]
        self.count = record['sequences']
        self.path = directory / record['file']
        if self.path.stat().st_size != self.count * self.length * 4:
            raise ValueError('packed token file size mismatch')
        self.tokens = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.tokens is None:
            self.tokens = np.memmap(self.path, mode='r', dtype='<u4', shape=(self.count, self.length))
        return {'input_ids': torch.from_numpy(self.tokens[index].astype(np.int64)),
                'attention_mask': torch.ones(self.length, dtype=torch.long)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tokens'] = None
        return state
