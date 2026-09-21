#!/usr/bin/env python3
"""Offline integration checks for the local OpenWebText-1k pipeline."""

import json
import os

import torch
from omegaconf import OmegaConf

import dataloader


def main():
    storage_dir = os.environ.get('FLM_STORAGE_DIR')
    if not storage_dir:
        raise SystemExit('Set FLM_STORAGE_DIR before verifying OWT-128 data')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    config = OmegaConf.create({
        'data': {
            'tokenizer_name_or_path': 'gpt2',
            'openwebtext_10k_dir': (
                f'{storage_dir}/data/openwebtext-10k/documents'),
            'train_documents': 900,
            'valid_documents': 100,
        },
    })
    tokenizer = dataloader.get_tokenizer(config)
    cache_dir = (
        f'{storage_dir}/data/huggingface/datasets/openwebtext-1k-local')
    common = {
        'dataset_name': 'openwebtext-1k-local-v2',
        'tokenizer': tokenizer,
        'wrap': True,
        'cache_dir': cache_dir,
        'insert_eos': True,
        'block_size': 128,
        'num_proc': 2,
        'streaming': False,
        'config': config,
    }
    train = dataloader.get_dataset(mode='train', **common)
    valid = dataloader.get_dataset(mode='validation', **common)

    batch_size = 16
    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=False)
    first_batch = next(iter(loader))
    last_batch_size = len(list(loader.batch_sampler)[-1])

    assert len(train) > 0
    assert len(valid) > 0
    assert tuple(first_batch['input_ids'].shape) == (batch_size, 128)
    assert tuple(first_batch['attention_mask'].shape) == (batch_size, 128)
    expected_last_batch_size = len(train) % batch_size or batch_size
    assert last_batch_size == expected_last_batch_size
    assert torch.all(first_batch['input_ids'][:, 0] == tokenizer.bos_token_id)
    assert torch.all(first_batch['input_ids'][:, -1] == tokenizer.eos_token_id)

    print(json.dumps({
        'status': 'ok',
        'train_sequences': len(train),
        'validation_sequences': len(valid),
        'sequence_length': first_batch['input_ids'].shape[1],
        'batch_size': batch_size,
        'last_batch_size': last_batch_size,
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }, indent=2))


if __name__ == '__main__':
    main()
