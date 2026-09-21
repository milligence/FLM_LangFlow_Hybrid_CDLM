"""Low-memory offline file check; full model/GPU execution is a separate gate."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    hashes = Path(__file__).resolve().parent / 'gpt2_large_frozen.sha256'
    subprocess.run(['sha256sum', '-c', str(hashes)], cwd=directory, check=True)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    assert (config.n_layer, config.n_embd, config.vocab_size) == (36, 1280, 50257)
    assert tokenizer.vocab_size == 50257
    with safe_open(directory / 'model.safetensors', framework='np') as weights:
        embedding = weights.get_slice('wte.weight')
        assert embedding.get_shape() == [50257, 1280]
        assert np.isfinite(embedding[:2]).all()
        tensor_count = len(weights.keys())
    print(json.dumps({'status': 'ok', 'model': 'gpt2-large', 'tensor_count': tensor_count,
                      'full_model_forward_tested': False,
                      'reason': 'no-GPU instance has a 2 GiB memory limit'}, indent=2))


if __name__ == '__main__':
    main()
