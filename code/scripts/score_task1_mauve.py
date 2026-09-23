"""Add official MAUVE to the two frozen Task1 generation metrics."""

import argparse
import json
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
from packed_dataset import PackedTokenDataset


def load_inputs(samples_path, packed_data_dir, tokenizer, num_samples):
    payload = json.loads(Path(samples_path).read_text())
    generated_ids = payload.get('generated_token_ids', [])
    if len(generated_ids) != num_samples:
        raise ValueError(
            f'Expected {num_samples} generated samples, got '
            f'{len(generated_ids)}.')
    reference = PackedTokenDataset(packed_data_dir, 'validation')
    if len(reference) < num_samples:
        raise ValueError(
            f'Expected at least {num_samples} validation samples, got '
            f'{len(reference)}.')
    reference_ids = [
        reference[index]['input_ids'].tolist()
        for index in range(num_samples)]
    reference_text = tokenizer.batch_decode(
        reference_ids, skip_special_tokens=True)
    generated_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True)
    return payload, reference_text, generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True)
    parser.add_argument('--packed-data-dir', required=True)
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--feature-model-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=1024)
    args = parser.parse_args()

    try:
        import mauve
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            'Task1 MAUVE evaluation requires mauve-text and transformers.') \
            from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True)
    payload, reference_text, generated_text = load_inputs(
        args.samples, args.packed_data_dir, tokenizer, args.num_samples)
    result = mauve.compute_mauve(
        p_text=reference_text,
        q_text=generated_text,
        featurize_model_name=args.feature_model_path,
        device_id=args.device_id,
        max_text_length=128,
        seed=42,
        batch_size=8,
        verbose=True)
    metrics = {
        'checkpoint_global_step': int(payload['checkpoint_global_step']),
        'generative_perplexity': float(payload['generative_ppl']),
        'mean_sample_unigram_entropy_nats': float(
            payload['sample_quality'][
                'mean_sample_unigram_entropy_nats']),
        'mauve': float(result.mauve),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
