"""Real-repository correctness preflight for the final F/P contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def _load_contract_validator(contract_dir):
    path = contract_dir / 'tools' / 'validate_contract.py'
    spec = importlib.util.spec_from_file_location('task1_contract_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_tests(repo, contract_dir):
    commands = (
        [sys.executable, '-m', 'pytest', '-q',
         'tests/test_task1_tvm_50k_final.py'],
        [sys.executable, '-m', 'pytest', '-q',
         str(contract_dir / 'tests' / 'test_reference_math.py'),
         str(contract_dir / 'tests' / 'test_sampler_control.py')],
    )
    records = []
    environment = os.environ.copy()
    environment['TASK1_TVM_CONTRACT_DIR'] = str(contract_dir)
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (
        str(contract_dir), environment.get('PYTHONPATH', ''))))
    for command in commands:
        result = subprocess.run(
            command, cwd=repo, text=True, capture_output=True,
            env=environment)
        records.append({
            'argv': command, 'returncode': result.returncode,
            'stdout_tail': result.stdout[-8000:],
            'stderr_tail': result.stderr[-8000:],
        })
        if result.returncode:
            break
    return records


def _full_model_config():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'model': {
            'hidden_size': 768, 'cond_dim': 128, 'n_blocks': 12,
            'n_heads': 12, 'dropout': 0.0, 'qk_norm': False,
            'scale_by_sigma': True, 'tie_word_embeddings': False,
        },
        'algo': {
            'causal_attention': False, 'embedding_state': False,
            'self_conditioning': True, 'codebook_gradient_mode': 'all',
            'classification_prototype_mode': 'direct_vocab_state',
            'double_temb': False, 'learnable_loss_weighting': False,
            'finite_map_only_sc': False,
            'task1_finite_time_conditioning': True,
            'task1_finite_init_seed': 20260922,
            'task1_tvm_final_line': 'F',
        },
        'is_di4c': False, 'is_di4c_deterministic': True,
    })


def _relative_l2(candidate, reference):
    import torch
    return float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12))


def _full_model_jvp():
    import torch
    from models.dit import DIT
    if not torch.cuda.is_available():
        raise RuntimeError('Full-model precheck requires CUDA.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260921)
    model = DIT(_full_model_config(), vocab_size=50257).cuda().eval()
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation.weight.normal_(0.0, 0.02)
            block.adaLN_modulation.bias.normal_(0.0, 0.02)
        model.output_layer.adaLN_modulation.weight.normal_(0.0, 0.02)
        model.output_layer.adaLN_modulation.bias.normal_(0.0, 0.02)
        model.output_layer.linear.weight.normal_(0.0, 0.02)
        model.finite_time_conditioner.output.weight.normal_(0.0, 0.02)
        model.finite_time_conditioner.output.bias.normal_(0.0, 0.02)
        model.finite_correction_head.weight.normal_(0.0, 0.02)
        model.finite_correction_head.bias.normal_(0.0, 0.02)
    generator = torch.Generator(device='cuda')
    generator.manual_seed(901)
    state = torch.randn(
        1, 128, 768, device='cuda', generator=generator)
    zeros = torch.zeros_like(state)
    cases = ((0.0, 0.5), (0.7, 0.02), (0.7, (0.95 - 0.7) / 0.3))
    rows = []
    for r_value, eta_value in cases:
        sigma = torch.tensor([r_value], device='cuda')
        features = torch.tensor(
            [[r_value, eta_value]], device='cuda')
        direction = torch.tensor([[0.0, 1.0]], device='cuda')

        def reference(value):
            return model(
                state, sigma, inputs_are_embeddings=True,
                use_jvp_attn=True, finite_time_features=value,
                return_output_features=True)

        reference_primal, reference_tangent = torch.func.jvp(
            reference, (features,), (direction,))
        layer_primal, layer_tangent = model.forward_with_jvp(
            state, zeros, sigma, sigma_jvp=torch.zeros_like(sigma),
            inputs_are_embeddings=True, finite_time_features=features,
            finite_time_features_jvp=direction,
            return_output_features=True)
        primal_error = _relative_l2(layer_primal[0], reference_primal[0])
        tangent_error = _relative_l2(layer_tangent[0], reference_tangent[0])
        if primal_error > 0.005 or tangent_error > 0.03:
            raise AssertionError(
                f'Full-model JVP failed at r={r_value}, eta={eta_value}: '
                f'primal={primal_error}, tangent={tangent_error}.')
        rows.append({
            'r': r_value, 'eta': eta_value,
            'primal_relative_l2': primal_error,
            'tangent_relative_l2': tangent_error,
        })
        del reference_primal, reference_tangent, layer_primal, layer_tangent
        torch.cuda.empty_cache()
    return {
        'passed': True, 'pairs': rows,
        'parameter_count': sum(value.numel() for value in model.parameters()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }


def _data_check(bindings):
    from packed_dataset import PackedTokenDataset
    packed = Path(bindings['data']['owt_train'])
    train = PackedTokenDataset(packed, 'train')
    validation = PackedTokenDataset(packed, 'validation')
    first_train = train[0]
    first_validation = validation[0]
    train_tokens = first_train['input_ids'] if isinstance(first_train, dict) else first_train
    validation_tokens = (
        first_validation['input_ids']
        if isinstance(first_validation, dict) else first_validation)
    if len(train_tokens) != 128 or len(validation_tokens) != 128:
        raise AssertionError('Packed data sequence length is not 128.')
    return {
        'passed': True, 'train_sequences': len(train),
        'validation_sequences': len(validation), 'sequence_length': 128,
    }


def _source_check(repo, manifest):
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    failures = []
    checked = 0
    for item in payload.get('files', []):
        path = (repo / item['path']).resolve()
        if not path.is_relative_to(repo):
            failures.append(f"path_escape:{item['path']}")
            continue
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            failures.append(f"sha256:{item['path']}")
        checked += 1
    if checked == 0:
        failures.append('empty_manifest')
    return {'passed': not failures, 'checked_files': checked,
            'failures': failures}


def _binding_check(bindings, repo, manifest_digest):
    failures = []
    if bindings.get('status') != 'RESOLVED':
        failures.append('status')
    if Path(bindings.get('repo_root') or '').resolve() != repo:
        failures.append('repo_root')
    paths = {
        'python_executable': bindings.get('python_executable'),
        'owt_train': bindings.get('data', {}).get('owt_train'),
        'owt_validation': bindings.get('data', {}).get('owt_validation'),
        'split_manifest': bindings.get('data', {}).get('split_manifest'),
        'tokenizer': bindings.get('data', {}).get('tokenizer'),
        'mse50k_eval_checkpoint': bindings.get('legacy', {}).get(
            'mse50k_eval_checkpoint'),
        'gpt2large_evaluator': bindings.get('legacy', {}).get(
            'gpt2large_evaluator'),
    }
    for name, value in paths.items():
        if not value or not Path(value).expanduser().exists():
            failures.append(name)
    provenance = bindings.get('provenance', {})
    required = (
        'code_commit', 'source_sha256_manifest', 'data_manifest_sha256',
        'tokenizer_sha256', 'tau_lut_sha256', 'legacy_config_sha256',
        'environment_lock', 'source_manifest_path')
    for name in required:
        if not provenance.get(name):
            failures.append(f'provenance.{name}')
    if provenance.get('source_sha256_manifest') != manifest_digest:
        failures.append('provenance.source_sha256_manifest_mismatch')
    return {'passed': not failures, 'resolved_paths': paths,
            'failures': failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contract_dir = args.contract_dir.resolve()
    bindings = json.loads(args.bindings.read_text(encoding='utf-8'))
    repo = Path(bindings['repo_root']).resolve()
    validator = _load_contract_validator(contract_dir)
    static = validator.validate(contract_dir)
    manifest = Path(bindings['provenance']['source_manifest_path'])
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sources = _source_check(repo, manifest)
    binding = _binding_check(bindings, repo, manifest_digest)
    tests = _run_tests(repo, contract_dir)
    failures = []
    if not static['passed']:
        failures.append('static_contract')
    if not sources['passed']:
        failures.append('source_manifest')
    if not binding['passed']:
        failures.append('runtime_bindings')
    if any(record['returncode'] for record in tests):
        failures.append('targeted_tests')
    data = None
    full_model = None
    try:
        data = _data_check(bindings)
    except Exception as error:
        failures.append('packed_data')
        data = {'passed': False, 'error': repr(error)}
    try:
        full_model = _full_model_jvp()
    except Exception as error:
        failures.append('full_model_jvp')
        full_model = {'passed': False, 'error': repr(error)}
    try:
        import flash_attn
        import torch
        environment = {
            'python': sys.version, 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'flash_attn': getattr(flash_attn, '__version__', 'unknown'),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as error:
        failures.append('environment')
        environment = {'error': repr(error)}
    report = {
        'scope': 'repository_full_model',
        'passed': not failures,
        'accepted_lines': ['F', 'P'] if not failures else [],
        'contract_sha256': validator.contract_hash(contract_dir),
        'source_manifest_sha256': manifest_digest,
        'tests': {
            'static_contract': static,
            'source_manifest': sources,
            'runtime_bindings': binding,
            'targeted_pytest': tests,
            'packed_data': data,
            'full_model_cuda_jvp': full_model,
        },
        'environment': environment,
        'limitations': [
            'No performance benchmark was rerun; this precheck is correctness-only.',
            'Long-horizon generation quality is measured only by the declared milestone matrix.',
        ],
        'failures': failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, args.output)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
