"""Machine-readable run manifests for the OWT-128 paired baseline."""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import omegaconf
import torch


SOURCE_ROOT = Path(__file__).resolve().parent


def _timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def _git(command):
    try:
        return subprocess.check_output(
            ['git', *command], cwd=SOURCE_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _data_identity(config):
    result = {
        'train': str(config.data.train),
        'valid': str(config.data.valid),
        'tokenizer': str(config.data.tokenizer_name_or_path),
        'source_url': str(getattr(config.data, 'source_url', 'unknown')),
        'train_documents': int(getattr(config.data, 'train_documents', 0)),
        'valid_documents': int(getattr(config.data, 'valid_documents', 0)),
        'split_rule': 'lexicographically sorted: first train_documents, then valid_documents',
    }
    data_dir = Path(str(getattr(config.data, 'openwebtext_10k_dir', '')))
    packed_dir = Path(str(getattr(config.data, 'packed_dir', '')))
    packed_manifest = packed_dir / 'manifest.json'
    if packed_manifest.is_file():
        packed = json.loads(packed_manifest.read_text(encoding='utf-8'))
        result.update({
            'packed_dir': str(packed_dir),
            'packed_format': packed.get('format'),
            'sequence_length': packed.get('sequence_length'),
            'splits': {
                name: {
                    key: value for key, value in split.items()
                    if key in {'file', 'sequences', 'tokens', 'bytes'}
                }
                for name, split in packed.get('splits', {}).items()
            },
        })
    if data_dir.is_dir():
        required = result['train_documents'] + result['valid_documents']
        files = sorted(data_dir.glob('*.txt'))[:required]
        result['selected_document_count'] = len(files)
    return result


def create(config, model):
    resolved = omegaconf.OmegaConf.to_container(config, resolve=True)
    status = _git(['status', '--porcelain'])
    manifest = {
        'schema': 'owt128-langflow-flm-hybrid-run-v1',
        'status': 'running',
        'started_at_asia_shanghai': _timestamp(),
        'wall_start_monotonic': time.monotonic(),
        'command': [sys.executable, *sys.argv],
        'source': {
            'root': str(SOURCE_ROOT),
            'commit': _git(['rev-parse', 'HEAD']),
            'branch': _git(['branch', '--show-current']),
            'dirty': bool(status),
            'dirty_paths': status.splitlines() if status else [],
        },
        'environment': {
            'hostname': platform.node(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
        },
        'resolved_config': resolved,
        'data_identity': _data_identity(config),
        'seed': int(config.seed),
        'architecture': {
            'name': str(config.algo.name),
            'sequence_length': int(config.model.length),
            'hidden_size': int(config.model.hidden_size),
            'conditioning_size': int(config.model.cond_dim),
            'layers': int(config.model.n_blocks),
            'heads': int(config.model.n_heads),
            'dropout': float(config.model.dropout),
            'embedding_space_ode': bool(getattr(
                config.algo, 'embedding_state', False)),
            'state_space': str(getattr(
                config.algo, 'state_space', 'embedding')),
            'corruption': str(getattr(
                config.algo, 'corruption', 'langflow_vp_gaussian')),
            'vocab_space_ode': str(getattr(
                config.algo, 'state_space', 'embedding')) == 'vocab',
            'normalized_embeddings': bool(getattr(
                config.algo, 'normalize_embeddings', False)),
            'input_projection': 'learned_vocab_to_hidden',
            'input_output_weights_tied': bool(getattr(
                config.model, 'tie_word_embeddings', False)),
            'time_conditioning': bool(getattr(
                config.algo, 'time_conditioning', False)),
            'time_sampling': str(getattr(
                config.algo, 'time_sampling', 'gumbel_quantile')),
            'training_time_sampling': str(getattr(
                config.algo, 'training_time_sampling', 'uniform_tau')),
            'model_time_condition': str(getattr(
                config.algo, 'model_time_condition', 'gumbel_log_nsr')),
            'gumbel_schedule': (
                {
                    'loc': float(config.algo.gumbel.loc),
                    'scale': float(config.algo.gumbel.scale),
                    'cutoff': float(config.algo.gumbel.cutoff),
                }
                if getattr(config.algo, 'gumbel', None) is not None else None),
            'flm_linear_time_eps': (
                float(config.algo.flm_time_eps)
                if getattr(config.algo, 'flm_time_eps', None) is not None
                else None),
            'self_conditioning': bool(getattr(
                config.algo, 'self_conditioning', False)),
            'self_conditioning_probability': float(getattr(
                config.algo, 'self_condition_probability', 0.0)),
            'tokenwise_bias': bool(getattr(
                config.algo, 'tokenwise_bias', False)),
            'token_bias_warmup_steps': int(getattr(
                config.algo, 'token_bias_warmup_steps', 0)),
            'token_bias_schedule': str(getattr(
                config.algo, 'token_bias_schedule', 'warmup')),
            'classification_prototype_mode': str(getattr(
                config.algo, 'classification_prototype_mode', 'shared')),
            'bias_interpretation': str(getattr(
                config.algo, 'bias_interpretation',
                'matched_gaussian_likelihood')),
            'task1_gaussian_bias_formula': (
                'warmup_weight * t / (1-t)^2 * x_t'
                if str(getattr(
                    config.algo, 'state_space', 'embedding')) == 'vocab'
                else None),
            'codebook_gradient_mode': str(getattr(
                config.algo, 'codebook_gradient_mode', 'all')),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameter_count': sum(
                p.numel() for p in model.parameters() if p.requires_grad),
        },
        'loss_contract': {
            'loss_type': str(config.algo.loss_type),
            'prediction_target': str(config.algo.prediction_target),
            'output_transform': str(config.algo.output_transform),
            'reduction': str(config.algo.reduction),
            'time_weighting': str(config.algo.time_weighting),
            'optimization_scale': (
                float(model.optimization_scale)
                if hasattr(model, 'optimization_scale') else 1.0),
        },
        'ema_contract': {
            'decay': float(config.training.ema),
            'uses_num_update_warmup': True,
            'shadow_initialization': 'clone_initial_trainable_parameters',
        },
        'experiment': omegaconf.OmegaConf.to_container(
            config.get('experiment', {}), resolve=True),
    }
    if hasattr(model, 'task1_training_time_contract'):
        manifest['training_time_sampling_contract'] = (
            model.task1_training_time_contract())
    training_rng_seed = getattr(
        config.get('experiment', {}), 'training_rng_seed', None)
    if training_rng_seed is not None:
        manifest['task1_training_rng_contract'] = {
            'base_seed': int(training_rng_seed),
            'streams': [
                'time_group', 'time_value', 'gaussian_noise',
                'self_conditioning'],
            'checkpointed': True,
            'data_sampler_rng_is_separate': True,
            'evaluation_consumes_training_rng': False,
        }
    return manifest


def finish(manifest, config, model, trainer, status='completed', error=None):
    manifest['status'] = status
    manifest['finished_at_asia_shanghai'] = _timestamp()
    started = manifest.pop('wall_start_monotonic', None)
    if started is not None:
        wall_seconds = max(time.monotonic() - started, 0.0)
        manifest['wall_seconds'] = wall_seconds
    else:
        wall_seconds = None
    global_step = int(getattr(trainer, 'global_step', 0))
    manifest['optimizer_steps'] = global_step
    nominal_tokens = (
        global_step * int(config.loader.global_batch_size)
        * int(config.model.length))
    manifest['nominal_tokens_seen'] = nominal_tokens
    if wall_seconds:
        manifest['nominal_tokens_per_second'] = nominal_tokens / wall_seconds
    if torch.cuda.is_available():
        manifest['peak_cuda_memory_bytes'] = int(
            torch.cuda.max_memory_allocated())
    manifest['embedding_nearest_neighbor_cosine_distance'] = getattr(
        model, 'embedding_diagnostics', {})
    manifest['codebook'] = {
        'gradient_mode': str(getattr(
            model, 'codebook_gradient_mode', 'all')),
        'trainable': bool(getattr(
            getattr(getattr(model, 'backbone', None), 'vocab_embed', None),
            'embedding', torch.empty(0)).requires_grad),
    }
    manifest['gamma_bin_diagnostics'] = getattr(
        model, 'gamma_bin_diagnostics', {})
    manifest['posterior_by_gamma_frequency'] = getattr(
        model, 'posterior_bucket_diagnostics', {})
    manifest['gradient_route_diagnostics'] = getattr(
        model, 'gradient_route_diagnostics', {})
    manifest['optimization_diagnostics'] = getattr(
        model, 'optimization_diagnostics', {})
    manifest['training_time_sampling_diagnostics'] = getattr(
        model, 'task1_training_time_diagnostics', {})
    manifest['nonfinite_counts'] = {
        'loss_events': int(getattr(
            model, '_task1_nonfinite_loss_events', torch.zeros(())).item()),
        'loss_values': int(getattr(
            model, '_task1_nonfinite_loss_values', torch.zeros(())).item()),
        'gradient_events': int(getattr(
            model, '_task1_nonfinite_gradient_events', torch.zeros(())).item()),
        'gradient_tensors': int(getattr(
            model, '_task1_nonfinite_gradient_tensors', torch.zeros(())).item()),
        'gradient_clip_events': int(getattr(
            model, '_task1_gradient_clip_events', torch.zeros(())).item()),
    }
    manifest['resume_contract'] = {
        'source_checkpoint': str(getattr(
            config.get('experiment', {}), 'selected_checkpoint', '')),
        'requested_source_learning_rate': getattr(
            config.get('experiment', {}).get('resume', {}),
            'source_learning_rate', None),
        'requested_source_global_step': getattr(
            config.get('experiment', {}).get('resume', {}),
            'source_global_step', None),
        'requested_constant_target_learning_rate_override': getattr(
            config.get('experiment', {}).get('resume', {}),
            'constant_target_learning_rate_override', None),
        'requested_transition_schedule': getattr(
            config.get('experiment', {}).get('resume', {}),
            'transition_schedule', None),
        'lr_graft_applied': getattr(
            model, '_task1_lr_graft_record', None),
        'lr_graft_history': getattr(
            model, '_task1_lr_graft_history', []),
    }
    manifest['bias_logit_diagnostics'] = getattr(
        model, 'bias_logit_diagnostics', {})
    manifest['prototype_diagnostics'] = getattr(
        model, 'prototype_diagnostics', {})
    if getattr(model, 'ema', None) is not None:
        manifest['ema_contract']['actual_update_count'] = int(
            model.ema.num_updates)
    checkpoint_dir = Path(str(config.checkpointing.save_dir)) / 'checkpoints'
    manifest['checkpoints'] = [
        str(path) for path in sorted(checkpoint_dir.glob('*.ckpt'))]
    if error is not None:
        manifest['error'] = str(error)
    return manifest


def write(manifest, directory):
    path = Path(str(directory)) / 'run_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)
    return path
