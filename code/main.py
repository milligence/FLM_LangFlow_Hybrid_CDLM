import json
import os
import itertools
import functools
import argparse
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
torch.load = functools.partial(torch.load, weights_only=False)
from torch.distributed import init_process_group, destroy_process_group
import wandb
import algo
import algorithm_registry
import dataloader
import utils
import run_manifest

import numpy as np
from datetime import datetime

import uuid

# Allow torch.load(weights_only=True) to safely unpickle Hydra configs stored in checkpoints
torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Metadata])

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if 'hf' in config.algo.backbone:
        return diffusion_model(
            config, tokenizer=tokenizer).to('cuda')

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        weights_only=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
            ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)


def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    model.metrics.gen_ppl_per_sample = []
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []
    all_sample_ids = []
    protocol_id = str(getattr(
        config.eval, 'gen_ppl_protocol_id', 'legacy'))

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            sample_tensor = (torch.stack(samples)
                             if isinstance(samples, list) else samples)
            all_sample_ids.extend(
                sample_tensor.detach().cpu().unbind(dim=0))
            text_samples = model.tokenizer.batch_decode(
                sample_tensor, skip_special_tokens=False)
            if config.eval.compute_generative_perplexity:
                if protocol_id == 'owt128-gpt2large-genppl-v1':
                    model.metrics.record_generative_perplexity(
                        sample_tensor,
                        config.model.length,
                        retokenize=False,
                        device=model.device,
                        score_all_nonpadding=True)
                else:
                    model.metrics.record_generative_perplexity(
                        text_samples,
                        config.model.length,
                        device=model.device)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        if config.eval.compute_generative_perplexity:
            generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
    sample_quality = (
        model.metrics.compute_sample_quality(
            all_sample_ids,
            special_token_ids=getattr(model.tokenizer, 'all_special_ids', []))
        if all_sample_ids else {})
    samples_path = config.eval.generated_samples_path
    samples_payload = {
        'generative_ppl': generative_ppl,
        'entropy': entropy,
        'protocol_id': protocol_id,
        'evaluator_model_name_or_path': str(
            config.eval.gen_ppl_eval_model_name_or_path),
        'comparable': bool(config.eval.gen_ppl_comparable),
        'formal_sample_count': int(config.eval.gen_ppl_formal_sample_count),
        'sequence_length': int(config.model.length),
        'nfe': (int(model.last_sampling_nfe)
                if hasattr(model, 'last_sampling_nfe') else None),
        'solver': str(config.sampling.solver),
        'temperature': float(config.sampling.temperature),
        'generation_seed': int(config.seed),
        'weights': str(getattr(
            config.algo, 'task1_eval_weight_role',
            'online' if config.eval.disable_ema else 'ema')),
        'checkpoint_global_step': int(getattr(
            model, '_loaded_checkpoint_global_step', model.global_step)),
        'token_bias_warmup_steps': int(getattr(
            model, 'token_bias_warmup_steps', 0)),
        'token_bias_weight': float(
            model._current_token_bias_weight()
            if hasattr(model, '_current_token_bias_weight') else 0.0),
        'num_samples': len(all_samples),
        'per_sample_scores': model.metrics.gen_ppl_per_sample,
        'sample_quality': sample_quality,
        'generated_token_ids': [
            sample.tolist() for sample in all_sample_ids],
        'generated_seqs': all_samples,
    }
    if hasattr(model, 'task1_same_state_sc_posterior_comparison'):
        comparison = model.task1_same_state_sc_posterior_comparison()
        if comparison is not None:
            samples_payload[
                'same_state_self_conditioning_posterior_comparison'
            ] = comparison
    if hasattr(model, 'task1_sc_gate_statistics'):
        gate_statistics = model.task1_sc_gate_statistics()
        if gate_statistics is not None:
            samples_payload['self_conditioning_gate_statistics'] = (
                gate_statistics)
    if config.algo.name == 'task1_tvm_50k_final':
        temporary_path = f'{samples_path}.tmp'
        with fsspec.open(temporary_path, 'w') as f:
            json.dump(samples_payload, f, indent=4)
            f.write('\n')
        os.replace(temporary_path, samples_path)
    else:
        with fsspec.open(samples_path, 'w') as f:
            json.dump(samples_payload, f, indent=4)
    print('Samples saved at:', samples_path)


def _generate_samples_with_tc(diffusion_model, config, logger,
                              tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for i in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            assert config.loader.eval_batch_size % config.sampling.duplicate == 0
            different_in_batch = config.loader.eval_batch_size // config.sampling.duplicate
            samples = model.restore_model_and_sample(
                num_steps=config.sampling.steps, duplicate=config.sampling.duplicate)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device)
            model.metrics.record_tc([i*different_in_batch + j for _ in range(
                config.sampling.duplicate) for j in range(different_in_batch)], samples)
            all_samples.extend(list(text_samples))

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    generative_ppl = 0.
    entropy = 0.
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        avg_tc, avg_joints, avg_marginals = model.metrics.tc.compute()
        print('Generative perplexity:', generative_ppl)
        print('Sample entropy:', entropy)
        print('Total average correlation:', avg_tc)
        print('Average joint entropy:', avg_joints)
        print('Average marginal entropy:', avg_marginals)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, 'w') as f:
        json.dump({'generative_ppl': generative_ppl,
                   'entropy': entropy,
                   'avg_tc': avg_tc,
                   'avg_joints': avg_joints,
                   'avg_marginals': avg_marginals,
                   'generated_seqs': all_samples}, f, indent=4)
    print('Samples saved at:', samples_path)


@torch.inference_mode()
def generate_reflow_dataset(diffusion_model, config, logger, tokenizer):
    # TODO: implement with lightning_module.test with pseudo-data
    logger.info('Generating samples.')
    model = _load_from_checkpoint(diffusion_model=diffusion_model,
                                  config=config,
                                  tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    # if model.ema:
    #   model.ema.store(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.ema.copy_to(itertools.chain(
    #       model.backbone.parameters(),
    #       model.noise.parameters()))
    #   model.backbone.eval()
    #   model.noise.eval()

    test_ds = dataloader.get_pseudo_dataloader(config, tokenizer, model)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=None,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=None)
    trainer.test(model, test_ds)
    return


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Perplexity Eval.')

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
            wandb_logger = L.pytorch.loggers.CSVLogger(
                save_dir=config.checkpointing.save_dir,
                name='local_metrics', version='')
        else:
            wandb_logger = L.pytorch.loggers.WandbLogger(
                config=omegaconf.OmegaConf.to_object(config),
                ** config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed)
    trainer.validate(model, valid_ds)


@torch.inference_mode()
def _build_task1_sc_pair_bank(diffusion_model, config, logger, tokenizer):
    logger.info('Building continuousized frozen-teacher SC pair bank.')
    model = diffusion_model(config, tokenizer=tokenizer).to('cuda')
    model.setup('sc_pair_bank')
    metadata = model.build_sc_reference_pair_bank(
        str(config.algo.tvm_sc_pair_bank_path))
    print(json.dumps(metadata, indent=2))


@torch.inference_mode()
def _run_task1_sc_diagnostics(diffusion_model, config, logger, tokenizer):
    logger.info('Running finite-map self-conditioning diagnostics.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config,
        tokenizer=tokenizer)
    model.setup('diagnostic')
    if config.eval.disable_ema:
        model.ema = None
    model._eval_mode()
    result = model.run_sc_diagnostics(
        str(config.eval.task1_sc_diagnostics_path),
        sample_count=int(config.eval.task1_sc_diagnostics_samples),
        batch_size=int(config.eval.task1_sc_diagnostics_batch_size))
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def generate_reflow_dataset_with_perturbed_rect(diffusion_model, config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None

    train_ds, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True)

    # i is given by random sequence of train_ds's N
    shuffled_indices = np.random.permutation(len(train_ds.dataset))

    eval_batch_size = config.loader.eval_batch_size
    generate_samples = config.sampling.num_reflow_samples

    x0s = []
    xTs = []
    ts = []

    print("generation start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for j in range(generate_samples // eval_batch_size):
        if config.sampling.semi_ar:
            raise NotImplementedError(
                "Semi-AR sampling is not implemented. Please use standard sampling.")
        else:
            assert eval_batch_size == 1
            x0 = train_ds.dataset[shuffled_indices[j *
                                                   eval_batch_size:(j+1)*eval_batch_size]]['input_ids']
            x0 = torch.from_numpy(x0).to(model.device)
            x1 = torch.randint(0, 50258, x0.shape,
                               device=model.device, dtype=x0.dtype)
            rand_t = torch.randint(
                0, x0.shape[1], (1, ), device=model.device).float().item() / x0.shape[1]
            num_step = max(int(config.sampling.steps * (1 - rand_t)), 1)
            # random interpolate between x1 and x0
            # =y_given_t where y0=noise, y1=data
            xt = torch.where(rand_t > torch.rand(
                x1.shape, device=model.device), x0, x1)

            samples = model.restore_model_and_sample(
                num_steps=num_step, xT=xt.clone(), given_t=rand_t)
            x0s.append(samples.clone())
            xTs.append(xt.clone())
            ts.append(rand_t)
        if j % 500 == 0:
            print(f"Generated {(j+1) * eval_batch_size} samples")
    x0s = torch.cat(x0s, dim=0)
    xTs = torch.cat(xTs, dim=0)
    ts = torch.tensor(ts, device=model.device)

    print("generation end: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    x0s = x0s.cpu().numpy()
    xTs = xTs.cpu().numpy()
    ts = ts.cpu().numpy()

    save_path = config.data.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    xT_path = os.path.join(save_path, 'xT.npy')
    x0_path = os.path.join(save_path, 'x0.npy')
    ts_path = os.path.join(save_path, 'ts.npy')

    np.save(x0_path, x0s)
    np.save(xT_path, xTs)
    np.save(ts_path, ts)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Training.')
    experiment_logger = None
    if config.get('wandb', None) is not None:
        if os.environ.get('WANDB_MODE', '').lower() == 'disabled':
            # A disabled W&B run is intentionally ephemeral. Keep a local,
            # machine-readable metric history so CE/MSE experiments remain
            # auditable without requiring an external account or network.
            experiment_logger = L.pytorch.loggers.CSVLogger(
                save_dir=config.checkpointing.save_dir,
                name=str(getattr(
                    config.get('experiment', {}),
                    'metrics_logger_name', 'local_metrics')),
                version='')
        else:
            wid = config.wandb.get('id')
            if not wid or len(str(wid)) > 16:
                wid = str(uuid.uuid4().hex[:8])
            config.wandb.id = wid
            if config.wandb.get('name'):
                config.wandb.name = f"{config.wandb.name}_{wid}"
            experiment_logger = L.pytorch.loggers.WandbLogger(
                config=omegaconf.OmegaConf.to_object(config),
                ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer,
            config=config,
            weights_only=False)
    else:
        if config.algo.name in {
                'langflow_flm_hybrid', 'task1_tvm_ce',
                'task1_tvm_endpoint500',
                'task1_tvm_sc_repair', 'task1_tvm_joint_j0',
                'task1_tvm_joint_j1', 'task1_posterior_tvm',
                'task1_tvm_50k_final'}:
            # Dataset cache construction may consume process RNG only on the
            # first arm. Re-seed at the pairing boundary so CE and MSE start
            # from byte-identical parameters and sampler state.
            L.seed_everything(config.seed)
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    manifest = None
    process_is_global_zero = int(os.environ.get('LOCAL_RANK', '0')) == 0
    if (config.algo.name in {
            'langflow_flm_hybrid', 'task1_tvm_ce',
            'task1_tvm_endpoint500',
            'task1_tvm_sc_repair', 'task1_tvm_joint_j0',
            'task1_tvm_joint_j1', 'task1_posterior_tvm',
            'task1_tvm_50k_final'}
            and process_is_global_zero):
        manifest = run_manifest.create(config, model)
        run_manifest.write(manifest, config.checkpointing.save_dir)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=experiment_logger)
    # Force weights_only=False to allow full checkpoint restore (PyTorch 2.6 defaults torch.load to weights_only=True)
    try:
        if bool(getattr(config.eval, 'validate_before_training', False)):
            trainer.validate(model, valid_ds)
        trainer.fit(
            model, train_ds, valid_ds,
            ckpt_path=ckpt_path, weights_only=False)
    except Exception as error:
        if manifest is not None:
            run_manifest.finish(
                manifest, config, model, trainer,
                status='failed', error=error)
            run_manifest.write(manifest, config.checkpointing.save_dir)
        raise
    if manifest is not None:
        run_manifest.finish(manifest, config, model, trainer)
        run_manifest.write(manifest, config.checkpointing.save_dir)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    diffusion_model = algorithm_registry.get_algorithm_class(
        config.algo.name)
    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_recon':
        _generate_samples(**kwargs)
    elif config.mode == 'sample_eval_with_tc':
        _generate_samples_with_tc(**kwargs)
    elif config.mode == 'ppl_eval':
        _eval_ppl(**kwargs)
    elif config.mode == 'task1_sc_pair_bank':
        _build_task1_sc_pair_bank(**kwargs)
    elif config.mode == 'task1_sc_diagnostics':
        _run_task1_sc_diagnostics(**kwargs)
    elif config.mode == 'generate_reflow_data':
        generate_reflow_dataset(diffusion_model, config, logger, tokenizer)
    elif config.mode == 'generate_reflow_data_with_perturbed_rect':
        generate_reflow_dataset_with_perturbed_rect(**kwargs)
    else:
        _train(**kwargs)


if __name__ == '__main__':
    # allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
