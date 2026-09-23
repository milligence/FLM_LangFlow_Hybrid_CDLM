import itertools
import math
import os
import random
import inspect

from dataclasses import dataclass


from tqdm import tqdm
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.cuda.amp import autocast
import torch.distributed as dist
import dataloader
import metrics
import models
import utils
from task1_continuation import cosine_transition_lr, linear_transition_lr
from omegaconf import ListConfig


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = - (1 - self.eps) + t * 0
        assert alpha_t.shape == dalpha_t.shape
        return dalpha_t, alpha_t

def sample_categorical(categorical_probs, temperature=1.0):
    categorical_probs = categorical_probs.float()
    if temperature != 1.0:
        categorical_probs = categorical_probs.pow(1.0 / temperature)
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm.to(categorical_probs.dtype)).argmax(dim=-1)

def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer,
            vocab_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, 'ignore_bos'):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
        if hasattr(self.config.algo, 'loss_type'):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization
        if self.config.algo.backbone == 'dit':
            self.backbone = models.dit.DIT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'no_sc_dit':
            self.backbone = models.dit.NoSCDIT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'hf_dit':
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True)
            
        self._pending_ema_state = None
        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # Noise Schedule
        self.noise = LogLinear()

        loss_type = getattr(
            self.config.algo, 'loss_type', 'cross_entropy')
        objective_is_nll = not (
            (self.config.algo.name == 'flm' and loss_type == 'mse')
            or (self.config.algo.name in {
                    'langflow_flm_hybrid', 'task1_no_sc_mse_teacher',
                    'task1_tvm_50k_final'}
                and loss_type == 'softmax_probability_mse')
            or self.config.algo.name in {
                'task1_tvm_ce', 'task1_tvm_endpoint500',
                'task1_tvm_sc_repair', 'task1_posterior_tvm'})
        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
            objective_is_nll=objective_is_nll)

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
            self._get_parameters(),
            decay=self.config.training.ema)
        else:
            self.ema = None


        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._pending_sampler_state = None
        self._resume_source_optimizer_contract = {}
        self._task1_lr_graft_history = []
        self._task1_lr_graft_record = None
        self.target_tokens = None


    def _validate_configuration(self):
        assert self.config.algo.backbone in {'dit', 'no_sc_dit', 'hf_dit'}
        if self.config.algo.parameterization == 'ar':
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == 'none'

        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != 'score'

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(),
                               self.noise.parameters())

    def _get_optimizer_parameters(self):
        """Parameters optimized by AdamW; subclasses may add non-EMA state."""
        return self._get_parameters()

    def _eval_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            print('Copying EMA parameters to model')
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        else:
            print('No EMA parameters')
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def load_state_dict(self, state_dict, strict=True):
        if any('_orig_mod' in k for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('._orig_mod.', '.')
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('teacher_model.'):
                    filtered_state_dict[k] = v
            state_dict = filtered_state_dict
        
        ret = super().load_state_dict(state_dict, strict=strict)
        
        if self.ema:
            ema_sd = getattr(self, "_pending_ema_state", None)
            ema_loaded = False

            if ema_sd is not None:
                try:
                    self.ema.load_state_dict(ema_sd)
                    # ExponentialMovingAverage stores only trainable parameters.
                    # Apply the identical filter when validating a restored EMA;
                    # otherwise a deliberately frozen parameter shifts every
                    # subsequent shadow/parameter shape comparison.
                    current_params = [
                        parameter for parameter in self._get_parameters()
                        if parameter.requires_grad]

                    if len(self.ema.shadow_params) == len(current_params):
                        shapes_match = all(
                            s.shape == p.shape
                            for s, p in zip(self.ema.shadow_params, current_params)
                        )
                        if shapes_match:
                            ema_loaded = True
                        else:
                            print("[WARNING] EMA shape mismatch - will reinitialize from loaded weights")
                    else:
                        print("[WARNING] EMA count mismatch - will reinitialize from loaded weights")

                except Exception as e:
                    print(f"[WARNING] Failed to load EMA after weights load: {e}")

            if not ema_loaded:
                print("Initializing EMA from loaded model weights")
                import models.ema
                self.ema = models.ema.ExponentialMovingAverage(
                    list(self._get_parameters()),
                    decay=self.config.training.ema
                )

            self._pending_ema_state = None

        return ret

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self._pending_ema_state = checkpoint.get('ema', None)
        self._pending_sampler_state = checkpoint.get('sampler', None)
        self._resume_source_optimizer_contract = dict(
            checkpoint.get('task1_optimizer_contract', {}))
        self._task1_lr_graft_history = list(
            checkpoint.get('task1_lr_graft_history', []))
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed']
        # is 1 iteration behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total'][
            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['current'][
            'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps,
        # not the number of local steps, so we don't multiply with
        # self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                   'state_dict'):
            checkpoint['sampler'].update(
                self.trainer.train_dataloader.sampler.state_dict())
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            pending = self._pending_sampler_state or {}
            if (distributed and 'epoch' in pending and 'counter' in pending):
                dl_sampler.load_state_dict(pending)
            elif (not distributed
                  and pending.get('random_state') is not None
                  and 'counter' in pending):
                dl_sampler.load_state_dict(pending)
            elif (distributed
                  and self.fast_forward_epochs is not None
                  and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                * self.config.loader.batch_size),
                })
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls
        self._pending_sampler_state = None
        self._apply_task1_resume_constant_lr()
        self._apply_task1_resume_lr_transition()

    def _task1_resume_value(self, name):
        experiment = self.config.get('experiment', None)
        if experiment is None:
            return None
        resume = experiment.get('resume', None)
        if resume is None:
            return None
        return resume.get(name, None)

    def _apply_task1_resume_constant_lr(self):
        requested = self._task1_resume_value(
            'constant_target_learning_rate_override')
        if requested is None:
            return
        if not bool(self.config.checkpointing.resume_from_ckpt):
            raise RuntimeError(
                'A constant LR graft is valid only for full-state resume.')
        target_lr = float(requested)
        if not target_lr > 0.0:
            raise ValueError('Resume constant target LR must be positive.')
        declared_source = self._task1_resume_value('source_learning_rate')
        declared_source_step = self._task1_resume_value('source_global_step')
        saved_source = self._resume_source_optimizer_contract.get(
            'target_learning_rate')
        saved_source_step = self._resume_source_optimizer_contract.get(
            'global_optimizer_step')
        if (declared_source is not None and saved_source is not None
                and float(declared_source) != float(saved_source)):
            raise RuntimeError(
                'SOURCE_LEARNING_RATE does not match the checkpoint contract.')
        if (declared_source_step is not None and saved_source_step is not None
                and int(declared_source_step) != int(saved_source_step)):
            raise RuntimeError(
                'SOURCE_GLOBAL_STEP does not match the checkpoint contract.')

        optimizer_lrs_before = []
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                optimizer_lrs_before.append(float(group['lr']))
                group['lr'] = target_lr
                group['initial_lr'] = target_lr

        scheduler_records = []
        for scheduler_config in self.trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            is_lambda = (
                hasattr(scheduler, 'base_lrs')
                and hasattr(scheduler, 'lr_lambdas'))
            is_task1_cosine = isinstance(
                scheduler, utils.CosineDecayWarmupLRScheduler)
            if not (is_lambda or is_task1_cosine):
                raise RuntimeError(
                    'Task1 LR graft requires LambdaLR or the Task1 cosine '
                    'scheduler.')
            last_epoch = int(getattr(
                scheduler, 'last_epoch', scheduler._last_epoch))
            step_count = int(getattr(scheduler, '_step_count', 0))
            if is_lambda:
                count = len(scheduler.base_lrs)
                scheduler.base_lrs = [target_lr] * count
                scheduler.lr_lambdas = [lambda _: 1.0 for _ in range(count)]
            else:
                count = len(scheduler.optimizer.param_groups)
                scheduler._task1_constant_lr = target_lr
            scheduler._last_lr = [target_lr] * count
            scheduler_records.append({
                'class': type(scheduler).__name__,
                'last_epoch_preserved': last_epoch,
                'step_count_preserved': step_count,
                'constant_lr_mode': (
                    'lambda' if is_lambda else 'task1_cosine_override'),
            })

        self.lr = target_lr
        source_lr = (
            float(saved_source) if saved_source is not None
            else (float(declared_source)
                  if declared_source is not None else None))
        record = {
            'applied_at_restored_global_step': int(self.global_step),
            'source_target_learning_rate': source_lr,
            'optimizer_learning_rates_before_graft': optimizer_lrs_before,
            'constant_target_learning_rate': target_lr,
            'scheduler_state': scheduler_records,
            'optimizer_moments_preserved': True,
            'ema_preserved': True,
            'global_step_preserved': True,
            'sampler_and_data_position_preserved': True,
            'lr_warmup_restarted': False,
            'gaussian_bias_warmup_restarted': False,
        }
        self._task1_lr_graft_history.append(record)
        self._task1_lr_graft_record = record

    def _apply_task1_resume_lr_transition(self):
        requested = self._task1_resume_value(
            'transition_target_learning_rate')
        transition_steps = self._task1_resume_value(
            'transition_optimizer_steps')
        if requested is None and transition_steps is None:
            return
        if requested is None or transition_steps is None:
            raise RuntimeError(
                'Resume LR transition requires both target LR and steps.')
        if self._task1_resume_value(
                'constant_target_learning_rate_override') is not None:
            raise RuntimeError(
                'Resume constant LR and LR transition are mutually exclusive.')
        if not bool(self.config.checkpointing.resume_from_ckpt):
            raise RuntimeError(
                'An LR transition is valid only for full-state resume.')

        target_lr = float(requested)
        transition_steps = int(transition_steps)
        transition_schedule = str(
            self._task1_resume_value('transition_schedule') or 'linear')
        declared_source = float(self._task1_resume_value(
            'source_learning_rate'))
        declared_source_step = int(self._task1_resume_value(
            'source_global_step'))
        saved_source = float(self._resume_source_optimizer_contract.get(
            'target_learning_rate', declared_source))
        saved_source_step = int(self._resume_source_optimizer_contract.get(
            'global_optimizer_step', declared_source_step))
        if int(self.global_step) != declared_source_step:
            raise RuntimeError(
                'Restored global step does not match SOURCE_GLOBAL_STEP.')
        if saved_source_step != declared_source_step:
            raise RuntimeError(
                'SOURCE_GLOBAL_STEP does not match the checkpoint contract.')
        if saved_source != declared_source:
            raise RuntimeError(
                'SOURCE_LEARNING_RATE does not match the checkpoint contract.')
        if (not math.isfinite(target_lr) or target_lr <= 0.0
                or target_lr > declared_source):
            raise ValueError(
                'Resume target LR must be finite, positive, and no greater '
                'than the source LR.')
        if transition_steps <= 0:
            raise ValueError('Resume LR transition steps must be positive.')
        if transition_schedule not in {'linear', 'cosine'}:
            raise ValueError(
                'Resume LR transition schedule must be linear or cosine.')

        optimizer_lrs_before = []
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                actual_lr = float(group['lr'])
                optimizer_lrs_before.append(actual_lr)
                if (not math.isfinite(actual_lr) or actual_lr <= 0.0
                        or actual_lr != declared_source):
                    raise RuntimeError(
                        'Restored optimizer LR is zero or inconsistent with '
                        'the declared source LR.')
                group['lr'] = declared_source
                group['initial_lr'] = declared_source

        def multiplier(epoch):
            helper = (
                cosine_transition_lr
                if transition_schedule == 'cosine'
                else linear_transition_lr)
            return helper(
                declared_source, target_lr, declared_source_step,
                transition_steps, epoch) / declared_source

        scheduler_records = []
        for scheduler_config in self.trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if not (hasattr(scheduler, 'base_lrs')
                    and hasattr(scheduler, 'lr_lambdas')):
                raise RuntimeError(
                    'Task1 LR transition requires a LambdaLR scheduler.')
            last_epoch = int(scheduler.last_epoch)
            if abs(last_epoch - declared_source_step) > 1:
                raise RuntimeError(
                    'Restored scheduler position is incompatible with the '
                    'source optimizer step.')
            step_count = int(getattr(scheduler, '_step_count', 0))
            count = len(scheduler.base_lrs)
            scheduler.base_lrs = [declared_source] * count
            scheduler.lr_lambdas = [multiplier for _ in range(count)]
            scheduler._last_lr = [declared_source] * count
            scheduler_records.append({
                'class': type(scheduler).__name__,
                'last_epoch_preserved': last_epoch,
                'step_count_preserved': step_count,
            })

        self.lr = target_lr
        record = {
            'applied_at_restored_global_step': int(self.global_step),
            'source_target_learning_rate': declared_source,
            'optimizer_learning_rates_before_graft': optimizer_lrs_before,
            'transition_target_learning_rate': target_lr,
            'transition_optimizer_steps': transition_steps,
            'transition_schedule': transition_schedule,
            'transition_end_global_step': (
                declared_source_step + transition_steps),
            'scheduler_state': scheduler_records,
            'optimizer_moments_preserved': True,
            'ema_preserved': True,
            'global_step_preserved': True,
            'sampler_and_data_position_preserved': True,
            'lr_warmup_restarted': False,
            'gaussian_bias_warmup_restarted': False,
        }
        self._task1_lr_graft_history.append(record)
        self._task1_lr_graft_record = record

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False):

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn)
        
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert all(metric.mean_value == 0 and metric.weight == 0
                   for metric in self.metrics.train_nlls.values())

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)

        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            current_accumulation_step,
                            train_mode=True,
                            xT=None if 'xT' not in batch else batch['xT'],
                            given_t=batch['given_t'] if 'given_t' in batch else None,
                            not_sampling_t=self.config.training.not_sampling_t
                            )
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='trainer/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return losses.loss

    def on_train_epoch_end(self):
        # NOTE:
        # Originally, this method re-logged validation NLL metrics at the end
        # of every *training* epoch by iterating over `self.metrics.valid_nlls`
        # and calling `.compute()` again.
        #
        # That extra logging turned out to be a non-trivial bottleneck and also
        # caused `val/*` metrics to appear much more frequently in WandB than
        # actual validation runs (which already log in `on_validation_epoch_end`).
        #
        # We therefore keep this hook but make it a no-op to avoid the
        # unnecessary per-train-epoch metric computation/logging. All
        # validation-related metrics are still logged from
        # `on_validation_epoch_end`, which is called whenever validation runs.
        return

    def on_validation_epoch_start(self):
        self.metrics.reset()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        self._eval_mode()
        assert all(metric.mean_value == 0 and metric.weight == 0
                   for metric in self.metrics.valid_nlls.values())

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            xT=None if 'xT' not in batch else batch['xT']
                            )
        self.metrics.update_valid(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        return losses.loss

    def on_validation_epoch_end(self):

        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k,  value=v.compute(), on_step=False,
                     on_epoch=True, sync_dist=True)
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            step_list = self.config.sampling.steps
            if isinstance(step_list, ListConfig):
                step_list = list(step_list)
            elif isinstance(step_list, int):
                step_list = [step_list]

            for num_steps in step_list:
                if hasattr(self.metrics, 'gen_ppl'):
                    self.metrics.gen_ppl.reset()
                if hasattr(self.metrics, 'sample_entropy'):
                    self.metrics.sample_entropy.reset()

                current_text_samples = []

                for _ in range(self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(
                        num_samples=self.config.loader.eval_batch_size,
                        num_steps=num_steps
                    )

                    self.metrics.record_entropy(samples)

                    decoded_batch = self.tokenizer.batch_decode(samples)

                    if len(current_text_samples) < self.config.sampling.num_sample_log:
                        current_text_samples.extend(decoded_batch)

                    if self.config.eval.compute_generative_perplexity:
                        self.metrics.record_generative_perplexity(
                            decoded_batch, self.num_tokens, self.device)

                if self.config.eval.compute_generative_perplexity:
                    self.log(f'val/gen_ppl_T{num_steps}',
                            self.metrics.gen_ppl.compute(),
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True)
                    self.log(f'val/sample_entropy_T{num_steps}',
                            self.metrics.sample_entropy.compute(),
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True)

                if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
                    log_samples = current_text_samples[:self.config.sampling.num_sample_log]

                    self.trainer.logger.log_table(
                        key=f'samples_T{num_steps}@global_step{self.global_step}',
                        columns=['Generated Samples'],
                        data=[[s] for s in log_samples]
                    )

        self._train_mode()

    def on_test_epoch_start(self):
        self._eval_mode()
        self.xTx0s = []

    def test_step(self, batch, batch_idx):
        xT = batch
        x0 = self.generate_samples(xT.shape[0], xT=xT.detach().clone())
        pair = torch.stack([xT, x0], dim=0)  # 2 B N
        self.xTx0s.append(pair)
        return 0.

    def on_test_epoch_end(self):
        # gather across all GPUs
        self.xTx0s = torch.cat(self.xTx0s, dim=1)  # 2 B N
        torch.distributed.barrier()

        # if multi gpu
        if torch.distributed.is_initialized():
            data_xTx0s_all = [torch.empty_like(self.xTx0s) for _ in range(
                torch.distributed.get_world_size())] if self.trainer.global_rank == 0 else None
            torch.distributed.gather(self.xTx0s,
                                     data_xTx0s_all,
                                     dst=0)

        if self.trainer.global_rank == 0:
            xTx0s = torch.cat(data_xTx0s_all, dim=1).cpu()[
                :, :self.config.sampling.num_reflow_samples]
            xTs, x0s = xTx0s[0], xTx0s[1]

            save_path = self.config.data.cache_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            xTs = xTs.cpu().numpy()
            x0s = x0s.cpu().numpy()
            xT_path = os.path.join(save_path, 'xT.npy')
            x0_path = os.path.join(save_path, 'x0.npy')
            np.save(xT_path, xTs)
            np.save(x0_path, x0s)
            print('xT shape:', xTs.shape)
            print('x0 shape:', x0s.shape)
            print('xT saved to:', xT_path)
            print('x0 saved to:', x0_path)
        return
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_optimizer_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                    self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def generate_samples(self, num_samples, num_steps, eps, xT, given_t):
        raise NotImplementedError

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self._eval_mode()

        step_list = self.config.sampling.steps
        if isinstance(step_list, ListConfig):
            step_list = list(step_list)
        elif isinstance(step_list, int):
            step_list = [step_list]
        all_samples = []
        for num_steps in step_list:
            batch_samples = self.generate_samples(
                num_samples=self.config.loader.eval_batch_size,
                num_steps=num_steps,
                eps=eps)
            # batch_samples is a tensor of shape (B, L)
            # Convert to list of tensors (one per sample in batch) for extend
            if isinstance(batch_samples, torch.Tensor):
                batch_samples = [batch_samples[i] for i in range(batch_samples.shape[0])]
            all_samples.extend(batch_samples)
        self._train_mode()
        return all_samples

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False):
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
            x0, valid_tokens)
        loss = self.nll(input_tokens, output_tokens,
                        current_accumulation_step, train_mode)
            

        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return Loss(loss=token_nll,
                    nlls=nlls,
                    prior_loss=0.0,
                    num_tokens=num_tokens)


class Diffusion(TrainerBase):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.noise_removal in {
            'none', 'ancestral', 'greedy', 'flow'}
        assert self.config.training.loss_type in {'elbo', 'low_var', 'mse', 'adaptive_l2', 'flow'}
        if self.config.sampling.noise_removal == 'greedy':
            assert self.sampler != 'analytic'
            assert self.parameterization in {'mean', 'subs'}

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _process_sigma(self, sigma):
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, accum_step):
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps  
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            t = t[:batch_dim]
        return t

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t)

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype,
                         device=self.device)
        sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
        model_output_t0 = self.forward(x0, sigma_t0)
        return - torch.gather(input=model_output_t0,
                              dim=-1,
                              index=x0[:, :, None]).squeeze(-1)

    def nll_per_token(self, model_output, xt, x0, alpha_t,
                      dalpha_t, low_var):
        raise NotImplementedError

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=False):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        xt = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(xt, sigma=sigma)
        utils.print_nans(log_x_theta, 'model_output')
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == 'low_var')

    def _get_score(self, **kwargs):
        del kwargs
        raise NotImplementedError

    def _denoiser_update(self, x, t):
        raise NotImplementedError

    def _analytic_update(self, x, t, dt):
        raise NotImplementedError

    def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
        raise NotImplementedError
    
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ancestral':
                _, x = self._ancestral_update(
                x=x, t=t, dt=dt, p_x0=None)
            elif self.sampler == 'ancestral_cache':
                p_x0_cache, x_next = self._ancestral_update(
                x=x, t=t, dt=dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)
                    or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x=x,t=t, dt=dt)

        t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                            device=self.device)
        if self.config.sampling.noise_removal == 'ancestral':
            if self.sampler == 'analytic':
                x = self._denoiser_update(x=x, t=t0)
            else:
                _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                        p_x0=p_x0_cache,
                                        noise_removal_step=True)
        elif self.config.sampling.noise_removal == 'greedy':
            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
        return x

    @torch.no_grad
    def _semi_ar_sampler(
            self, n_samples, stride_length, num_strides, dt=0.001):
        # TODO(subham): Test this method after refactoring.
        ones = torch.ones(n_samples, dtype=self.dtype,
                          device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self.prior_sample(n_samples, self.num_tokens)
            if target is not None:
                x[:, : -stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)
                        or self.time_conditioning):
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(
                x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = ((
            np.concatenate(intermediate_tokens, axis=1)[:, 1:]
            == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)))
        return (sampling_steps, intermediate_text_samples,
                sequence_lengths)

    def restore_model_and_semi_ar_sample(
            self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # TODO(subham): Test this method after refactoring.
        self._eval_mode()
        (sampling_steps, samples,
         sequence_lengths) = self._semi_ar_sampler(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides,
            dt=dt)
        self._train_mode()
        return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
    def __init__(self, config, tokenizer):
        # NOTE: Ideally, we should do
        # vocab_size = len(tokenizer), so that we account
        # for the special tokens added in dataloader.py.
        # But we use tokenizer.vocab_size so as to to be
        # consistent with the prior checkpoints.
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer,
                         vocab_size=vocab_size)
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        assert not (self.parameterization == 'mean'
                    and self.T == 0)
        if self.T > 0:
            assert self.parameterization in {'mean', 'subs'}
        if self.subs_masking:
            assert self.parameterization == 'mean'

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input. 
          alpha_t: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(
            * x.shape, device=x.device) < 1 - alpha_t
        xt = torch.where(move_indices, self.mask_index, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return self.mask_index * torch.ones(
            * batch_dims, dtype=torch.int64, device=self.device)

    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        if p_x0 is None:
            p_x0 = self.forward(
                x, self._sigma_from_alphat(alpha_t)).exp()

        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self._get_score(x, sigma_t)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        score = self._get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
            i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge


class UniformState(Diffusion):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.time_conditioning
        assert self.parameterization == 'mean'
        if self.config.algo.name != 'distillation':
            assert self.T == 0

    def q_xt(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(
            *x.shape, device=x.device) < 1 - alpha_t
        uniform_tensor = torch.randint(
            0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt  # (B, L) int

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64,
            device=self.device)
