"""Lightning callbacks used by FLM training experiments."""

import json
import os
import time

import lightning
import torch


class GradientInspectionCallback(lightning.Callback):
    def __init__(self, num_grads_log):
        self.num_grads_log = 10

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        gradients = []
        for name, param in pl_module.backbone.blocks.named_parameters():
            gradients.append(param.grad.view(-1))

        if gradients:
            grads = torch.cat((gradients))
            if not hasattr(pl_module, 'grad_accum_buffer'):
                pl_module.grad_step = torch.tensor(
                    0, device=pl_module.device)
                pl_module.grad_accum_buffer = torch.zeros(
                    self.num_grads_log,
                    grads.shape[0],
                    device=pl_module.device)
            pl_module.grad_accum_buffer[pl_module.grad_step] = grads
            pl_module.grad_step += 1

        if (hasattr(pl_module, 'grad_accum_buffer')
                and pl_module.grad_step == self.num_grads_log):
            grads = pl_module.grad_accum_buffer
            grad_var = grads.std(0).mean()
            pl_module.log(name='trainer/grad_var',
                          value=grad_var.item(),
                          on_step=True,
                          on_epoch=False,
                          sync_dist=True)
            # TODO: save the grads tensor as a numpy array
            # and visualize mean, median, top-k
            pl_module.grad_accum_buffer.zero_()
            pl_module.grad_step = 0


class CUDAPeakMemoryCallback(lightning.Callback):
    """Print peak CUDA allocator usage for reproducible batch-size probes."""

    def on_train_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_end(self, trainer, pl_module):
        if not torch.cuda.is_available() or not trainer.is_global_zero:
            return
        gib = 1024 ** 3
        allocated = torch.cuda.max_memory_allocated(pl_module.device) / gib
        reserved = torch.cuda.max_memory_reserved(pl_module.device) / gib
        print(
            'CUDA_PEAK_MEMORY '
            f'allocated_gib={allocated:.3f} reserved_gib={reserved:.3f}')


class OptimizerStepTimerCallback(lightning.Callback):
    """Persist rank-zero end-to-end optimizer-step throughput as JSONL."""

    def __init__(self, output_path, global_batch_size, sequence_length,
                 dataset_tokens, every_n_steps=1):
        self.output_path = str(output_path)
        self.global_batch_size = int(global_batch_size)
        self.sequence_length = int(sequence_length)
        self.dataset_tokens = int(dataset_tokens)
        self.every_n_steps = int(every_n_steps)
        self._last_step = 0
        self._last_time = None
        self._started_at = None
        self._last_batch_end = None
        self._phase_records = []
        self._current_phase = None

    @staticmethod
    def _synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_start(self, trainer, pl_module):
        del pl_module
        self._synchronize()
        self._last_step = int(trainer.global_step)
        self._last_logged_step = self._last_step
        self._last_time = time.monotonic()
        self._started_at = self._last_time
        self._last_batch_end = self._last_time
        if trainer.is_global_zero:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def _event(self):
        if not torch.cuda.is_available():
            return time.monotonic()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _elapsed(start, end):
        if start is None or end is None:
            return 0.0
        if isinstance(start, float):
            return max(float(end - start), 0.0)
        return max(float(start.elapsed_time(end)) / 1000.0, 0.0)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, pl_module, batch, batch_idx
        now = time.monotonic()
        self._current_phase = {
            'dataloader_seconds': max(now - self._last_batch_end, 0.0),
            'forward_start': self._event(),
            'forward_end': None,
            'backward_end': None,
            'optimizer_start': None,
            'batch_end': None,
        }

    def on_before_backward(self, trainer, pl_module, loss):
        del trainer, pl_module, loss
        if self._current_phase is not None:
            self._current_phase['forward_end'] = self._event()

    def on_after_backward(self, trainer, pl_module):
        del trainer, pl_module
        if self._current_phase is not None:
            self._current_phase['backward_end'] = self._event()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del trainer, pl_module, optimizer
        if self._current_phase is not None:
            self._current_phase['optimizer_start'] = self._event()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        if self._current_phase is not None:
            self._current_phase['batch_end'] = self._event()
            self._phase_records.append(self._current_phase)
            self._current_phase = None
        self._last_batch_end = time.monotonic()
        step = int(trainer.global_step)
        if step <= self._last_step:
            return
        self._last_step = step
        if not trainer.is_global_zero:
            self._phase_records.clear()
            return
        if (step % self.every_n_steps != 0
                and step != int(trainer.max_steps)):
            return
        self._synchronize()
        now = time.monotonic()
        steps_in_window = max(
            step - getattr(self, '_last_logged_step', 0), 1)
        window_seconds = now - self._last_time
        step_seconds = window_seconds / steps_in_window
        self._last_time = now
        self._last_logged_step = step
        tokens_per_step = self.global_batch_size * self.sequence_length
        forward_seconds = sum(self._elapsed(
            item['forward_start'], item['forward_end'])
            for item in self._phase_records)
        backward_seconds = sum(self._elapsed(
            item['forward_end'], item['backward_end'])
            for item in self._phase_records)
        optimizer_seconds = sum(self._elapsed(
            item['optimizer_start'], item['batch_end'])
            for item in self._phase_records)
        dataloader_seconds = sum(
            item['dataloader_seconds'] for item in self._phase_records)
        self._phase_records.clear()
        record = {
            'optimizer_step': step,
            'wall_clock_seconds': now - self._started_at,
            'step_seconds': step_seconds,
            'optimizer_steps_in_window': steps_in_window,
            'global_batch_size': self.global_batch_size,
            'sequence_length': self.sequence_length,
            'cumulative_tokens_seen': step * tokens_per_step,
            'effective_data_passes': (
                step * tokens_per_step / self.dataset_tokens),
            'aggregate_tokens_per_second': (
                steps_in_window * tokens_per_step / window_seconds),
            'dataloader_seconds_per_step': (
                dataloader_seconds / steps_in_window),
            'forward_seconds_per_step': forward_seconds / steps_in_window,
            'backward_and_ddp_seconds_per_step': (
                backward_seconds / steps_in_window),
            'optimizer_seconds_per_step': optimizer_seconds / steps_in_window,
            'gpu_count': int(trainer.num_devices * trainer.num_nodes),
            'per_gpu_tokens_per_second': (
                tokens_per_step / step_seconds
                / int(trainer.num_devices * trainer.num_nodes)),
            'gpu_memory_allocated_bytes': int(
                torch.cuda.memory_allocated(pl_module.device))
                if torch.cuda.is_available() else 0,
            'gpu_memory_reserved_bytes': int(
                torch.cuda.memory_reserved(pl_module.device))
                if torch.cuda.is_available() else 0,
        }
        with open(self.output_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')


class AdaptiveValidationIntervalCallback(lightning.Callback):
    """Use 500-step validation through 5k, then switch to every 2k."""

    def __init__(self, early_every_n_steps=500, early_until_step=5000,
                 later_every_n_steps=2000):
        self.early_every_n_steps = int(early_every_n_steps)
        self.early_until_step = int(early_until_step)
        self.later_every_n_steps = int(later_every_n_steps)
        self._switched = False

    @staticmethod
    def _batches(trainer, optimizer_steps):
        accumulation = trainer.accumulate_grad_batches
        if not isinstance(accumulation, int):
            raise TypeError(
                'Adaptive validation requires integer gradient accumulation.')
        return int(optimizer_steps) * accumulation

    def on_train_start(self, trainer, pl_module):
        del pl_module
        trainer.val_check_batch = self._batches(
            trainer, self.early_every_n_steps)

    def on_validation_end(self, trainer, pl_module):
        del pl_module
        if (not self._switched
                and int(trainer.global_step) >= self.early_until_step):
            trainer.val_check_batch = self._batches(
                trainer, self.later_every_n_steps)
            self._switched = True


class MilestoneCheckpointCallback(lightning.Callback):
    """Save evaluation checkpoints at the registered optimizer steps."""

    def __init__(self, directory, queue_path, global_batch_size,
                 sequence_length, milestones, save_weights_only=True,
                 last_every_n_steps=0, last_filename='last.ckpt',
                 stop_marker_path='', transfer_queue_directory='',
                 planned_stop_marker_path='', run_directory='',
                 launcher_log_path='', arm_id=''):
        self.directory = str(directory)
        self.queue_path = str(queue_path)
        self.global_batch_size = int(global_batch_size)
        self.sequence_length = int(sequence_length)
        self.milestones = sorted({int(step) for step in milestones})
        self.save_weights_only = bool(save_weights_only)
        self.last_every_n_steps = int(last_every_n_steps)
        self.last_filename = str(last_filename)
        self.stop_marker_path = str(stop_marker_path)
        self.planned_stop_marker_path = str(planned_stop_marker_path)
        self.transfer_queue_directory = str(transfer_queue_directory)
        self.run_directory = str(run_directory)
        self.launcher_log_path = str(launcher_log_path)
        self.arm_id = str(arm_id)
        self._saved = {}
        self._last_saved_step = None
        self._stop_requested = False
        self._stop_mode = ''
        self._stop_reason = ''
        self._planned_stop_target_step = None

    @staticmethod
    def _atomic_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _transfer_record(self, trainer, pl_module, step, path, final=False):
        if not trainer.is_global_zero or not self.transfer_queue_directory:
            return
        run_dir = self.run_directory or os.path.dirname(self.directory)
        metadata_path = os.path.join(
            self.transfer_queue_directory, f'step_{step:06d}.metadata.json')
        metadata = {
            'schema': 'task1-v1-30k-45k-checkpoint-metadata-v1',
            'arm': self.arm_id,
            'optimizer_step': int(step),
            'checkpoint': str(path),
            'final': bool(final),
            'save_weights_only': self.save_weights_only,
            'actual_learning_rates': [
                float(group['lr'])
                for optimizer in trainer.optimizers
                for group in optimizer.param_groups],
            'training_time_sampling_diagnostics': getattr(
                pl_module, 'task1_training_time_diagnostics', {}),
            'stop_mode': self._stop_mode or None,
            'stop_reason': self._stop_reason or None,
            'planned_stop_target_step': self._planned_stop_target_step,
        }
        self._atomic_json(metadata_path, metadata)
        candidates = [
            str(path), metadata_path,
            os.path.join(run_dir, 'run_manifest.json'),
            os.path.join(run_dir, '.hydra', 'config.yaml'),
            os.path.join(run_dir, '.hydra', 'overrides.yaml'),
            os.path.join(run_dir, 'throughput.jsonl'),
            self.launcher_log_path,
        ]
        whitelist = [item for item in candidates if item and os.path.isfile(item)]
        record = {
            'schema': 'task1-v1-30k-45k-transfer-ready-v1',
            'arm': self.arm_id,
            'optimizer_step': int(step),
            'checkpoint': str(path),
            'checkpoint_included': True,
            'final': bool(final),
            'online_transfer_required': False,
            'recovery_mode': (
                'later_gpu_free_instance_from_persistent_storage'),
            'whitelist': whitelist,
        }
        ready_path = os.path.join(
            self.transfer_queue_directory, f'step_{step:06d}.ready.json')
        self._atomic_json(ready_path, record)

    def _record(self, trainer, step, path, final=False):
        if not trainer.is_global_zero:
            return
        record = {
            'optimizer_step': int(step),
            'cumulative_tokens_seen': (
                int(step) * self.global_batch_size * self.sequence_length),
            'checkpoint_path': str(path),
            'final': bool(final),
            'save_weights_only': self.save_weights_only,
        }
        with open(self.queue_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    def _save(self, trainer, pl_module, step, final=False):
        if step in self._saved:
            if final:
                self._transfer_record(
                    trainer, pl_module, step, self._saved[step], final=True)
            return self._saved[step]
        path = os.path.join(self.directory, f'step_{step:06d}.ckpt')
        trainer.save_checkpoint(
            path, weights_only=self.save_weights_only)
        self._saved[step] = path
        self._record(trainer, step, path, final=final)
        self._transfer_record(trainer, pl_module, step, path, final=final)
        return path

    def _save_last(self, trainer, step, force=False):
        if self.last_every_n_steps <= 0:
            return None
        if (not force and (step <= 0
                           or step % self.last_every_n_steps != 0)):
            return None
        if step == self._last_saved_step:
            return os.path.join(self.directory, self.last_filename)
        path = os.path.join(self.directory, self.last_filename)
        temporary_path = f'{path}.tmp'
        trainer.save_checkpoint(temporary_path, weights_only=False)
        if trainer.is_global_zero:
            os.replace(temporary_path, path)
        trainer.strategy.barrier('rolling_last_checkpoint')
        self._last_saved_step = step
        return path

    def on_train_start(self, trainer, pl_module):
        del pl_module
        if trainer.is_global_zero:
            os.makedirs(self.directory, exist_ok=True)
            os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
            if self.transfer_queue_directory:
                os.makedirs(self.transfer_queue_directory, exist_ok=True)
        trainer.strategy.barrier('milestone_checkpoint_directory')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del batch, batch_idx
        if 0 in self.milestones:
            self._save(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del outputs, batch
        step = int(trainer.global_step)
        if step in self.milestones:
            self._save(trainer, pl_module, step)
        self._save_last(trainer, step)
        accumulation_boundary = (
            (int(batch_idx) + 1) % int(trainer.accumulate_grad_batches) == 0)
        if not accumulation_boundary:
            return
        if self.stop_marker_path and os.path.isfile(self.stop_marker_path):
            with open(self.stop_marker_path, encoding='utf-8') as handle:
                request = json.load(handle)
            self._stop_requested = True
            self._stop_mode = 'immediate_at_optimizer_step_boundary'
            self._stop_reason = str(request.get('reason', 'requested_stop'))
            trainer.should_stop = True
            return
        if (self.planned_stop_marker_path
                and os.path.isfile(self.planned_stop_marker_path)):
            with open(self.planned_stop_marker_path, encoding='utf-8') as handle:
                request = json.load(handle)
            target = int(request['stop_after_optimizer_step'])
            if step >= target:
                self._stop_requested = True
                self._stop_mode = 'planned_after_milestone'
                self._stop_reason = str(
                    request.get('reason', 'planned_stop_after_milestone'))
                self._planned_stop_target_step = target
                trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step not in self._saved:
            self._save(trainer, pl_module, step, final=True)
        elif self._stop_requested:
            self._transfer_record(
                trainer, pl_module, step, self._saved[step], final=True)
        self._save_last(trainer, step, force=True)
        if trainer.is_global_zero and self.transfer_queue_directory:
            state = {
                'schema': 'task1-v1-30k-45k-arm-state-v1',
                'arm': self.arm_id,
                'optimizer_step': step,
                'target_step': int(trainer.max_steps),
                'status': (
                    'intentional_stop' if self._stop_requested
                    and step < int(trainer.max_steps) else 'completed'),
                'stop_marker': self.stop_marker_path,
                'planned_stop_marker': self.planned_stop_marker_path,
                'stop_mode': self._stop_mode or None,
                'stop_reason': self._stop_reason or None,
                'planned_stop_target_step': self._planned_stop_target_step,
            }
            self._atomic_json(os.path.join(
                self.transfer_queue_directory, 'arm_state.json'), state)


class GradientNormCallback(lightning.Callback):
    """Log unscaled gradient norms before clipping and optimizer update."""

    def __init__(self, every_n_steps=0):
        self.every_n_steps = int(every_n_steps)

    @staticmethod
    def _global_l2_norm(parameters, device):
        squared_norm = torch.zeros((), device=device, dtype=torch.float32)
        found_gradient = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            found_gradient = True
            gradient = parameter.grad.detach().float()
            squared_norm.add_(gradient.square().sum())
        if not found_gradient:
            return None
        return squared_norm.sqrt()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del optimizer
        if (self.every_n_steps <= 0
                or trainer.global_step % self.every_n_steps != 0):
            return
        total_norm = self._global_l2_norm(
            pl_module.backbone.parameters(), pl_module.device)
        output_norm = self._global_l2_norm(
            pl_module.backbone.output_layer.parameters(), pl_module.device)
        embedding_norm = self._global_l2_norm(
            pl_module.backbone.vocab_embed.parameters(), pl_module.device)
        if total_norm is not None:
            pl_module.log(
                'diagnostics/grad_norm', total_norm,
                on_step=True, on_epoch=False, sync_dist=True)
        if output_norm is not None:
            pl_module.log(
                'diagnostics/output_grad_norm', output_norm,
                on_step=True, on_epoch=False, sync_dist=True)
        if embedding_norm is not None:
            pl_module.log(
                'diagnostics/embedding_grad_norm', embedding_norm,
                on_step=True, on_epoch=False, sync_dist=True)


__all__ = [
    'AdaptiveValidationIntervalCallback',
    'CUDAPeakMemoryCallback',
    'GradientInspectionCallback',
    'GradientNormCallback',
    'MilestoneCheckpointCallback',
    'OptimizerStepTimerCallback',
]
