"""Minimal checkpoint state for the LangFlow hybrid."""

import copy


class LangFlowDiagnosticsMixin:
    """Persist only state required for exact Task1 continuation."""

    def on_save_checkpoint(self, checkpoint):
        checkpoint['hybrid_token_bias_step'] = int(self.global_step)
        optimizer_lrs = [
            float(group['lr'])
            for optimizer in self.trainer.optimizers
            for group in optimizer.param_groups]
        transition_target = self._task1_resume_value(
            'transition_target_learning_rate')
        transition_schedule = self._task1_resume_value(
            'transition_schedule')
        checkpoint['task1_optimizer_contract'] = {
            'target_learning_rate': float(
                transition_target if transition_target is not None
                else self.config.optim.lr),
            'actual_learning_rates': optimizer_lrs,
            'scheduler': (
                f'{transition_schedule or "linear"}_transition_then_constant'
                if transition_target is not None
                else 'constant_after_warmup'),
            'lr_warmup_optimizer_steps': int(
                self.config.lr_scheduler.get(
                    'num_warmup_steps',
                    self.config.lr_scheduler.get('warmup_t', 0))),
            'gaussian_bias_warmup_optimizer_steps': int(
                self.token_bias_warmup_steps),
            'global_optimizer_step': int(self.global_step),
        }
        checkpoint['task1_lr_graft_history'] = copy.deepcopy(
            self._task1_lr_graft_history)
        checkpoint['task1_training_rng_state'] = (
            self.task1_training_rng_state())
        if hasattr(self, 'learned_gumbel_state'):
            checkpoint['task1_learned_gumbel'] = (
                self.learned_gumbel_state())
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self._loaded_checkpoint_global_step = int(checkpoint.get(
            'hybrid_token_bias_step', checkpoint.get('global_step', 0)))
        self._task1_pending_training_rng_state = copy.deepcopy(
            checkpoint.get('task1_training_rng_state'))
        super().on_load_checkpoint(checkpoint)
