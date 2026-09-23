import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import experiment_callbacks


class _Strategy:
    def barrier(self, name):
        self.name = name


class _CheckpointTrainer:
    def __init__(self):
        self.is_global_zero = True
        self.global_step = 0
        self.strategy = _Strategy()
        self.saved = []

    def save_checkpoint(self, path, weights_only=None):
        self.saved.append((path, weights_only))
        Path(path).write_text('checkpoint')


class CEScaleCallbackTests(unittest.TestCase):
    def test_adaptive_validation_uses_registered_optimizer_steps(self):
        trainer = SimpleNamespace(
            accumulate_grad_batches=2,
            global_step=0,
            val_check_batch=None)
        callback = experiment_callbacks.AdaptiveValidationIntervalCallback()
        callback.on_train_start(trainer, None)
        self.assertEqual(trainer.val_check_batch, 1000)
        trainer.global_step = 5000
        callback.on_validation_end(trainer, None)
        self.assertEqual(trainer.val_check_batch, 4000)

    def test_milestone_checkpoints_and_final_are_not_duplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _CheckpointTrainer()
            callback = experiment_callbacks.MilestoneCheckpointCallback(
                directory=root / 'checkpoints',
                queue_path=root / 'eval_queue.jsonl',
                global_batch_size=256,
                sequence_length=128,
                milestones=[0, 1000, 2000])
            callback.on_train_start(trainer, None)
            callback.on_train_batch_start(trainer, None, None, 0)
            trainer.global_step = 1000
            callback.on_train_batch_end(trainer, None, None, None, 0)
            callback.on_train_end(trainer, None)
            self.assertEqual(len(trainer.saved), 2)
            self.assertTrue(all(
                weights_only is True for _, weights_only in trainer.saved))
            records = [
                json.loads(line) for line in
                (root / 'eval_queue.jsonl').read_text().splitlines()]
            self.assertEqual(
                [record['optimizer_step'] for record in records], [0, 1000])
            self.assertEqual(
                records[-1]['cumulative_tokens_seen'], 32_768_000)

    def test_final_non_milestone_checkpoint_is_saved_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _CheckpointTrainer()
            callback = experiment_callbacks.MilestoneCheckpointCallback(
                directory=root / 'checkpoints',
                queue_path=root / 'eval_queue.jsonl',
                global_batch_size=256,
                sequence_length=128,
                milestones=[0, 1000])
            callback.on_train_start(trainer, None)
            callback.on_train_batch_start(trainer, None, None, 0)
            trainer.global_step = 750
            callback.on_train_end(trainer, None)
            self.assertEqual(len(trainer.saved), 2)
            records = [
                json.loads(line) for line in
                (root / 'eval_queue.jsonl').read_text().splitlines()]
            self.assertTrue(records[-1]['final'])
            self.assertEqual(records[-1]['optimizer_step'], 750)

    def test_milestone_can_save_full_resume_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = _CheckpointTrainer()
            callback = experiment_callbacks.MilestoneCheckpointCallback(
                directory=root / 'checkpoints',
                queue_path=root / 'eval_queue.jsonl',
                global_batch_size=64,
                sequence_length=128,
                milestones=[2000],
                save_weights_only=False)
            callback.on_train_start(trainer, None)
            trainer.global_step = 2000
            callback.on_train_batch_end(trainer, None, None, None, 0)
            self.assertEqual(trainer.saved, [
                (str(root / 'checkpoints' / 'step_002000.ckpt'), False)])
            record = json.loads(
                (root / 'eval_queue.jsonl').read_text().strip())
            self.assertFalse(record['save_weights_only'])


if __name__ == '__main__':
    unittest.main()
