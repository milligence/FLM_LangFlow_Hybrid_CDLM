import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = SOURCE_ROOT / 'configs'
ENTRYPOINT = SOURCE_ROOT / 'scripts' / 'resume_task1_a_high_noise.sh'
VERIFIER = SOURCE_ROOT / 'scripts' / 'verify_task1_checkpoint.py'


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        'task1_checkpoint_verifier', VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_env(path):
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            values[key] = value
    return values


def valid_checkpoint(step=15000, lr=6e-4, model_time_condition='tau'):
    config = {
        'algo': {
            'name': 'langflow_flm_hybrid',
            'state_space': 'vocab',
            'corruption': 'flm_linear_gaussian',
            'model_time_condition': model_time_condition,
            'self_condition_probability': 0.25,
            'token_bias_schedule': 'warmup',
            'classification_prototype_mode': 'direct_vocab_state',
            'loss_type': 'softmax_probability_mse',
            'prediction_target': 'clean_token_one_hot',
            'output_transform': 'softmax',
            'reduction': 'vocab_sum_then_valid_token_mean',
            'optimization_scale': 'one_half',
        },
        'model': {
            'length': 128,
            'hidden_size': 768,
            'n_blocks': 12,
            'tie_word_embeddings': False,
        },
        'data': {'train': 'openwebtext-packed-v1'},
        'loader': {'global_batch_size': 256},
    }
    return {
        'state_dict': {'backbone.weight': object()},
        'loops': {'fit_loop': object()},
        'optimizer_states': [{
            'state': {'parameter': object()},
            'param_groups': [{'lr': lr}]}],
        'lr_schedulers': [{'last_epoch': step}],
        'ema': {'shadow_params': [object()]},
        'global_step': step,
        'hybrid_token_bias_step': step,
        'sampler': {'random_state': object(), 'counter': step * 256},
        'task1_optimizer_contract': {
            'target_learning_rate': lr,
            'actual_learning_rates': [lr],
            'lr_warmup_optimizer_steps': 2500,
            'gaussian_bias_warmup_optimizer_steps': 5000,
            'global_optimizer_step': step,
            'scheduler': 'constant_after_warmup',
        },
        'task1_lr_graft_history': [],
        'hybrid_token_bias_schedule': 'warmup',
        'hyper_parameters': {'config': {
            **config, 'optim': {'lr': lr}}},
    }


class Task1HighNoiseContractTests(unittest.TestCase):
    def test_two_templates_freeze_paired_15k_to_30k_contract(self):
        expected = {
            'task1_h0_from_15k.env': ('high_noise_quota32', '15000'),
            'task1_v1_from_15k.env': ('v1_staged_quota32', '15000'),
        }
        for name, (sampling, source) in expected.items():
            with self.subTest(name=name):
                values = read_env(CONFIG_DIR / name)
                self.assertEqual(values['TRAINING_TIME_SAMPLING'], sampling)
                self.assertEqual(values['SOURCE_GLOBAL_STEP'], source)
                self.assertEqual(values['TARGET_MAX_STEPS'], '30000')
                for key, value in {
                        'SOURCE_LEARNING_RATE': '6e-4',
                        'RESUME_CONSTANT_LEARNING_RATE': '6e-4',
                        'GLOBAL_BATCH_SIZE': '256',
                        'MICRO_BATCH_SIZE': '32',
                        'ACCUMULATE_GRAD_BATCHES': '8',
                        'TRAINER_DEVICES': '1',
                        'CHECKPOINT_EVERY_N_STEPS': '',
                        'TASK1_VARIANT': 'A'}.items():
                    self.assertEqual(values[key], value)
                expected_milestones = (
                    '20000,25000,30000' if sampling == 'high_noise_quota32'
                    else '18000,24000,30000')
                self.assertEqual(
                    values['CHECKPOINT_MILESTONE_STEPS'],
                    expected_milestones)
        self.assertFalse(
            (CONFIG_DIR / 'task1_high_noise_from_6k.env').exists())

    def test_checkpoint_preflight_accepts_15k_full_state(self):
        verifier = load_verifier()
        result = verifier.verify_loaded_checkpoint(
            valid_checkpoint(), 15000, 6e-4,
            require_task1_a=True, expected_global_batch=256)
        self.assertEqual(result['global_optimizer_step'], 15000)
        self.assertTrue(result['optimizer_state_present'])
        self.assertTrue(result['ema_state_present'])

    def test_checkpoint_preflight_rejects_wrong_identity_step_and_lr(self):
        verifier = load_verifier()
        cases = [
            (valid_checkpoint(model_time_condition='log_nsr'),
             15000, 6e-4, 'model_time_condition'),
            (valid_checkpoint(10000), 15000, 6e-4, 'Expected global step'),
            (valid_checkpoint(lr=3e-4), 15000, 6e-4, 'Expected target LR'),
        ]
        for checkpoint, step, lr, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(AssertionError, message):
                    verifier.verify_loaded_checkpoint(
                        checkpoint, step, lr, require_task1_a=True,
                        expected_global_batch=256)

    def test_loss_keeps_single_full_batch_reduction(self):
        source = (SOURCE_ROOT / 'langflow_hybrid' / 'model.py').read_text()
        self.assertIn(
            "optimized = diagnostics['raw_brier'] * self.optimization_scale",
            source)
        self.assertIn('return optimized', source)
        self.assertIn("'extra_group_loss_weighting': False", source)
        for forbidden in ('importance_weight', 'group_mean'):
            self.assertNotIn(forbidden, source)

    def test_v1_schedule_and_tau_regions_are_frozen(self):
        source = (SOURCE_ROOT / 'langflow_hybrid' / 'model.py').read_text()
        for start, end, counts in (
                (15000, 18000, [2, 4, 6, 7, 7, 4, 2]),
                (18000, 24000, [1, 3, 5, 8, 8, 5, 2]),
                (24000, 30000, [1, 1, 4, 7, 10, 6, 3])):
            self.assertIn(f'({start}, {end}, {counts})', source)
        self.assertIn(
            "'tau_ranges': {4: (0.1, 0.4), 5: (0.4, 0.7), "
            "6: (0.7, 1.0)}", source)

    def test_entrypoint_rejects_wrong_batch_target_and_existing_output(self):
        config = CONFIG_DIR / 'task1_h0_from_15k.env'
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'source.ckpt'
            checkpoint.write_bytes(b'not-loaded-for-prevalidation-failures')
            existing = Path(directory) / 'existing'
            existing.mkdir()
            base = os.environ.copy()
            base.update({
                'TASK1_TRAIN_CONFIG': str(config),
                'RESUME_CHECKPOINT_PATH': str(checkpoint),
                'TASK1_RUN_ID': 'test-high-noise',
                'TASK1_PYTHON_BIN': '/usr/bin/true',
            })
            wrong_batch = dict(base, GLOBAL_BATCH_SIZE='64',
                               RUN_DIR=str(Path(directory) / 'new'))
            result = subprocess.run(
                ['bash', str(ENTRYPOINT)], env=wrong_batch,
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn('GLOBAL_BATCH_SIZE=256', result.stderr)

            wrong_layout = dict(
                base, MICRO_BATCH_SIZE='64',
                ACCUMULATE_GRAD_BATCHES='8',
                RUN_DIR=str(Path(directory) / 'wrong-layout'))
            result = subprocess.run(
                ['bash', str(ENTRYPOINT)], env=wrong_layout,
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn('32/8, 64/4, or 128/2', result.stderr)

            uniform = dict(
                base, TRAINING_TIME_SAMPLING='uniform_tau',
                RUN_DIR=str(Path(directory) / 'uniform'))
            result = subprocess.run(
                ['bash', str(ENTRYPOINT)], env=uniform,
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn('H0 or V1', result.stderr)

            for target in ('15000', '29999', '50001'):
                invalid_target = dict(
                    base, TARGET_MAX_STEPS=target,
                    RUN_DIR=str(Path(directory) / f'target-{target}'))
                result = subprocess.run(
                    ['bash', str(ENTRYPOINT)], env=invalid_target,
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn('final global step 30000', result.stderr)

            existing_output = dict(base, RUN_DIR=str(existing))
            result = subprocess.run(
                ['bash', str(ENTRYPOINT)], env=existing_output,
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn('already exists', result.stderr)

    def test_entrypoint_exposes_bounded_h800_layouts(self):
        source = ENTRYPOINT.read_text(encoding='utf-8')
        self.assertIn('32/8|64/4|128/2', source)
        self.assertNotIn('256/1', source)

    def test_target_is_final_step_30000_with_explicit_milestones(self):
        source = ENTRYPOINT.read_text(encoding='utf-8')
        self.assertIn('TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-30000}"', source)
        self.assertIn('TARGET_MAX_STEPS" != 30000', source)
        resume = (SOURCE_ROOT / 'scripts' /
                  'resume_task1_vocab_mse.sh').read_text(encoding='utf-8')
        self.assertIn('CHECKPOINT_MILESTONE_STEPS', resume)
        self.assertIn('previous_step="$SOURCE_GLOBAL_STEP"', resume)
        self.assertIn('final checkpoint milestone', resume)


if __name__ == '__main__':
    unittest.main()
