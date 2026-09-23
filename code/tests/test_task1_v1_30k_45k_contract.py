import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    import torch
    from langflow_test_support import TrainingTimeHarness
except ModuleNotFoundError:
    torch = None
    TrainingTimeHarness = None


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def load_pure_helpers():
    path = SOURCE_ROOT / 'task1_continuation.py'
    spec = importlib.util.spec_from_file_location('task1_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task1V130k45kContractTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, 'PyTorch is a remote-only dependency')
    def test_c45_global_plan_is_exact_before_micro_slicing(self):
        model = TrainingTimeHarness('v1_q30_frozen_global256')
        model.global_step = 30000
        slices = [
            model._sample_task1_training_times(32, index)[2]
            for index in range(8)]
        group_ids = torch.cat(slices)
        self.assertEqual(
            torch.bincount(group_ids, minlength=7).tolist(),
            [8, 8, 32, 56, 80, 48, 24])
        unshuffled = torch.repeat_interleave(
            torch.arange(7), torch.tensor([8, 8, 32, 56, 80, 48, 24]))
        self.assertFalse(torch.equal(group_ids.cpu(), unshuffled))

    @unittest.skipUnless(torch is not None, 'PyTorch is a remote-only dependency')
    def test_m_tau25_global_plan_is_exact_across_eight_micro_batches(self):
        model = TrainingTimeHarness('v1_m_tau25_global256')
        model.global_step = 30000
        tau_parts = []
        group_parts = []
        for index in range(8):
            tau, _, group_id = model._sample_task1_training_times(32, index)
            tau_parts.append(tau)
            group_parts.append(group_id)
        tau = torch.cat(tau_parts)
        group_ids = torch.cat(group_parts)
        self.assertEqual(
            torch.bincount(group_ids, minlength=7).tolist(),
            [6, 6, 24, 42, 92, 68, 18])
        self.assertTrue(torch.all(tau[group_ids == 4] >= 0.1))
        self.assertTrue(torch.all(tau[group_ids == 4] < 0.4))
        self.assertTrue(torch.all(tau[group_ids == 5] >= 0.4))
        self.assertTrue(torch.all(tau[group_ids == 5] < 0.7))

    def test_lr_transition_anchors(self):
        helper = load_pure_helpers().linear_transition_lr
        self.assertEqual(helper(6e-4, 3e-4, 30000, 500, 30000), 6e-4)
        self.assertEqual(helper(6e-4, 3e-4, 30000, 500, 30500), 3e-4)
        self.assertEqual(helper(6e-4, 3e-4, 30000, 500, 45000), 3e-4)

    def test_pure_global_quota_contract(self):
        helpers = load_pure_helpers()
        self.assertEqual(
            helpers.global_group_counts('v1_q30_frozen_global256'),
            [8, 8, 32, 56, 80, 48, 24])
        self.assertEqual(
            helpers.global_group_counts('v1_m_tau25_global256'),
            [6, 6, 24, 42, 92, 68, 18])

    def test_m_only_stop_does_not_create_c_marker(self):
        script = SOURCE_ROOT / 'scripts' / 'request_task1_v1_45k_stop.sh'
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                ['bash', str(script), directory, 'M', 'eta_over_budget'],
                check=True, capture_output=True, text=True)
            m_marker = Path(directory) / 'control' / 'STOP_M-tau25.json'
            c_marker = Path(directory) / 'control' / 'STOP_C45.json'
            self.assertTrue(m_marker.is_file())
            self.assertFalse(c_marker.exists())
            self.assertEqual(
                json.loads(m_marker.read_text())['reason'], 'eta_over_budget')

    def test_m_after_35k_stop_is_a_separate_recoverable_plan(self):
        script = SOURCE_ROOT / 'scripts' / 'request_task1_v1_45k_stop.sh'
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                ['bash', str(script), directory, 'M_AFTER_35000',
                 'ten_minute_plan'],
                check=True, capture_output=True, text=True)
            planned = (Path(directory) / 'control' /
                       'STOP_M-tau25_AFTER_35000.json')
            request = json.loads(planned.read_text())
            self.assertEqual(request['stop_after_optimizer_step'], 35000)
            self.assertFalse((Path(directory) / 'control' /
                              'STOP_M-tau25.json').exists())
            self.assertFalse((Path(directory) / 'control' /
                              'STOP_C45.json').exists())

    def test_supervisor_freezes_deadlines_and_checkpoint_transfer(self):
        source = (SOURCE_ROOT / 'scripts' /
                  'run_task1_v1_30k_45k_dual_h800.sh').read_text()
        for required in (
                'ETA_DECISION_CUTOFF_SECONDS="${ETA_DECISION_CUTOFF_SECONDS:-17700}"',
                'ALL_GRACEFUL_STOP_SECONDS="${ALL_GRACEFUL_STOP_SECONDS:-18000}"',
                'ABSOLUTE_LIMIT_SECONDS="${ABSOLUTE_LIMIT_SECONDS:-18300}"',
                "'checkpoints_included': any",
                "'online_transfer_required': False",
                "'ppl_in_supervisor': False",
                "'evaluator_required': False"):
            self.assertIn(required, source)
        self.assertNotIn('EVAL_MODEL_DIR', source)
        self.assertNotIn('gpt2', source.lower())
        callback = (SOURCE_ROOT / 'experiment_callbacks.py').read_text()
        self.assertIn("'checkpoint_included': True", callback)
        self.assertIn("'online_transfer_required': False", callback)
        self.assertIn("self._stop_mode = 'planned_after_milestone'", callback)
        self.assertLess(callback.index('self._save(trainer, pl_module, step)'),
                        callback.index("self._stop_mode = 'planned_after_milestone'"))

    def test_two_templates_and_entrypoint_freeze_the_new_contract(self):
        expected = {
            'task1_c45_from_v1_30k.env': (
                'v1_q30_frozen_global256', '45101', 'C45'),
            'task1_m_tau25_from_v1_30k.env': (
                'v1_m_tau25_global256', '45201', 'M-tau25'),
        }
        for name, (sampling, rng_seed, arm) in expected.items():
            values = {}
            for line in (SOURCE_ROOT / 'configs' / name).read_text().splitlines():
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    values[key] = value
            self.assertEqual(values['TRAINING_TIME_SAMPLING'], sampling)
            self.assertEqual(values['TASK1_TRAINING_RNG_SEED'], rng_seed)
            self.assertEqual(values['TASK1_ARM_ID'], arm)
            self.assertEqual(values['SOURCE_GLOBAL_STEP'], '30000')
            self.assertEqual(values['TARGET_MAX_STEPS'], '45000')
            self.assertEqual(
                values['CHECKPOINT_MILESTONE_STEPS'], '35000,40000,45000')
            self.assertEqual(values['RESUME_TARGET_LEARNING_RATE'], '3e-4')
            self.assertEqual(values['RESUME_LR_TRANSITION_STEPS'], '500')

    def test_checkpoint_preflight_rejects_zero_lr_and_old_transition(self):
        from test_task1_high_noise_contract import (
            load_verifier, valid_checkpoint)
        verifier = load_verifier()
        zero = valid_checkpoint(step=30000, lr=6e-4)
        zero['optimizer_states'][0]['param_groups'][0]['lr'] = 0.0
        with self.assertRaisesRegex(AssertionError, 'zero'):
            verifier.verify_loaded_checkpoint(
                zero, 30000, 6e-4, require_task1_a=True,
                expected_global_batch=256)
        incompatible = valid_checkpoint(step=30000, lr=6e-4)
        incompatible['task1_optimizer_contract']['scheduler'] = (
            'linear_transition_then_constant')
        with self.assertRaisesRegex(AssertionError, 'incompatible'):
            verifier.verify_loaded_checkpoint(
                incompatible, 30000, 6e-4, require_task1_a=True,
                expected_global_batch=256)

    def test_ten_minute_report_exposes_three_state_decision(self):
        script = SOURCE_ROOT / 'scripts' / 'report_task1_v1_45k_10min.py'
        cases = [(1.0, 'keep_both'), (1.3, 'stop_m_after_35000'),
                 (3.0, 'stop_m_now')]
        for step_seconds, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'supervisor_manifest.json').write_text(json.dumps({
                    'card_started_at_epoch': 1000,
                    'deadlines_elapsed_seconds': {
                        'eta_decision_cutoff': 17700},
                }))
                for arm in ('C45', 'M-tau25'):
                    run = root / 'training' / arm
                    run.mkdir(parents=True)
                    (run / 'throughput.jsonl').write_text(json.dumps({
                        'optimizer_step': 31000,
                        'step_seconds': step_seconds,
                        'gpu_memory_allocated_bytes': 1,
                        'gpu_memory_reserved_bytes': 2,
                    }) + '\n')
                result = subprocess.run(
                    ['python3', str(script), str(root), '--now-epoch', '1600'],
                    check=True, capture_output=True, text=True)
                report = json.loads(result.stdout)
                self.assertEqual(report['recommendation'], expected)
                self.assertIn('conservative projection',
                              report['single_arm_forecast']['method'])
                if expected == 'keep_both':
                    self.assertIsNone(report['recommended_command'])
                else:
                    self.assertIn('request_task1_v1_45k_stop.sh',
                                  report['recommended_command'])


if __name__ == '__main__':
    unittest.main()
