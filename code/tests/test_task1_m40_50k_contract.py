import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from langflow_hybrid.ops import flm_linear_euler_update
    from langflow_hybrid.task1_trajectory import Task1TrajectoryDiagnostics
except ModuleNotFoundError:
    torch = None


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    path = SOURCE_ROOT / 'task1_continuation.py'
    spec = importlib.util.spec_from_file_location('task1_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task1M4050kContractTests(unittest.TestCase):
    def test_cosine_lr_anchors_match_the_frozen_contract(self):
        helper = load_helpers().cosine_transition_lr
        source = 3e-4
        target = 7.5e-5
        expected = {
            40000: 1.0,
            42500: 0.8901650429449554,
            45000: 0.625,
            47500: 0.35983495705504476,
            50000: 0.25,
        }
        for step, ratio in expected.items():
            self.assertAlmostEqual(
                helper(source, target, 40000, 10000, step) / source,
                ratio)

    def test_two_configs_differ_only_in_lr_schedule_identity(self):
        names = ('task1_m40_stable_to_50k.env',
                 'task1_m40_cooldown_to_50k.env')
        configs = []
        for name in names:
            values = {}
            for line in (SOURCE_ROOT / 'configs' / name).read_text().splitlines():
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    values[key] = value
            configs.append(values)
        stable, cooldown = configs
        for key in (
                'TRAINING_TIME_SAMPLING', 'GLOBAL_BATCH_SIZE',
                'MICRO_BATCH_SIZE', 'ACCUMULATE_GRAD_BATCHES',
                'SOURCE_GLOBAL_STEP', 'TARGET_MAX_STEPS',
                'TASK1_TRAINING_RNG_SEED', 'CHECKPOINT_MILESTONE_STEPS'):
            self.assertEqual(stable[key], cooldown[key])
        self.assertEqual(stable['TRAINING_TIME_SAMPLING'],
                         'v1_m_tau25_global256')
        self.assertEqual(stable['CHECKPOINT_MILESTONE_STEPS'],
                         '42500,45000,47500,50000')
        self.assertEqual(stable['RESUME_CONSTANT_LEARNING_RATE'], '3e-4')
        self.assertEqual(cooldown['RESUME_TARGET_LEARNING_RATE'], '7.5e-5')
        self.assertEqual(cooldown['RESUME_LR_TRANSITION_SCHEDULE'], 'cosine')

    def test_supervisor_evaluates_only_requested_checkpoints(self):
        source = (SOURCE_ROOT / 'scripts' /
                  'run_task1_m40_50k_dual_h800.sh').read_text()
        for label in ('m40_stable_45k', 'm40_stable_50k',
                      'm40_cooldown_45k', 'm40_cooldown_50k'):
            self.assertIn(label, source)
        self.assertNotIn('m40_stable_47500', source)
        self.assertIn('not a significance claim', source)
        self.assertIn('eval_task1_m40_50k_trajectory.sh', source)
        self.assertIn('eval_task1_a_physical_t_bins.sh', source)

    def test_supervisor_supports_paired_full_state_recovery(self):
        supervisor = (SOURCE_ROOT / 'scripts' /
                      'run_task1_m40_50k_dual_h800.sh').read_text()
        resume = (SOURCE_ROOT / 'scripts' /
                  'resume_task1_m40_50k.sh').read_text()
        self.assertIn('M40_STABLE_RESUME_CHECKPOINT', supervisor)
        self.assertIn('M40_COOLDOWN_RESUME_CHECKPOINT', supervisor)
        self.assertIn('RESUME_EXPECTED_CHECKPOINT_STEP', supervisor)
        self.assertIn('RESUME_EXPECTED_CHECKPOINT_STEP', resume)
        self.assertIn('--expected-step "$resume_expected_checkpoint_step"',
                      resume)

    def test_m40_preflight_accepts_a_completed_transition(self):
        from test_task1_high_noise_contract import (
            load_verifier, valid_checkpoint)
        checkpoint = valid_checkpoint(step=40000, lr=3e-4)
        checkpoint['task1_optimizer_contract']['scheduler'] = (
            'linear_transition_then_constant')
        checkpoint['hyper_parameters']['config']['optim']['lr'] = 6e-4
        checkpoint['task1_lr_graft_history'] = [{
            'transition_end_global_step': 30500,
            'optimizer_moments_preserved': True,
            'ema_preserved': True,
            'global_step_preserved': True,
            'sampler_and_data_position_preserved': True,
        }]
        result = load_verifier().verify_loaded_checkpoint(
            checkpoint, 40000, 3e-4, require_task1_a=True,
            expected_global_batch=256, allow_completed_transition=True)
        self.assertEqual(result['global_optimizer_step'], 40000)

    def test_m40_recovery_accepts_an_active_transition_without_regrafting(self):
        from test_task1_high_noise_contract import (
            load_verifier, valid_checkpoint)
        checkpoint = valid_checkpoint(step=44000, lr=3e-4)
        checkpoint['task1_optimizer_contract']['scheduler'] = (
            'cosine_transition_then_constant')
        checkpoint['task1_lr_graft_history'] = [{
            'transition_end_global_step': 50000,
            'optimizer_moments_preserved': True,
            'ema_preserved': True,
            'global_step_preserved': True,
            'sampler_and_data_position_preserved': True,
        }]
        result = load_verifier().verify_loaded_checkpoint(
            checkpoint, 44000, require_task1_a=True,
            expected_global_batch=256, allow_completed_transition=True,
            allow_active_transition=True)
        self.assertEqual(result['global_optimizer_step'], 44000)
        resume = (SOURCE_ROOT / 'scripts' /
                  'resume_task1_m40_50k.sh').read_text()
        generic = (SOURCE_ROOT / 'scripts' /
                   'resume_task1_vocab_mse.sh').read_text()
        self.assertIn('--allow-active-transition', resume)
        self.assertIn('RESUME_PRESERVE_CHECKPOINT_LR_SCHEDULER=1', resume)
        self.assertIn('unset RESUME_CONSTANT_LEARNING_RATE', generic)

    def test_real_text_and_free_generation_label_semantics_are_separate(self):
        trajectory = (SOURCE_ROOT / 'langflow_hybrid' /
                      'task1_trajectory.py').read_text()
        physical = (SOURCE_ROOT / 'scripts' /
                    'eval_task1_a_physical_t_bins.sh').read_text()
        self.assertIn('generated tokens are never', trajectory)
        self.assertIn("'labels': 'none for free generation'", trajectory)
        self.assertIn('original_real_text_tokens', physical)
        for metric in ('raw_brier', 'token_ce', 'target_probability'):
            self.assertIn(metric, (SOURCE_ROOT / 'scripts' /
                                   'validate_task1_physical_t_bins.py').read_text())

    @unittest.skipUnless(torch is not None, 'PyTorch is a remote dependency')
    def test_trajectory_records_direct_p_v_tv_flip_sc_and_fp32(self):
        nfe = 512
        t_points = torch.linspace(0.0, 1.0, nfe + 1)
        grid = {
            'name': 'physical_time_uniform_official_inverse_lut',
            'nfe': nfe,
            'tau_points': t_points.sqrt().tolist(),
            'physical_time_points': t_points.tolist(),
            'interval_mid_physical_time_points': (
                (t_points[:-1] + t_points[1:]) / 2).tolist(),
            'interval_mid_tau_points': (
                ((t_points[:-1] + t_points[1:]) / 2).sqrt()).tolist(),
        }
        state = torch.zeros(1, 2, 3)
        diagnostic = Task1TrajectoryDiagnostics(
            grid, state, 42, 'base_seed_plus_sample_index', 'selected', 4,
            fp32_sample_count=1, fp32_node_count=2)
        diagnostic.start_batch(state)
        sc = None
        for index in range(nfe):
            diagnostic.arrive_at_fine_state(index, state)
            probabilities = torch.softmax(torch.tensor(
                [[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]]), dim=-1)
            fp32 = probabilities.clone() if diagnostic.should_run_fp32(index) else None
            diagnostic.observe_prediction(
                index, state, probabilities, t_points[index], 1e-5,
                flm_linear_euler_update, self_conditioning=sc,
                fp32_probabilities=fp32)
            sc = probabilities[..., :2]
            state = flm_linear_euler_update(
                state, probabilities, t_points[index], t_points[index + 1])
        diagnostic.finish_batch(state)
        payload = diagnostic.finalize()
        row = payload['physical_intervals']['rows'][1]
        self.assertIsNotNone(row['probability_change_direct'])
        self.assertIsNotNone(row['velocity_change'])
        self.assertIsNotNone(row['probability_total_variation_mean'])
        self.assertIn('flip_confidence', row)
        self.assertTrue(row['self_conditioning']['input_available'])
        self.assertEqual(len(payload['fp32_forward_sensitivity']['rows']), 2)


if __name__ == '__main__':
    unittest.main()
