from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_env(name):
    values = {}
    path = ROOT / 'configs' / name
    for line in path.read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            values[key] = value
    return values


class FrozenMSEReleaseTests(unittest.TestCase):
    def test_selected_four_stage_route(self):
        a15 = read_env('task1_a_train.env')
        self.assertEqual(a15['LEARNING_RATE'], '6e-4')
        self.assertEqual(a15['MAX_STEPS'], '15000')
        self.assertEqual(a15['TRAINING_TIME_SAMPLING'], 'uniform_tau')

        v30 = read_env('task1_v1_from_15k.env')
        self.assertEqual(v30['SOURCE_GLOBAL_STEP'], '15000')
        self.assertEqual(v30['TARGET_MAX_STEPS'], '30000')
        self.assertEqual(v30['TRAINING_TIME_SAMPLING'], 'v1_staged_quota32')
        self.assertEqual(v30['CHECKPOINT_MILESTONE_STEPS'],
                         '18000,24000,30000')

        m45 = read_env('task1_m_tau25_from_v1_30k.env')
        self.assertEqual(m45['SOURCE_GLOBAL_STEP'], '30000')
        self.assertEqual(m45['RESUME_TARGET_LEARNING_RATE'], '3e-4')
        self.assertEqual(m45['RESUME_LR_TRANSITION_STEPS'], '500')
        self.assertEqual(m45['TRAINING_TIME_SAMPLING'],
                         'v1_m_tau25_global256')
        self.assertEqual(m45['CHECKPOINT_MILESTONE_STEPS'],
                         '35000,40000,45000')

        stable = read_env('task1_m40_stable_to_50k.env')
        self.assertEqual(stable['SOURCE_GLOBAL_STEP'], '40000')
        self.assertEqual(stable['TARGET_MAX_STEPS'], '50000')
        self.assertEqual(stable['RESUME_CONSTANT_LEARNING_RATE'], '3e-4')
        self.assertEqual(stable['TASK1_ARM_ID'], 'M40-STABLE')

        for config in (a15, v30, m45, stable):
            self.assertEqual(config['GLOBAL_BATCH_SIZE'], '256')
            self.assertEqual(config['MICRO_BATCH_SIZE'], '32')
            self.assertEqual(config['ACCUMULATE_GRAD_BATCHES'], '8')

    def test_quota_literals_are_frozen_in_training_code(self):
        source = (ROOT / 'langflow_hybrid' / 'model.py').read_text()
        for start, end, counts in (
                (15000, 18000, [2, 4, 6, 7, 7, 4, 2]),
                (18000, 24000, [1, 3, 5, 8, 8, 5, 2]),
                (24000, 30000, [1, 1, 4, 7, 10, 6, 3])):
            self.assertIn(f'({start}, {end}, {counts})', source)
        continuation = (ROOT / 'task1_continuation.py').read_text()
        self.assertIn('(6, 6, 24, 42, 92, 68, 18)', continuation)

    def test_release_entrypoint_rejects_non_selected_variant(self):
        script = ROOT / 'scripts' / 'train_owt_128_langflow_hybrid.sh'
        environment = os.environ.copy()
        environment['LOSS_VARIANT'] = 'task1_c'
        with tempfile.TemporaryDirectory() as directory:
            environment['FLM_STORAGE_DIR'] = directory
            result = subprocess.run(
                ['bash', str(script)], env=environment,
                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn('supports only LOSS_VARIANT=task1_a', result.stderr)


if __name__ == '__main__':
    unittest.main()
