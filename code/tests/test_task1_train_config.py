import os
from pathlib import Path
import subprocess
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
LOADER = SOURCE_ROOT / 'scripts' / 'task1_train_config.sh'
CONFIG = SOURCE_ROOT / 'configs' / 'task1_a_train.env'


class Task1TrainConfigTests(unittest.TestCase):
    def resolve(self, profile='h800', **overrides):
        environment = os.environ.copy()
        for name in (
                'HARDWARE_PROFILE', 'GLOBAL_BATCH_SIZE', 'MICRO_BATCH_SIZE',
                'ACCUMULATE_GRAD_BATCHES'):
            environment.pop(name, None)
        environment.update({key: str(value) for key, value in overrides.items()})
        command = (
            f'source "{LOADER}"; '
            f'task1_load_train_config "{CONFIG}"; '
            f'task1_resolve_batch "{profile}" && '
            'printf "%s %s %s" "$GLOBAL_BATCH_SIZE" '
            '"$MICRO_BATCH_SIZE" "$ACCUMULATE_GRAD_BATCHES"'
        )
        return subprocess.run(
            ['bash', '-c', command],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_h800_defaults_to_global_and_micro_batch_64(self):
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '64 64 1')

    def test_micro_batch_override_derives_accumulation(self):
        result = self.resolve(MICRO_BATCH_SIZE=32)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '64 32 2')

    def test_4090_auto_micro_batch_is_eight(self):
        result = self.resolve(profile='4090')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '64 8 8')

    def test_non_divisor_micro_batch_is_rejected(self):
        result = self.resolve(MICRO_BATCH_SIZE=24)
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            'MICRO_BATCH_SIZE must be a positive divisor', result.stderr)

    def test_conflicting_explicit_accumulation_is_rejected(self):
        result = self.resolve(
            MICRO_BATCH_SIZE=32, ACCUMULATE_GRAD_BATCHES=8)
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            'ACCUMULATE_GRAD_BATCHES must equal', result.stderr)


if __name__ == '__main__':
    unittest.main()
