import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_task1_gamma_bins import validate_posterior


class ValidateTask1GammaBinsTests(unittest.TestCase):
    def write_posterior(self, directory, counts):
        path = Path(directory) / 'posterior.json'
        path.write_text(json.dumps({
            'checkpoint_global_step': 30000,
            'posterior_by_tau': {
                '30000': {
                    f'bin-{index}': {'sample_count': count}
                    for index, count in enumerate(counts)
                }
            },
        }), encoding='utf-8')
        return path

    def test_accepts_exact_512_sequences_per_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_posterior(directory, [512.0] * 10)
            result = validate_posterior(path, 10, 512)
        self.assertEqual(result['samples_per_tau_bin'], 512)
        self.assertEqual(result['total_samples'], 5120)

    def test_rejects_any_underfilled_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_posterior(
                directory, [512.0] * 9 + [511.0])
            with self.assertRaisesRegex(ValueError, 'bin-9=511'):
                validate_posterior(path, 10, 512)

    def test_reads_only_the_loaded_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_posterior(directory, [512.0] * 10)
            data = json.loads(path.read_text(encoding='utf-8'))
            data['posterior_by_tau']['15000'] = {
                f'bin-{index}': {'sample_count': 2.0}
                for index in range(10)
            }
            path.write_text(json.dumps(data), encoding='utf-8')
            result = validate_posterior(path, 10, 512)
        self.assertEqual(result['checkpoint_global_step'], 30000)


if __name__ == '__main__':
    unittest.main()
