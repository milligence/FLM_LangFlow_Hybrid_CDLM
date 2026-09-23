import json
import tempfile
import unittest
from pathlib import Path

import torch

import metrics as metrics_module
from scripts.assess_langflow_pair import _terminal_validation_series


class LangFlowEvaluationTests(unittest.TestCase):
    def test_sample_quality_reports_repetition_and_diversity(self):
        samples = torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5, 6],
        ])
        result = metrics_module.Metrics.compute_sample_quality(samples)
        self.assertEqual(
            set(result), {
                'mean_sample_unigram_entropy_nats',
                'distinct_1',
                'distinct_2',
                'repeated_4gram_fraction',
                'max_single_token_fraction',
                'special_token_frequency',
                'sample_to_sample_duplication',
            })
        self.assertGreater(result['repeated_4gram_fraction'], 0.0)
        self.assertGreater(result['distinct_2'], 0.0)
        special = metrics_module.Metrics.compute_sample_quality(
            samples, special_token_ids=[1])
        self.assertGreater(special['special_token_frequency'], 0.0)

    def test_terminal_assessment_recovers_registered_step0_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = {
                'schema': 'owt128-langflow-pair-decision-v1',
                'mse_validation': {
                    'raw_brier_initial': 0.9,
                    'target_probability_initial': 0.1,
                    'target_margin_initial': -0.2,
                },
            }
            (root / 'decision_500.json').write_text(
                json.dumps(decision), encoding='utf-8')

            brier, target, margin, source = _terminal_validation_series(
                root / 'decision_1000.json', [0.7], [0.3], [-0.1])

            self.assertEqual(brier, [0.9, 0.7])
            self.assertEqual(target, [0.1, 0.3])
            self.assertEqual(margin, [-0.2, -0.1])
            self.assertEqual(source, str(root / 'decision_500.json'))

    def test_terminal_assessment_does_not_hide_missing_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'decision_500.json').write_text(json.dumps({
                'schema': 'owt128-langflow-pair-decision-v1',
                'mse_validation': {
                    'raw_brier_initial': 0.9,
                    'target_probability_initial': 0.1,
                    'target_margin_initial': -0.2,
                },
            }), encoding='utf-8')

            recovered = _terminal_validation_series(
                root / 'decision_1000.json', [], [], [])

            self.assertEqual(recovered, ([], [], [], None))

    def test_terminal_assessment_requires_aligned_final_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'decision_500.json').write_text(json.dumps({
                'schema': 'owt128-langflow-pair-decision-v1',
                'mse_validation': {
                    'raw_brier_initial': 0.9,
                    'target_probability_initial': 0.1,
                    'target_margin_initial': -0.2,
                },
            }), encoding='utf-8')

            recovered = _terminal_validation_series(
                root / 'decision_1000.json', [0.8, 0.7], [0.3], [-0.1])

            self.assertEqual(recovered, ([0.8, 0.7], [0.3], [-0.1], None))


if __name__ == '__main__':
    unittest.main()
