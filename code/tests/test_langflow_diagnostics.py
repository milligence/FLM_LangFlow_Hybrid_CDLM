import collections
import unittest

import torch

from langflow_test_support import DiagnosticHarness


class LangFlowDiagnosticsTests(unittest.TestCase):
    def test_stratified_validation_tau_balances_ten_bins(self):
        model = DiagnosticHarness()
        counts = torch.zeros(10, dtype=torch.long)
        for batch_idx in range(50):
            model._active_validation_batch_idx = batch_idx
            q = model._sample_validation_q(10)
            counts += torch.bincount((q * 10).long(), minlength=10)
        self.assertEqual(counts.tolist(), [50] * 10)

    def test_stratified_validation_accepts_fine_high_noise_edges(self):
        model = DiagnosticHarness()
        edges = [0.0, 1e-4, 1e-3, 1 / 128, 0.03, 0.1,
                 0.3, 0.6, 1.0]
        model.config.algo.validation_tau_bin_edges = edges
        q = model._sample_validation_q(8)
        for index, value in enumerate(q.tolist()):
            self.assertGreaterEqual(value, edges[index])
            self.assertLess(value, edges[index + 1])

    def test_gamma_bins_record_samples_tokens_and_requested_metrics(self):
        model = DiagnosticHarness()
        q = torch.tensor([0.05, 0.95])
        valid = torch.ones(2, 3)
        diagnostics = {
            name: torch.ones(2, 3) * value
            for value, name in enumerate(
                model._posterior_diagnostic_names(), start=1)}
        model._accumulate_gamma_bins(q, diagnostics, valid)
        self.assertEqual(model._validation_gamma_bins[0, 0], 1)
        self.assertEqual(model._validation_gamma_bins[0, 1], 3)
        self.assertEqual(model._validation_gamma_bins[9, 0], 1)
        self.assertEqual(model._validation_gamma_bins[9, 1], 3)
        self.assertEqual(model._validation_gamma_bins[0, 2], 3)

    def test_gamma_frequency_rows_have_metric_mapping(self):
        model = DiagnosticHarness()
        model._validation_posterior_rows = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list)))
        model._training_token_counts[:] = torch.arange(12)
        q = torch.tensor([0.05, 0.95])
        targets = torch.tensor([[1, 2, 3], [9, 10, 11]])
        valid = torch.ones_like(targets)
        diagnostics = {
            name: torch.ones(2, 3) * value
            for value, name in enumerate(
                model._posterior_diagnostic_names(), start=1)}
        model._accumulate_gamma_bins(
            q, diagnostics, valid, target_tokens=targets)
        rows = model._validation_posterior_rows['0.0-0.1']
        self.assertTrue(rows)
        bucket = next(iter(rows.values()))
        self.assertIn('matrix', bucket)
        self.assertIsInstance(bucket['matrix'], list)
        self.assertEqual(bucket['matrix'][0].shape[-1], 18)

    def test_embedding_geometry_contains_registered_statistics(self):
        model = DiagnosticHarness()
        result = model.embedding_nearest_neighbor_diagnostic(sample_size=8)
        required = {
            'raw_norm_mean', 'raw_norm_std', 'raw_norm_p05',
            'raw_norm_p50', 'raw_norm_p95',
            'nearest_cosine_distance_mean',
            'nearest_cosine_distance_p05',
            'nearest_cosine_distance_p50',
            'nearest_cosine_distance_p95', 'centroid_norm',
            'physical_codebook_mean_norm', 'centered_covariance_trace',
            'centered_effective_rank', 'angular_drift_mean_radians'}
        self.assertTrue(required.issubset(result))
        self.assertTrue(all(torch.isfinite(value) for value in result.values()))


if __name__ == '__main__':
    unittest.main()
