import unittest

import torch

import algo
from langflow_test_support import SamplingHarness, VocabSamplingHarness


class LangFlowSamplingTests(unittest.TestCase):
    def test_sampling_uses_exact_requested_nfe_and_self_conditioning(self):
        model = SamplingHarness()
        samples = model.generate_samples(num_samples=2, num_steps=128)
        self.assertEqual(samples.shape, (2, 6))
        self.assertEqual(model.last_sampling_nfe, 128)
        self.assertEqual(len(model.self_conditions), 128)
        self.assertIsNone(model.self_conditions[0])
        self.assertTrue(all(
            value is not None for value in model.self_conditions[1:]))

    def test_vocab_sampling_c_uses_log_nsr_on_the_same_physical_grid(self):
        model = VocabSamplingHarness()
        model.model_time_condition = 'log_nsr'
        model.generate_samples(num_samples=2, num_steps=7)
        expected_u = torch.arange(7, dtype=torch.float32) / 7.0
        expected_t = expected_u.square()
        actual_condition = torch.stack(
            [record[0][0] for record in model.time_records])
        actual_t = torch.stack(
            [record[1][0] for record in model.time_records])
        torch.testing.assert_close(actual_t, expected_t)
        torch.testing.assert_close(
            actual_condition,
            algo.flm_linear_gamma(expected_t, model.flm_time_eps))

    def test_vocab_sampling_keeps_flm_state_until_langflow_output(self):
        model = VocabSamplingHarness()
        samples = model.generate_samples(num_samples=2, num_steps=7)
        self.assertEqual(samples.shape, (2, 6))
        self.assertEqual(model.last_sampling_nfe, 7)
        self.assertEqual(model.state_shapes, [(2, 6, 5)] * 7)
        self.assertIsNone(model.self_conditions[0])
        self.assertTrue(all(
            value is not None for value in model.self_conditions[1:]))
        expected_u = torch.arange(7, dtype=torch.float32) / 7.0
        actual_condition = torch.stack(
            [record[0][0] for record in model.time_records])
        actual_t = torch.stack(
            [record[1][0] for record in model.time_records])
        torch.testing.assert_close(actual_condition, expected_u)
        torch.testing.assert_close(actual_t, expected_u.square())

    def test_vocab_first_interval_physical_split_preserves_nfe_and_tail(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'first_interval_physical_split')
        model.config.sampling.task1_first_interval_substeps = 4
        u_grid, t_grid, metadata = model._task1_vocab_sampling_grid(12)
        self.assertEqual(u_grid.numel(), 13)
        self.assertEqual(t_grid.numel(), 13)
        self.assertEqual(metadata['removed_uniform_tau_indices'], [8, 9, 10])
        torch.testing.assert_close(t_grid[:5], torch.linspace(0.0, 1 / 144, 5))
        self.assertAlmostEqual(float(u_grid[-2]), 11 / 12)
        self.assertTrue(bool(torch.all(t_grid[1:] > t_grid[:-1])))

    def test_vocab_sampling_can_replay_initial_noise_independent_of_weights(self):
        first = VocabSamplingHarness()
        second = VocabSamplingHarness()
        for model in (first, second):
            model.config.sampling.task1_initial_noise_seed = 42
            model.config.sampling.task1_diagnostic_steps = [1]
            model.generate_samples(num_samples=2, num_steps=7)
        self.assertEqual(
            first._task1_sampling_diagnostic_metadata['initial_state_probe'],
            second._task1_sampling_diagnostic_metadata['initial_state_probe'])

    def test_tau_box_grid_has_frozen_query_ownership_and_final_jump(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'tau_box_physical_equal_0p6_jump')
        model.config.sampling.task1_tau_box_query_counts = [
            32, 23, 23, 24, 12, 13, 1]
        u_grid, t_grid, metadata = model._task1_vocab_sampling_grid(128)
        self.assertEqual(u_grid.numel(), 129)
        self.assertEqual(t_grid.numel(), 129)
        self.assertEqual(metadata['nfe'], 128)
        self.assertEqual(metadata['update_count'], 128)
        self.assertTrue(metadata['update_after_last_query'])
        self.assertEqual(
            metadata['query_count_by_tau_decile'],
            [32, 23, 23, 24, 12, 13, 1, 0, 0, 0])
        self.assertEqual(
            metadata['query_counts_first_seven_boxes'],
            [32, 23, 23, 24, 12, 13, 1])
        self.assertAlmostEqual(float(u_grid[-2]), 0.6, places=6)
        self.assertAlmostEqual(float(t_grid[-2]), 0.36, places=6)
        self.assertEqual(float(u_grid[-1]), 1.0)
        self.assertEqual(float(t_grid[-1]), 1.0)
        for box_index, expected_count in enumerate(
                [32, 23, 23, 24, 12, 13]):
            mask = (u_grid[:-1] >= box_index / 10.0) & (
                u_grid[:-1] < (box_index + 1) / 10.0)
            self.assertEqual(int(mask.sum()), expected_count)

    def test_tau_box_sampling_updates_from_final_query_to_endpoint(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'tau_box_physical_equal_0p6_jump')
        model.config.sampling.task1_tau_box_query_counts = [
            32, 23, 23, 24, 12, 13, 1]
        model.config.sampling.task1_diagnostic_steps = [128]
        model.generate_samples(num_samples=2, num_steps=128)
        self.assertEqual(model.last_sampling_nfe, 128)
        self.assertEqual(len(model.time_records), 128)
        self.assertAlmostEqual(
            float(model.time_records[-1][0][0]), 0.6, places=6)
        self.assertAlmostEqual(
            float(model.time_records[-1][1][0]), 0.36, places=6)
        row = model.task1_finalize_sampling_diagnostics()['rows'][0]
        self.assertEqual(row['step'], 128)
        self.assertAlmostEqual(row['mean_state_rms'], 0.2, places=6)

    def test_first_interval32_balanced_tail_grid_preserves_nfe_and_last_query(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'first_interval32_tail_bins_balanced')
        model.config.sampling.task1_tail_bin_removal_counts = [8, 8, 8, 7]
        u_grid, t_grid, metadata = model._task1_vocab_sampling_grid(128)
        self.assertEqual(u_grid.numel(), 129)
        self.assertEqual(t_grid.numel(), 129)
        self.assertEqual(metadata['nfe'], 128)
        self.assertEqual(metadata['first_interval_substeps'], 32)
        self.assertEqual(metadata['tail_removal_counts'], [8, 8, 8, 7])
        self.assertEqual(len(metadata['removed_uniform_tau_indices']), 31)
        self.assertEqual(
            [row['removal_count'] for row in metadata['removed_by_tau_decile']],
            [8, 8, 8, 7])
        self.assertNotIn(127, metadata['removed_uniform_tau_indices'])
        self.assertAlmostEqual(float(u_grid[-2]), 127 / 128, places=6)
        torch.testing.assert_close(
            t_grid[:33], torch.linspace(0.0, 1 / 16384, 33))
        self.assertTrue(bool(torch.all(t_grid[1:] > t_grid[:-1])))

    def test_first_interval32_balanced_tail_grid_refines_256_and_512_nfe(self):
        grids = {}
        for nfe in (256, 512):
            with self.subTest(nfe=nfe):
                model = VocabSamplingHarness()
                model.config.sampling.task1_time_grid = (
                    'first_interval32_tail_bins_balanced')
                model.config.sampling.task1_tail_bin_removal_counts = [
                    8, 8, 8, 7]
                u_grid, t_grid, metadata = (
                    model._task1_vocab_sampling_grid(nfe))
                self.assertEqual(u_grid.numel(), nfe + 1)
                self.assertEqual(t_grid.numel(), nfe + 1)
                self.assertEqual(metadata['nfe'], nfe)
                self.assertEqual(metadata['update_count'], nfe - 1)
                self.assertFalse(metadata['update_after_last_query'])
                self.assertFalse(metadata['endpoint_update_performed'])
                factor = nfe // 128
                self.assertEqual(metadata['base_grid_nfe'], 128)
                self.assertEqual(metadata['refinement_factor'], factor)
                self.assertEqual(
                    metadata['refinement_coordinate'], 'physical_time')
                self.assertTrue(metadata['base_grid_points_preserved'])
                self.assertEqual(
                    metadata['first_interval_substeps'], 32 * factor)
                self.assertEqual(
                    len(metadata['removed_uniform_tau_indices']), 31)
                self.assertNotIn(
                    127, metadata['removed_uniform_tau_indices'])
                self.assertAlmostEqual(
                    metadata['first_uniform_tau'], 1 / 128, places=7)
                torch.testing.assert_close(
                    t_grid[:32 * factor + 1],
                    torch.linspace(0.0, 1 / 128 ** 2,
                                   32 * factor + 1))
                base_u, base_t, _ = model._task1_vocab_sampling_grid(128)
                torch.testing.assert_close(u_grid[::factor], base_u)
                torch.testing.assert_close(t_grid[::factor], base_t)
                self.assertTrue(bool(torch.all(t_grid[1:] > t_grid[:-1])))
                self.assertEqual(float(u_grid[0]), 0.0)
                self.assertEqual(float(u_grid[-1]), 1.0)
                self.assertEqual(float(t_grid[0]), 0.0)
                self.assertEqual(float(t_grid[-1]), 1.0)
                grids[nfe] = (u_grid, t_grid)
        torch.testing.assert_close(grids[512][0][::2], grids[256][0])
        torch.testing.assert_close(grids[512][1][::2], grids[256][1])

    def test_first_interval64_tail_keep_grid_freezes_all_decile_counts(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'first_interval64_tail_keep_4_3_2_1')
        u_grid, t_grid, metadata = model._task1_vocab_sampling_grid(128)
        self.assertEqual(u_grid.numel(), 129)
        self.assertEqual(t_grid.numel(), 129)
        self.assertEqual(metadata['nfe'], 128)
        self.assertEqual(metadata['update_count'], 127)
        self.assertFalse(metadata['update_after_last_query'])
        self.assertFalse(metadata['endpoint_update_performed'])
        self.assertEqual(metadata['first_interval_substeps'], 64)
        self.assertEqual(
            metadata['query_count_by_tau_decile'],
            [76, 9, 9, 8, 8, 8, 4, 3, 2, 1])
        self.assertEqual(metadata['removed_uniform_tau_indices'], [
            14, 17, 21, 24, 27, 30, 34, 37, 40, 42, 45, 48, 50,
            53, 56, 59, 62, 65, 67, 70, 73, 75, 77, 79, 80, 82,
            83, 84, 86, 87, 89, 90, 91, 93, 94, 95, 97, 98, 99,
            101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 113,
            114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            125, 126])
        self.assertEqual(metadata['preserved_last_uniform_query_index'], 127)
        self.assertAlmostEqual(metadata['final_query_tau'], 127 / 128)
        self.assertAlmostEqual(
            metadata['final_query_physical_time'], (127 / 128) ** 2)
        torch.testing.assert_close(
            t_grid[:65], torch.linspace(0.0, 1 / 16384, 65))

    def test_physical_time_uniform_grid_uses_inverse_paired_tau(self):
        model = VocabSamplingHarness()
        model.config.sampling.task1_time_grid = (
            'physical_time_uniform_official_inverse_lut')
        u_grid, t_grid, metadata = model._task1_vocab_sampling_grid(128)
        expected_t = torch.linspace(0.0, 1.0, 129)
        torch.testing.assert_close(t_grid, expected_t)
        torch.testing.assert_close(u_grid, expected_t.sqrt())
        self.assertEqual(metadata['nfe'], 128)
        self.assertEqual(metadata['update_count'], 127)
        self.assertFalse(metadata['update_after_last_query'])
        self.assertFalse(metadata['endpoint_update_performed'])
        self.assertEqual(
            metadata['tau_pairing'], 'official_inverse_lut_gamma_to_alpha')
        self.assertAlmostEqual(metadata['final_query_physical_time'], 127 / 128)
        self.assertAlmostEqual(metadata['last_update_target_physical_time'],
                               127 / 128)
        self.assertEqual(metadata['endpoint_physical_time'], 1.0)

    def test_physical_time_uniform_grid_supports_256_and_512_nfe(self):
        grids = {}
        for nfe in (256, 512):
            with self.subTest(nfe=nfe):
                model = VocabSamplingHarness()
                model.config.sampling.task1_time_grid = (
                    'physical_time_uniform_official_inverse_lut')
                u_grid, t_grid, metadata = (
                    model._task1_vocab_sampling_grid(nfe))
                expected_t = torch.linspace(0.0, 1.0, nfe + 1)
                torch.testing.assert_close(t_grid, expected_t)
                torch.testing.assert_close(u_grid, expected_t.sqrt())
                self.assertEqual(metadata['nfe'], nfe)
                self.assertEqual(metadata['update_count'], nfe - 1)
                self.assertEqual(
                    metadata['physical_time_grid'],
                    f'linspace(0,1,{nfe + 1})')
                self.assertAlmostEqual(
                    metadata['physical_time_step'], 1 / nfe)
                self.assertEqual(float(u_grid[0]), 0.0)
                self.assertEqual(float(u_grid[-1]), 1.0)
                self.assertEqual(float(t_grid[0]), 0.0)
                self.assertEqual(float(t_grid[-1]), 1.0)
                grids[nfe] = (u_grid, t_grid)
        torch.testing.assert_close(grids[512][0][::2], grids[256][0])
        torch.testing.assert_close(grids[512][1][::2], grids[256][1])


if __name__ == '__main__':
    unittest.main()
