import unittest

import torch

from langflow_test_support import TrainingTimeHarness


class Task1TimeSamplingTests(unittest.TestCase):
    def test_learned_gumbel_controls_sampling_from_the_first_step(self):
        model = TrainingTimeHarness('learned_gumbel')
        q, physical_t, group_id = model._sample_task1_training_times(32, 0)
        gamma = model._learned_gumbel_icdf(q).detach()
        torch.testing.assert_close(
            physical_t, torch.sigmoid(-0.5 * gamma))
        self.assertEqual(group_id.unique().tolist(), [-2])
        self.assertFalse(physical_t.requires_grad)
        self.assertGreaterEqual(float(q.min()), model.gumbel_cutoff)
        self.assertLessEqual(float(q.max()), 1.0 - model.gumbel_cutoff)

    def test_high_noise_quota_is_exact_and_shuffled(self):
        model = TrainingTimeHarness('high_noise_quota32')
        torch.manual_seed(7)
        _, _, group_id = model._sample_task1_training_times(32, 0)
        self.assertEqual(
            torch.bincount(group_id, minlength=5).tolist(),
            [1, 4, 4, 17, 6])
        unshuffled = torch.tensor(
            [0] + [1] * 4 + [2] * 4 + [3] * 17 + [4] * 6)
        self.assertFalse(torch.equal(group_id.cpu(), unshuffled))

    def test_physical_time_branch_returns_inverse_paired_tau(self):
        model = TrainingTimeHarness('high_noise_quota32')
        torch.manual_seed(11)
        tau, physical_t, group_id = model._sample_task1_training_times(32, 0)
        torch.testing.assert_close(physical_t, tau.square())
        self.assertTrue(torch.all(tau[group_id == 0] == 0))
        self.assertTrue(torch.all(physical_t[group_id == 0] == 0))
        first_tau = 1.0 / 128.0
        self.assertTrue(torch.all(tau[group_id == 1] < first_tau))
        self.assertTrue(torch.all(tau[group_id == 2] <= first_tau))
        self.assertTrue(torch.all(tau[group_id == 3] >= first_tau))
        self.assertTrue(torch.all(tau[group_id == 3] < 0.1))
        self.assertTrue(torch.all(tau[group_id == 4] >= 0.1))
        self.assertTrue(torch.all(tau[group_id == 4] < 1.0))

    def test_uniform_tau_delegates_to_the_legacy_sampler(self):
        model = TrainingTimeHarness('uniform_tau')
        tau, physical_t, group_id = model._sample_task1_training_times(32, 5)
        self.assertEqual(model.uniform_call, (32, 5, 0.0, 1.0))
        torch.testing.assert_close(physical_t, tau.square())
        self.assertEqual(group_id.unique().tolist(), [-1])

    def test_high_noise_sampler_rejects_non_contract_micro_batch(self):
        model = TrainingTimeHarness('high_noise_quota32')
        with self.assertRaisesRegex(ValueError, '32, 64, or 128'):
            model._sample_task1_training_times(96, 0)

    def test_high_noise_quota_scales_by_exact_blocks_of_32(self):
        model = TrainingTimeHarness('high_noise_quota32')
        for batch_size, multiplier in ((64, 2), (128, 4)):
            with self.subTest(batch_size=batch_size):
                _, _, group_id = model._sample_task1_training_times(
                    batch_size, 0)
                self.assertEqual(
                    torch.bincount(group_id, minlength=5).tolist(),
                    [multiplier * value for value in [1, 4, 4, 17, 6]])

    def test_v1_stage_quotas_change_at_frozen_boundaries(self):
        model = TrainingTimeHarness('v1_staged_quota32')
        expected = {
            15000: [2, 4, 6, 7, 7, 4, 2],
            17999: [2, 4, 6, 7, 7, 4, 2],
            18000: [1, 3, 5, 8, 8, 5, 2],
            23999: [1, 3, 5, 8, 8, 5, 2],
            24000: [1, 1, 4, 7, 10, 6, 3],
            29999: [1, 1, 4, 7, 10, 6, 3],
        }
        for step, counts in expected.items():
            with self.subTest(step=step):
                model.global_step = step
                tau, physical_t, group_id = (
                    model._sample_task1_training_times(32, 0))
                self.assertEqual(
                    torch.bincount(group_id, minlength=7).tolist(), counts)
                torch.testing.assert_close(physical_t, tau.square())
                self.assertTrue(torch.all(tau[group_id == 4] >= 0.1))
                self.assertTrue(torch.all(tau[group_id == 4] < 0.4))
                self.assertTrue(torch.all(tau[group_id == 5] >= 0.4))
                self.assertTrue(torch.all(tau[group_id == 5] < 0.7))
                self.assertTrue(torch.all(tau[group_id == 6] >= 0.7))
                self.assertTrue(torch.all(tau[group_id == 6] < 1.0))


if __name__ == '__main__':
    unittest.main()
