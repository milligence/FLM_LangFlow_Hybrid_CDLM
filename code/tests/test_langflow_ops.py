import math
import unittest

import torch
import torch.nn.functional as F

import algo
from models.dit import EmbeddingLayer


class LangFlowOpsTests(unittest.TestCase):
    def test_gumbel_inverse_cdf_and_cutoff(self):
        q = torch.tensor([0.0, 0.25, 0.75, 1.0])
        gamma = algo.langflow_gumbel_gamma(q)
        expected = 4.723 - 0.852 * torch.log(
            -torch.log(q.clamp(1e-5, 1.0 - 1e-5)))
        torch.testing.assert_close(gamma, expected)
        self.assertTrue(torch.isfinite(gamma).all())

    def test_alpha_sigma_form_a_vp_path(self):
        gamma = torch.linspace(-10, 10, 41)
        alpha, sigma = algo.langflow_alpha_sigma(gamma)
        torch.testing.assert_close(
            alpha.square() + sigma.square(), torch.ones_like(gamma))

    def test_embedding_corruption_shape_and_end_membership(self):
        clean = torch.randn(2, 7, 5)
        noise = torch.randn_like(clean)
        gamma = torch.tensor([-4.0, 4.0])
        corrupted = algo.langflow_corrupt_embedding(clean, gamma, noise)
        self.assertEqual(corrupted.shape, clean.shape)
        alpha, sigma = algo.langflow_alpha_sigma(gamma)
        expected = (alpha[:, None, None] * clean
                    + sigma[:, None, None] * noise)
        torch.testing.assert_close(corrupted, expected)

    def test_flm_vocab_corruption_is_linear_gaussian_one_hot_path(self):
        tokens = torch.tensor([[0, 2], [1, 0]])
        t = torch.tensor([0.0, 1.0])
        noise = torch.tensor([
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ])
        state, target, returned_noise = algo.flm_corrupt_vocab(
            tokens, t, vocab_size=3, noise=noise)
        torch.testing.assert_close(state[0], noise[0])
        torch.testing.assert_close(state[1], target[1])
        torch.testing.assert_close(returned_noise, noise)
        torch.testing.assert_close(
            target, F.one_hot(tokens, num_classes=3).float())

    def test_flm_vocab_corruption_can_skip_dense_target(self):
        tokens = torch.tensor([[0, 2], [1, 0]])
        t = torch.tensor([0.25, 0.75])
        noise = torch.randn(2, 2, 3)
        reference_state, reference_target, reference_noise = (
            algo.flm_corrupt_vocab(tokens, t, 3, noise=noise))
        state, target, returned_noise = algo.flm_corrupt_vocab(
            tokens, t, 3, noise=noise, return_target=False)
        self.assertIsNone(target)
        torch.testing.assert_close(state, reference_state)
        torch.testing.assert_close(returned_noise, reference_noise)
        torch.testing.assert_close(
            reference_target, F.one_hot(tokens, num_classes=3).float())

    def test_flm_vocab_corruption_can_reuse_internal_noise_buffer(self):
        tokens = torch.tensor([[0, 2], [1, 0]])
        t = torch.tensor([0.25, 0.75])
        torch.manual_seed(19)
        reference_state, _, reference_noise = algo.flm_corrupt_vocab(
            tokens, t, 3, return_target=False)
        torch.manual_seed(19)
        state, target, returned_noise = algo.flm_corrupt_vocab(
            tokens, t, 3, return_target=False, return_noise=False)
        self.assertIsNone(target)
        self.assertIsNone(returned_noise)
        self.assertIsNotNone(reference_noise)
        torch.testing.assert_close(state, reference_state)
        _, _, omitted_supplied_noise = algo.flm_corrupt_vocab(
            tokens, t, 3, noise=reference_noise, return_target=False,
            return_noise=False)
        self.assertIsNone(omitted_supplied_noise)

    def test_flm_linear_gamma_round_trip_and_euler_step(self):
        t = torch.tensor([0.1, 0.5, 0.9])
        gamma = algo.flm_linear_gamma(t)
        alpha, sigma = algo.flm_linear_alpha_sigma(gamma)
        torch.testing.assert_close(alpha, t)
        torch.testing.assert_close(sigma, 1.0 - t)
        state = torch.zeros(1, 2, 3)
        clean = torch.ones_like(state)
        updated = algo.flm_linear_euler_update(
            state, clean, torch.tensor(0.0), torch.tensor(0.5))
        torch.testing.assert_close(updated, torch.full_like(state, 0.5))

    def test_task1_time_conditioning_changes_only_the_coordinate(self):
        u = torch.tensor([0.2, 0.7])
        t = torch.tensor([0.1, 0.8])
        torch.testing.assert_close(
            algo.flm_model_time_condition(u, t, 'tau'), u)
        torch.testing.assert_close(
            algo.flm_model_time_condition(u, t, 'log_nsr'),
            2.0 * (torch.log1p(-t) - torch.log(t)))

    def test_task1_gaussian_bias_is_direct_vocab_likelihood(self):
        state = torch.tensor([
            [[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]],
            [[4.0, 2.0, 1.0], [-1.0, 0.0, 2.0]],
        ])
        t = torch.tensor([0.25, 0.75])
        bias = algo.flm_vocab_gaussian_bias(state, t, weight=0.4)
        coefficient = 0.4 * t / (1.0 - t).square()
        torch.testing.assert_close(
            bias, coefficient[:, None, None] * state)

    def test_probability_mse_raw_and_scaled_semantics(self):
        logits = torch.tensor([[[0.2, -0.1, 0.4]]])
        targets = torch.tensor([[2]])
        metrics = algo.probability_prediction_metrics(logits, targets)
        probabilities = logits.softmax(dim=-1)
        one_hot = F.one_hot(targets, num_classes=3).float()
        expected_raw = (probabilities - one_hot).square().sum(dim=-1)
        torch.testing.assert_close(metrics['raw_brier'], expected_raw)
        self.assertEqual(metrics['top1_accuracy'].shape, targets.shape)
        self.assertTrue(torch.isfinite(metrics['posterior_entropy']).all())
        self.assertTrue(torch.all(metrics['max_probability'] >= 0))
        self.assertTrue(torch.all(metrics['max_probability'] <= 1))
        torch.testing.assert_close(
            metrics['raw_brier'] * (3 / 2), expected_raw * (3 / 2))
        torch.testing.assert_close(
            metrics['raw_brier'] * 0.5,
            0.5 * (probabilities - one_hot).square().sum(dim=-1))
        s2 = probabilities.square().sum(dim=-1)
        p_y = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        expected_g_y = 3 * p_y * (2 * p_y - 1 - s2)
        torch.testing.assert_close(metrics['probability_square_sum'], s2)
        torch.testing.assert_close(
            metrics['abs_mse_target_logit_gradient'], expected_g_y.abs())
        torch.testing.assert_close(
            metrics['target_gradient_ratio'],
            expected_g_y.abs() / (1 - p_y + 1e-12))

    def test_dbias_preserves_forward_and_only_removes_candidate_gradient(self):
        torch.manual_seed(7)
        vocab_size, hidden = 11, 5
        targets = torch.tensor([[1, 4, 8], [2, 5, 9]])
        gamma = torch.tensor([-0.7, 1.2])
        noise = torch.randn(2, 3, hidden)
        raw = torch.nn.Parameter(torch.randn(vocab_size, hidden))
        base_logits = torch.nn.Parameter(torch.randn(2, 3, vocab_size))

        def run(mode):
            weight = F.normalize(raw.float(), dim=-1) * math.sqrt(hidden)
            clean = weight[targets]
            z = algo.langflow_corrupt_embedding(clean, gamma, noise)
            alpha, sigma = algo.langflow_alpha_sigma(gamma)
            candidate = algo.candidate_bias_codebook(weight, mode)
            bias = torch.matmul(z, candidate.transpose(0, 1))
            logits = base_logits + (
                alpha / sigma.square())[:, None, None] * bias
            loss = (algo.probability_prediction_metrics(
                logits, targets)['raw_brier'].mean() * (vocab_size / 2))
            grads = torch.autograd.grad(
                loss, (base_logits, raw), allow_unused=True)
            return logits.detach(), loss.detach(), grads

        original_logits, original_loss, original_grads = run('all')
        dbias_logits, dbias_loss, dbias_grads = run('detach_candidate')
        torch.testing.assert_close(original_logits, dbias_logits)
        torch.testing.assert_close(original_loss, dbias_loss)
        torch.testing.assert_close(original_grads[0], dbias_grads[0])
        self.assertGreater(dbias_grads[1].norm().item(), 0.0)
        self.assertFalse(torch.allclose(original_grads[1], dbias_grads[1]))

        candidate_raw = raw.detach().clone().requires_grad_(True)
        candidate_weight = F.normalize(
            candidate_raw.float(), dim=-1) * math.sqrt(hidden)
        detached_z = torch.randn(2, 3, hidden)
        original_candidate = torch.matmul(
            detached_z,
            algo.candidate_bias_codebook(
                candidate_weight, 'all').transpose(0, 1)).sum()
        candidate_grad = torch.autograd.grad(
            original_candidate, candidate_raw)[0]
        self.assertGreater(candidate_grad.norm().item(), 0.0)
        detached_candidate = algo.candidate_bias_codebook(
            candidate_weight, 'detach_candidate')
        self.assertFalse(detached_candidate.requires_grad)

    def test_vocab_half_scale_matches_ce_gradient_at_uniform_logits(self):
        vocab_size = 13
        targets = torch.tensor([[4]])
        mse_logits = torch.zeros(1, 1, vocab_size, requires_grad=True)
        ce_logits = mse_logits.detach().clone().requires_grad_(True)
        mse = algo.probability_prediction_metrics(
            mse_logits, targets)['raw_brier'].mean() * (vocab_size / 2)
        ce = F.cross_entropy(ce_logits.flatten(0, 1), targets.flatten())
        mse.backward()
        ce.backward()
        torch.testing.assert_close(mse_logits.grad, ce_logits.grad)

    def test_self_conditioning_expected_embedding_is_detached(self):
        layer = EmbeddingLayer(dim=4, vocab_dim=7, normalize=True)
        probabilities = torch.randn(2, 3, 7, requires_grad=True).softmax(-1)
        conditioned = algo.detached_self_conditioning_embedding(
            probabilities, layer)
        self.assertEqual(conditioned.shape, (2, 3, 4))
        self.assertFalse(conditioned.requires_grad)

    def test_token_bias_warmup_endpoints(self):
        cases = [(0, 0.0), (2500, 0.5), (5000, 1.0), (9000, 1.0)]
        for step, expected in cases:
            with self.subTest(step=step):
                self.assertEqual(
                    algo.langflow_token_bias_weight(step, 5000), expected)

    def test_euler_edm_identity_when_gamma_does_not_change(self):
        z = torch.randn(2, 4, 3)
        clean = torch.randn_like(z)
        gamma = torch.tensor(0.7)
        updated = algo.langflow_euler_edm_update(z, clean, gamma, gamma)
        torch.testing.assert_close(updated, z)


if __name__ == '__main__':
    unittest.main()
