import math
import unittest

import torch
from omegaconf import OmegaConf

from models.dit import DIT, EmbeddingLayer
from models.ema import ExponentialMovingAverage
from langflow_test_support import (
    DirectVocabForwardHarness,
    NoBiasForwardHarness,
    TokenBiasHarness,
)


class LangFlowModelTests(unittest.TestCase):
    def test_task1_input_projection_and_classifier_are_independent(self):
        config = OmegaConf.create({
            'is_di4c': False,
            'algo': {
                'name': 'langflow_flm_hybrid',
                'causal_attention': False,
                'embedding_state': False,
                'self_conditioning': True,
                'normalize_embeddings': False,
                'codebook_gradient_mode': 'all',
                'classification_prototype_mode': 'direct_vocab_state',
                'double_temb': False,
                'learnable_loss_weighting': False,
            },
            'model': {
                'hidden_size': 8, 'cond_dim': 8, 'n_blocks': 1,
                'n_heads': 1, 'dropout': 0.0, 'scale_by_sigma': True,
                'tie_word_embeddings': False,
            },
        })
        backbone = DIT(config, vocab_size=13)
        self.assertEqual(backbone.vocab_embed.embedding.shape, (13, 8))
        self.assertEqual(backbone.output_layer.linear.weight.shape, (13, 8))
        self.assertIsNot(
            backbone.vocab_embed.embedding,
            backbone.output_layer.linear.weight)

    def test_frozen_codebook_is_excluded_before_ema_construction(self):
        config = OmegaConf.create({
            'is_di4c': False,
            'algo': {
                'name': 'langflow_flm_hybrid',
                'causal_attention': False,
                'embedding_state': True,
                'self_conditioning': True,
                'normalize_embeddings': True,
                'codebook_gradient_mode': 'frozen',
                'double_temb': False,
                'learnable_loss_weighting': False,
            },
            'model': {
                'hidden_size': 8,
                'cond_dim': 8,
                'n_blocks': 1,
                'n_heads': 1,
                'dropout': 0.0,
                'scale_by_sigma': True,
            },
        })
        backbone = DIT(config, vocab_size=13)
        self.assertFalse(backbone.vocab_embed.embedding.requires_grad)
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(backbone.parameters(), decay=0.9999)
        self.assertEqual(
            [tuple(p.shape) for p in trainable],
            [tuple(p.shape) for p in ema.shadow_params])
        ema.store(backbone.parameters())
        ema.copy_to(backbone.parameters())
        ema.restore(backbone.parameters())

    def test_normalized_embedding_rows_have_sqrt_hidden_norm(self):
        layer = EmbeddingLayer(dim=6, vocab_dim=11, normalize=True)
        norms = layer.normalized_weight().float().norm(dim=-1)
        torch.testing.assert_close(
            norms, torch.full_like(norms, math.sqrt(6)))

    def test_token_bias_schedule_modes(self):
        none = TokenBiasHarness('train', 2500)
        none.token_bias_schedule = 'none'
        full = TokenBiasHarness('train', 0)
        full.token_bias_schedule = 'full'
        self.assertEqual(none._current_token_bias_weight(), 0.0)
        self.assertEqual(full._current_token_bias_weight(), 1.0)

    def test_no_bias_forward_returns_residual_without_prototype_access(self):
        model = NoBiasForwardHarness()
        z = torch.randn(2, 3, 5)
        gamma = torch.randn(2)
        expected = model.backbone(z, gamma)
        actual = model._forward_logits(z, gamma)
        torch.testing.assert_close(actual, expected)

    def test_task1_forward_adds_direct_vocab_bias(self):
        model = DirectVocabForwardHarness()
        state = torch.randn(2, 4, 5)
        physical_t = torch.tensor([0.2, 0.7])
        actual = model._forward_logits(
            state, torch.tensor([0.1, 0.9]), bias_weight=0.4,
            physical_t=physical_t)
        expected = algo.flm_vocab_gaussian_bias(
            state, physical_t, weight=0.4)
        torch.testing.assert_close(actual, expected)

    def test_task1_vocab_summed_brier_uses_one_half_scale(self):
        harness = type('Harness', (), {
            'loss_type': 'softmax_probability_mse',
            'state_space': 'vocab',
            'vocab_size': 50257,
        })()
        self.assertEqual(
            algo.LangFlowFLMHybrid.optimization_scale.fget(harness), 0.5)

    def test_independent_prototype_copies_codebook_without_rng_draw(self):
        config = OmegaConf.create({
            'is_di4c': False,
            'algo': {
                'name': 'langflow_flm_hybrid',
                'causal_attention': False,
                'embedding_state': True,
                'self_conditioning': True,
                'normalize_embeddings': True,
                'codebook_gradient_mode': 'all',
                'classification_prototype_mode': 'independent',
                'double_temb': False,
                'learnable_loss_weighting': False,
            },
            'model': {
                'hidden_size': 8, 'cond_dim': 8, 'n_blocks': 1,
                'n_heads': 1, 'dropout': 0.0, 'scale_by_sigma': True,
            },
        })
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        backbone = DIT(config, vocab_size=13)
        after_model = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        shared_config = OmegaConf.create(
            OmegaConf.to_container(config, resolve=True))
        shared_config.algo.classification_prototype_mode = 'shared'
        shared = DIT(shared_config, vocab_size=13)
        after_shared = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(after_model, after_shared))
        torch.testing.assert_close(
            backbone.classification_prototype,
            backbone.vocab_embed.embedding)
        self.assertTrue(backbone.classification_prototype.requires_grad)
        self.assertTrue(backbone.vocab_embed.embedding.requires_grad)
        self.assertIsNot(
            backbone.classification_prototype,
            backbone.vocab_embed.embedding)

    def test_sample_eval_uses_loaded_checkpoint_step_for_token_bias(self):
        sample_eval = TokenBiasHarness('sample_eval', 0, loaded_step=1000)
        posterior_eval = TokenBiasHarness('ppl_eval', 0, loaded_step=5000)
        training = TokenBiasHarness('train', 1250, loaded_step=1000)
        self.assertEqual(sample_eval._current_token_bias_weight(), 0.2)
        self.assertEqual(posterior_eval._current_token_bias_weight(), 1.0)
        self.assertEqual(training._current_token_bias_weight(), 0.25)


if __name__ == '__main__':
    unittest.main()
