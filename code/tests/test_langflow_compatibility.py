import unittest

import torch

import algo
import algorithm_registry
import experiment_callbacks
import utils
from langflow_hybrid import diagnostics, model, ops, sampling


class _CheckpointBase:
    def on_save_checkpoint(self, checkpoint):
        checkpoint['_base_hook_called'] = True


class _CheckpointHarness(diagnostics.LangFlowDiagnosticsMixin,
                         _CheckpointBase):
    def __init__(self):
        self.global_step = 17
        self.codebook_gradient_mode = 'frozen'
        self.token_bias_schedule = 'warmup'
        self.classification_prototype_mode = 'shared'
        self.embedding_diagnostics = {'initial': {}}
        self.gamma_bin_diagnostics = {'17': {}}
        self.posterior_bucket_diagnostics = {'17': {}}
        self.gradient_route_diagnostics = {'17': {}}
        self.bias_logit_diagnostics = {'17': {}}
        self.prototype_diagnostics = {'17': {}}
        self._training_token_counts = torch.tensor([1, 2, 3])


class LangFlowCompatibilityTests(unittest.TestCase):
    def test_algo_reexports_legacy_helpers(self):
        expected = {
            'langflow_gumbel_gamma': ops.langflow_gumbel_gamma,
            'langflow_alpha_sigma': ops.langflow_alpha_sigma,
            'langflow_corrupt_embedding': ops.langflow_corrupt_embedding,
            'flm_corrupt_vocab': ops.flm_corrupt_vocab,
            'flm_linear_alpha_sigma': ops.flm_linear_alpha_sigma,
            'flm_linear_euler_update': ops.flm_linear_euler_update,
            'flm_linear_gamma': ops.flm_linear_gamma,
            'flm_model_time_condition': ops.flm_model_time_condition,
            'flm_vocab_gaussian_bias': ops.flm_vocab_gaussian_bias,
            'detached_self_conditioning_embedding': (
                ops.detached_self_conditioning_embedding),
            'candidate_bias_codebook': ops.candidate_bias_codebook,
            'probability_prediction_metrics': (
                ops.probability_prediction_metrics),
            'langflow_token_bias_weight': ops.langflow_token_bias_weight,
            'langflow_euler_edm_update': ops.langflow_euler_edm_update,
        }
        for name, implementation in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(algo, name), implementation)

    def test_public_class_remains_defined_in_algo_with_legacy_methods(self):
        self.assertEqual(algo.LangFlowFLMHybrid.__module__, 'algo')
        self.assertTrue(issubclass(
            algo.LangFlowFLMHybrid, model.LangFlowModelMixin))
        self.assertTrue(issubclass(
            algo.LangFlowFLMHybrid, diagnostics.LangFlowDiagnosticsMixin))
        self.assertTrue(issubclass(
            algo.LangFlowFLMHybrid, sampling.LangFlowSamplingMixin))
        legacy_methods = {
            '__init__', '_validate_configuration', 'optimization_scale',
            'embedding_weight', 'classification_prototype_weight',
            '_classification_prototype_raw', 'embed_tokens',
            'embed_probabilities', 'corrupt_embeddings',
            'corrupt_vocab_state', '_state_embedding', '_path_alpha_sigma',
            '_current_token_bias_weight', '_l2_norm',
            'configure_gradient_clipping', '_update_tensor_moments',
            '_record_validation_logit_stats', '_forward_logits', 'forward',
            '_process_model_input', 'validation_step',
            'on_validation_epoch_start', '_posterior_diagnostic_names',
            '_frequency_group_ids', '_accumulate_gamma_bins',
            '_finish_gamma_bin_diagnostics',
            '_finish_validation_logit_stats', 'on_validation_epoch_end',
            '_log_probability_metrics', 'loss',
            '_raw_gradient_from_normalized', '_geometry_separation_proxy',
            'on_after_backward', 'optimizer_step',
            'embedding_nearest_neighbor_diagnostic',
            'classification_prototype_diagnostic', 'generate_samples',
            'on_save_checkpoint', 'on_load_checkpoint', 'on_train_start',
            'on_train_end',
        }
        self.assertTrue(all(
            hasattr(algo.LangFlowFLMHybrid, name)
            for name in legacy_methods))

    def test_mixins_do_not_create_a_state_dict_namespace(self):
        for mixin in (
                model.LangFlowModelMixin,
                diagnostics.LangFlowDiagnosticsMixin,
                sampling.LangFlowSamplingMixin):
            self.assertFalse(issubclass(mixin, torch.nn.Module))
        self.assertFalse(any(
            name.startswith('langflow_hybrid')
            for name in algo.LangFlowFLMHybrid.__dict__))

    def test_checkpoint_field_names_are_unchanged(self):
        checkpoint = {'state_dict': {}}
        _CheckpointHarness().on_save_checkpoint(checkpoint)
        expected = {
            'hybrid_token_bias_step',
            'hybrid_codebook_gradient_mode',
            'hybrid_token_bias_schedule',
            'hybrid_classification_prototype_mode',
            'hybrid_embedding_diagnostics',
            'hybrid_gamma_bin_diagnostics',
            'hybrid_posterior_bucket_diagnostics',
            'hybrid_gradient_route_diagnostics',
            'hybrid_bias_logit_diagnostics',
            'hybrid_prototype_diagnostics',
            'hybrid_training_token_counts',
        }
        self.assertTrue(expected.issubset(checkpoint))
        self.assertTrue(checkpoint['_base_hook_called'])

    def test_registry_preserves_names_and_error(self):
        expected = {
            'ar': algo.AR,
            'mdlm': algo.MDLM,
            'duo_base': algo.DUO_BASE,
            'duo': algo.DUO,
            'flm': algo.FLM,
            'langflow_flm_hybrid': algo.LangFlowFLMHybrid,
            'fmlm_twomodel': algo.FMLM_TwoModel,
            'fmlm_twostage': algo.FMLM_TwoStage,
            'fmlm': algo.FMLM,
            'd3pm': algo.D3PMAbsorb,
            'sedd': algo.SEDDAbsorb,
            'distillation': algo.Distillation,
            'rectification': algo.Rectification,
        }
        for name, expected_class in expected.items():
            with self.subTest(name=name):
                self.assertIs(
                    algorithm_registry.get_algorithm_class(name),
                    expected_class)
        with self.assertRaisesRegex(
                ValueError, '^Invalid algorithm name: unknown$'):
            algorithm_registry.get_algorithm_class('unknown')

    def test_callback_legacy_aliases_remain_available(self):
        names = (
            'AdaptiveValidationIntervalCallback',
            'CUDAPeakMemoryCallback',
            'GradientInspectionCallback',
            'GradientNormCallback',
            'MilestoneCheckpointCallback',
            'OptimizerStepTimerCallback',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(utils, name),
                    getattr(experiment_callbacks, name))


if __name__ == '__main__':
    unittest.main()
