import tempfile
import unittest
from pathlib import Path

import omegaconf
import torch

import run_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class LangFlowConfigContractTests(unittest.TestCase):
    @staticmethod
    def _load(name):
        config = omegaconf.OmegaConf.load(
            SOURCE_ROOT / 'configs' / 'algo' / name)
        return omegaconf.OmegaConf.to_container(config, resolve=True)

    def test_paired_configs_only_change_loss_and_optimization_scale(self):
        ce = self._load('langflow_hybrid_ce.yaml')
        mse = self._load('langflow_hybrid_softmax_mse.yaml')

        self.assertEqual(ce.pop('loss_type'), 'cross_entropy')
        self.assertEqual(mse.pop('loss_type'), 'softmax_probability_mse')
        self.assertEqual(ce.pop('optimization_scale'), 1.0)
        self.assertEqual(mse.pop('optimization_scale'), 'vocab_size_over_2')
        self.assertEqual(ce, mse)

    def test_causal_variants_change_only_registered_variables(self):
        original_mse = self._load('langflow_hybrid_softmax_mse.yaml')
        dbias = self._load('langflow_hybrid_dbias_mse.yaml')
        frozen_mse = self._load('langflow_hybrid_frozen_mse.yaml')
        frozen_ce = self._load('langflow_hybrid_frozen_ce.yaml')

        self.assertEqual(dbias['codebook_gradient_mode'], 'detach_candidate')
        self.assertEqual(frozen_mse['codebook_gradient_mode'], 'frozen')
        self.assertEqual(frozen_ce['codebook_gradient_mode'], 'frozen')
        for variant, ignored in (
                (dbias, {'codebook_gradient_mode'}),
                (frozen_mse, {
                    'codebook_gradient_mode', 'token_bias_schedule',
                    'classification_prototype_mode',
                    'bias_interpretation'}),
                (frozen_ce, {
                    'codebook_gradient_mode', 'loss_type',
                    'optimization_scale'})):
            expected = {
                key: value for key, value in original_mse.items()
                if key not in ignored}
            actual = {
                key: value for key, value in variant.items()
                if key not in ignored}
            self.assertEqual(expected, actual)

    def test_independent_noise_matrix_changes_only_registered_variable(self):
        native = self._load('langflow_hybrid_softmax_mse.yaml')
        independent = self._load(
            'langflow_hybrid_independent_noise_matrix_mse.yaml')

        self.assertEqual(
            independent['classification_prototype_mode'], 'independent')
        self.assertEqual(independent['codebook_gradient_mode'], 'all')
        registered = {
            'classification_prototype_mode', 'bias_interpretation'}
        self.assertEqual(
            {key: value for key, value in native.items()
             if key not in registered},
            {key: value for key, value in independent.items()
             if key not in registered})

    def test_task1_vocab_mse_names_the_flm_state_and_loss_contract(self):
        config = self._load('langflow_hybrid_vocab_mse.yaml')
        self.assertEqual(config['state_space'], 'vocab')
        self.assertEqual(config['corruption'], 'flm_linear_gaussian')
        self.assertFalse(config['embedding_state'])
        self.assertFalse(config['normalize_embeddings'])
        self.assertEqual(
            config['time_sampling'],
            'flm_decoding_error_rate_tau_inverse')
        self.assertEqual(config['model_time_condition'], 'tau')
        self.assertEqual(
            config['classification_prototype_mode'], 'direct_vocab_state')
        self.assertEqual(
            config['bias_interpretation'],
            'matched_flm_vocab_gaussian_direct')
        self.assertEqual(config['loss_type'], 'softmax_probability_mse')
        self.assertEqual(config['prediction_target'], 'clean_token_one_hot')
        self.assertEqual(config['output_transform'], 'softmax')
        self.assertEqual(
            config['reduction'], 'vocab_sum_then_valid_token_mean')
        self.assertEqual(
            config['time_weighting'],
            'uniform_tau_unit_weight')
        self.assertEqual(config['optimization_scale'], 'one_half')

    def test_task1_a_and_c_change_only_model_time_condition(self):
        variant_a = self._load('langflow_hybrid_task1_a.yaml')
        variant_c = self._load('langflow_hybrid_task1_c.yaml')
        self.assertEqual(variant_a.pop('model_time_condition'), 'tau')
        self.assertEqual(variant_c.pop('model_time_condition'), 'log_nsr')
        self.assertEqual(variant_a, variant_c)

    def test_task1_ce_changes_only_loss_definition(self):
        mse = self._load('langflow_hybrid_task1_a.yaml')
        ce = self._load('task1_vocab_ce.yaml')

        self.assertEqual(mse.pop('loss_type'), 'softmax_probability_mse')
        self.assertEqual(ce.pop('loss_type'), 'cross_entropy')
        self.assertEqual(mse.pop('optimization_scale'), 'one_half')
        self.assertEqual(ce.pop('optimization_scale'), 'one')
        self.assertEqual(ce, mse)

    def test_four_bias_ablation_configs_change_only_registered_fields(self):
        warmup = self._load('langflow_hybrid_frozen_mse.yaml')
        variants = {
            'none': self._load('langflow_hybrid_frozen_mse_no_bias.yaml'),
            'full': self._load('langflow_hybrid_frozen_mse_full_bias.yaml'),
            'independent': self._load(
                'langflow_hybrid_frozen_mse_independent_prototype.yaml'),
        }
        registered = {
            'tokenwise_bias', 'token_bias_schedule',
            'classification_prototype_mode', 'bias_interpretation'}
        fixed = {key: value for key, value in warmup.items()
                 if key not in registered}
        for name, variant in variants.items():
            with self.subTest(name=name):
                self.assertEqual(
                    fixed, {key: value for key, value in variant.items()
                            if key not in registered})
                self.assertEqual(variant['codebook_gradient_mode'], 'frozen')
                self.assertEqual(
                    variant['loss_type'], 'softmax_probability_mse')


class RunManifestArchitectureContractTests(unittest.TestCase):
    def test_manifest_names_frozen_hybrid_components(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing')
            config = omegaconf.OmegaConf.create({
                'seed': 1,
                'data': {
                    'train': 'openwebtext-packed-v1',
                    'valid': 'openwebtext-packed-v1',
                    'tokenizer_name_or_path': 'gpt2',
                    'openwebtext_10k_dir': missing,
                    'packed_dir': missing,
                },
                'model': {
                    'length': 128,
                    'hidden_size': 768,
                    'cond_dim': 128,
                    'n_blocks': 12,
                    'n_heads': 12,
                    'dropout': 0.1,
                },
                'algo': {
                    'name': 'langflow_flm_hybrid',
                    'loss_type': 'cross_entropy',
                    'prediction_target': 'clean_token_posterior',
                    'output_transform': 'softmax',
                    'reduction': 'vocab_sum_then_valid_token_mean',
                    'time_weighting': 'fixed_gumbel_proposal_unit_weight',
                    'embedding_state': True,
                    'normalize_embeddings': True,
                    'time_conditioning': True,
                    'gumbel': {
                        'loc': 4.723,
                        'scale': 0.852,
                        'cutoff': 1.0e-5,
                    },
                    'self_conditioning': True,
                    'self_condition_probability': 0.25,
                    'tokenwise_bias': True,
                    'token_bias_warmup_steps': 5000,
                },
                'experiment': {},
                'training': {'ema': 0.9999},
            })
            manifest = run_manifest.create(config, torch.nn.Linear(2, 2))

        architecture = manifest['architecture']
        self.assertTrue(architecture['embedding_space_ode'])
        self.assertTrue(architecture['normalized_embeddings'])
        self.assertTrue(architecture['time_conditioning'])
        self.assertEqual(architecture['gumbel_schedule'], {
            'loc': 4.723,
            'scale': 0.852,
            'cutoff': 1.0e-5,
        })
        self.assertTrue(architecture['self_conditioning'])
        self.assertEqual(architecture['self_conditioning_probability'], 0.25)
        self.assertTrue(architecture['tokenwise_bias'])
        self.assertEqual(architecture['token_bias_warmup_steps'], 5000)


if __name__ == '__main__':
    unittest.main()
