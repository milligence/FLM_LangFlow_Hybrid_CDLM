import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import algo
import dataloader
import metrics
import trainer_base
from models.ema import ExponentialMovingAverage


class GroupTextsTests(unittest.TestCase):
    def test_short_input_produces_no_partial_sequence(self):
        grouped = dataloader._group_texts(
            {'input_ids': [[1, 2, 3]]}, block_size=8, bos=10, eos=11)
        self.assertEqual(grouped['input_ids'], [])
        self.assertEqual(grouped['attention_mask'], [])

    def test_exact_block_gets_bos_and_eos(self):
        grouped = dataloader._group_texts(
            {'input_ids': [[1, 2], [3, 4]]},
            block_size=6, bos=10, eos=11)
        self.assertEqual(grouped['input_ids'], [[10, 1, 2, 3, 4, 11]])
        self.assertEqual(tuple(grouped['attention_mask'][0].shape), (6,))

    def test_remainder_is_dropped_deterministically(self):
        grouped = dataloader._group_texts(
            {'input_ids': [list(range(10))]},
            block_size=6, bos=10, eos=11)
        self.assertEqual(len(grouped['input_ids']), 2)
        self.assertEqual(grouped['input_ids'][-1], [10, 4, 5, 6, 7, 11])


class DataLoaderTests(unittest.TestCase):
    def test_local_owt_loader_keeps_one_file_as_one_document(self):
        with tempfile.TemporaryDirectory() as data_dir:
            first_path = os.path.join(data_dir, 'a.txt')
            second_path = os.path.join(data_dir, 'b.txt')
            with open(first_path, 'w', encoding='utf-8') as handle:
                handle.write('first line\nsecond line\n\n\n\nlast paragraph')
            with open(second_path, 'w', encoding='utf-8') as handle:
                handle.write('second document')
            dataset = dataloader._load_local_text_documents(
                [first_path, second_path])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            dataset[0]['text'],
            'first line\nsecond line\n\nlast paragraph')
        self.assertEqual(dataset[1]['text'], 'second document')

    def test_batch_one_and_final_partial_batch(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(5))
        batch_one = torch.utils.data.DataLoader(dataset, batch_size=1)
        self.assertEqual(next(iter(batch_one))[0].shape, (1,))

        batches = list(torch.utils.data.DataLoader(dataset, batch_size=2))
        self.assertEqual([batch[0].shape[0] for batch in batches], [2, 2, 1])

    def test_existing_dataset_cache_never_calls_network_loader(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = os.path.join(
                cache_dir, 'cached_train_bs8_wrapped.dat')
            os.mkdir(cache_path)
            cached_dataset = mock.Mock()
            cached_dataset.with_format.return_value = 'cached-result'
            with mock.patch.object(
                    dataloader.datasets, 'load_from_disk',
                    return_value=cached_dataset) as load_from_disk, \
                 mock.patch.object(
                    dataloader.datasets, 'load_dataset',
                    side_effect=AssertionError('network loader called')):
                result = dataloader.get_dataset(
                    'cached', tokenizer=mock.Mock(), wrap=True,
                    mode='train', cache_dir=cache_dir, block_size=8,
                    num_proc=1, streaming=False)
            self.assertEqual(result, 'cached-result')
            load_from_disk.assert_called_once_with(cache_path)

    def test_fault_tolerant_sampler_resumes_current_permutation(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(20))
        original = dataloader.RandomFaultTolerantSampler(
            dataset, generator=torch.Generator().manual_seed(7))
        iterator = iter(original)
        consumed = [next(iterator) for _ in range(6)]
        self.assertEqual(len(consumed), 6)
        state = original.state_dict()
        remaining = list(iterator)

        restored = dataloader.RandomFaultTolerantSampler(
            dataset, generator=torch.Generator().manual_seed(99))
        restored.load_state_dict(state)
        self.assertEqual(list(restored), remaining)


class MetricsTests(unittest.TestCase):
    def test_generation_ppl_tokenizer_is_lazy(self):
        with mock.patch.object(
                metrics.transformers.AutoTokenizer, 'from_pretrained',
                side_effect=AssertionError('unexpected network access')):
            metric_set = metrics.Metrics(
                gen_ppl_eval_model_name_or_path='gpt2-large',
                eval_ppl_batch_size=1)
        self.assertIsNone(metric_set.tokenizer)

    def test_mse_objective_is_not_reported_as_perplexity(self):
        metric_set = metrics.Metrics(objective_is_nll=False)
        self.assertEqual(
            set(metric_set.valid_nlls.keys()), {'val/objective'})


class FLMLossTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.tensor([[[1.0, 0.0, 0.0]]])
        self.probabilities = torch.tensor([[[0.5, 0.25, 0.25]]])
        self.log_probabilities = self.probabilities.log()

    def test_cross_entropy_uses_simplex_log_probability(self):
        loss = algo.flm_clean_data_loss(
            self.log_probabilities, self.target,
            loss_type='cross_entropy', denoiser_output='simplex')
        torch.testing.assert_close(loss, torch.tensor([[0.6931472]]))

    def test_simplex_mse_is_squared_norm_over_vocabulary(self):
        loss = algo.flm_clean_data_loss(
            self.log_probabilities, self.target,
            loss_type='mse', denoiser_output='simplex')
        torch.testing.assert_close(loss, torch.tensor([[0.375]]))

    def test_euclidean_mse_does_not_apply_softmax(self):
        prediction = torch.tensor([[[0.5, -0.5, 0.0]]])
        loss = algo.flm_clean_data_loss(
            prediction, self.target,
            loss_type='mse', denoiser_output='euclidean')
        torch.testing.assert_close(loss, torch.tensor([[0.5]]))

    def test_cross_entropy_rejects_euclidean_output(self):
        with self.assertRaisesRegex(ValueError, 'requires.*simplex'):
            algo.flm_clean_data_loss(
                self.log_probabilities, self.target,
                loss_type='cross_entropy', denoiser_output='euclidean')

    def test_loss_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            algo.flm_clean_data_loss(
                self.log_probabilities[..., :2], self.target,
                loss_type='mse', denoiser_output='simplex')


class GradientNormTests(unittest.TestCase):
    def test_global_l2_norm_ignores_parameters_without_gradient(self):
        first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        second = torch.nn.Parameter(torch.tensor([3.0]))
        first.grad = torch.tensor([3.0, 4.0])
        value = metrics.torch.tensor(0.0)
        del value
        norm = __import__('utils').GradientNormCallback._global_l2_norm(
            [first, second], torch.device('cpu'))
        torch.testing.assert_close(norm, torch.tensor(5.0))


class ExponentialMovingAverageTests(unittest.TestCase):
    def test_device_move_normalizes_inference_shadow_tensors(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        ema = ExponentialMovingAverage([parameter], decay=0.9)
        with torch.inference_mode():
            ema.shadow_params = [ema.shadow_params[0].clone()]
        self.assertTrue(torch.is_inference(ema.shadow_params[0]))

        with torch.inference_mode():
            ema.move_shadow_params_to_device(torch.device('cpu'))
        self.assertFalse(torch.is_inference(ema.shadow_params[0]))

        parameter.data.add_(1.0)
        ema.update([parameter])
        self.assertTrue(torch.isfinite(ema.shadow_params[0]).all())

    def test_checkpoint_restore_compares_only_trainable_parameters(self):
        class Harness(trainer_base.TrainerBase):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.frozen = torch.nn.Parameter(
                    torch.zeros(13, 8), requires_grad=False)
                self.trainable = torch.nn.Parameter(torch.zeros(8, 16))
                self.config = SimpleNamespace(
                    training=SimpleNamespace(ema=0.9999))
                self.ema = ExponentialMovingAverage(
                    self._get_parameters(), decay=0.9999)
                self._pending_ema_state = None

            def _get_parameters(self):
                return iter((self.frozen, self.trainable))

        model = Harness()
        saved_shadow = torch.full_like(model.trainable, 7.0)
        model._pending_ema_state = {
            'decay': 0.9999,
            'num_updates': 10,
            'shadow_params': [saved_shadow.clone()],
        }
        with mock.patch.object(
                trainer_base.models.ema, 'ExponentialMovingAverage',
                wraps=ExponentialMovingAverage) as constructor:
            model.load_state_dict(model.state_dict())
        constructor.assert_not_called()
        self.assertEqual(model.ema.num_updates, 10)
        torch.testing.assert_close(model.ema.shadow_params[0], saved_shadow)


if __name__ == '__main__':
    unittest.main()
