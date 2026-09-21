from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read_env(name):
    values = {}
    for line in (ROOT / 'configs' / name).read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            values[key] = value
    return values


class SmallStepReleaseTests(unittest.TestCase):
    def test_only_selected_tvm_classes_are_published(self):
        source = (ROOT / 'task1_tvm_ce.py').read_text()
        self.assertIn('class Task1TVMCE(', source)
        self.assertIn('class Task1TVMSCRepair(', source)
        self.assertIn('class Task1TVMJointJ0(', source)
        self.assertNotIn('class Task1TVMEndpoint500(', source)
        self.assertNotIn('class Task1TVMJointJ1(', source)

    def test_sc_code_has_no_composition_interface_or_loss(self):
        paths = (
            ROOT / 'task1_tvm_ce.py',
            ROOT / 'configs/algo/task1_tvm_sc_repair.yaml',
            ROOT / 'configs/algo/task1_tvm_joint_j0.yaml',
            ROOT / 'scripts/train_owt_128_langflow_hybrid.sh',
        )
        combined = '\n'.join(path.read_text() for path in paths)
        for forbidden in (
                'tvm_sc_comp_', 'composition_weight',
                'composition_loss', "losses['composition']"):
            self.assertNotIn(forbidden, combined)

    def test_fixed_teacher_10k_contract(self):
        values = read_env('task1_tvm_ce_10k.env')
        expected = {
            'GLOBAL_BATCH_SIZE': '128',
            'MICRO_BATCH_SIZE': '32',
            'ACCUMULATE_GRAD_BATCHES': '4',
            'MAX_STEPS': '10000',
            'LEARNING_RATE': '1e-4',
            'WARMUP_STEPS': '2500',
            'EMA_DECAY': '0.9999',
        }
        for key, value in expected.items():
            self.assertEqual(values[key], value)

    def test_posterior_tvm_exact_options_remain_opt_in(self):
        model_config = (
            ROOT / 'configs/model/small_128_posterior_tvm.yaml').read_text()
        self.assertIn('attention_jvp_backend: reference', model_config)
        model_source = (ROOT / 'models/dit.py').read_text()
        self.assertIn("{'reference', 'bmm'}", model_source)
        posterior = (ROOT / 'task1_posterior_tvm.py').read_text()
        self.assertIn("{'reference', 'detached'}", posterior)
        self.assertIn("{'no_grad', 'inference_clone'}", posterior)

    def test_registry_has_no_excluded_algorithms(self):
        registry = (ROOT / 'algorithm_registry.py').read_text()
        for selected in (
                'task1_tvm_ce', 'task1_tvm_sc_repair',
                'task1_tvm_joint_j0', 'task1_posterior_tvm'):
            self.assertIn(selected, registry)
        for excluded in (
                'task1_tvm_endpoint500', 'task1_tvm_joint_j1',
                'task1_no_sc_mse_teacher'):
            self.assertNotIn(excluded, registry)


if __name__ == '__main__':
    unittest.main()
