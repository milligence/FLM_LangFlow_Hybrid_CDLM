import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BiasAblationHelperTests(unittest.TestCase):
    def test_p90_selection_is_conservative(self):
        module = load_script('select_bias_ablation_steps.py')
        self.assertEqual(module.percentile90(list(range(1, 11))), 9)

    def test_registered_posterior_dominance(self):
        module = load_script('summarize_bias_ablation.py')
        control = {'raw_brier': 0.9, 'pY': 0.1, 'top1': 0.2}
        better = {'raw_brier': 0.8, 'pY': 0.2, 'top1': 0.2}
        mixed = {'raw_brier': 0.8, 'pY': 0.05, 'top1': 0.3}
        self.assertTrue(module.dominates(better, control))
        self.assertFalse(module.dominates(mixed, control))

    def test_collapse_gate_requires_all_three_metrics(self):
        module = load_script('summarize_bias_ablation.py')
        healthy = {
            'entropy': 3.5, 'repeated_4gram': 0.05, 'max_token': 0.10}
        collapsed = dict(healthy, repeated_4gram=0.051)
        self.assertTrue(module.noncollapsed(healthy))
        self.assertFalse(module.noncollapsed(collapsed))


if __name__ == '__main__':
    unittest.main()
