import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from decision_policy import decide, profile_plan


def good36():
    return {
        'schema_version': 'f30-final-v2', 'completed_step': 36000,
        'current_profile': 'uniform', 'lineage_verified': True,
        'protocol_compatible': True, 'all_required_metrics_ready': True,
        'generation_health': True, 'risk_checks': [],
        'metrics': {
            'online_B_delta_nll_32_36': -.01,
            'eval_B_delta_nll_32_36': 0.,
            'late_signal_share_34': .8, 'late_signal_share_36': .82,
            'terminal_ref_state_mse_ratio_36_32': .97,
            'terminal_mse_over_matching_floor_36': 10000.,
            'eval_B_minus_canonical_nll_36': .25,
            'eval_B_minus_canonical_nll_ci_low_36': .15,
            'canonical_ppl_ratio_36_32': .98,
            'local_aggregate_mse_ratio_36_32': .999,
            'sensitive_local_mse_max_ratio_36_32': 1.,
            'sensitive_local_ce_max_ratio_36_32': 1.,
            'weighted_map_ratio_36': .30,
        },
    }


class FinalV2DecisionTests(unittest.TestCase):
    def test_profiles_preserve_budget_and_original_d(self):
        for name in ('pre32', 'uniform', 'late_trial', 'late_fixed', 'off'):
            plan = profile_plan(name)
            self.assertEqual(sum(plan['counts'].values()), 96)
            self.assertEqual(plan['counts']['D_original'], 24)
            self.assertEqual(sum(plan['D_patch_interval_rows']),
                             plan['counts']['D_patch'])

    def test_boundary_semantics(self):
        report = good36()
        report.update(completed_step=32000, current_profile='pre32')
        result = decide(report)
        self.assertEqual(result['new_profile'], 'uniform')
        self.assertEqual(result['action_first_update'], 32001)
        self.assertFalse(result['reset_model_optimizer_ema_rng'])

    def test_late_trial_requires_every_gate(self):
        self.assertEqual(decide(good36())['new_profile'], 'late_trial')
        report = good36()
        report['metrics'].pop('canonical_ppl_ratio_36_32')
        self.assertEqual(decide(report)['new_profile'], 'uniform')
        report = good36()
        report['metrics']['online_B_delta_nll_32_36'] = math.log(.93)
        self.assertEqual(decide(report)['new_profile'], 'uniform')

    def test_risk_guard_precedes_gate(self):
        report = good36()
        report['risk_checks'] = [{
            'completed_step': step,
            'eval_A_delta_nll': .2, 'eval_A_delta_nll_ci_low': .1,
            'eval_B_delta_nll': .2, 'eval_B_delta_nll_ci_low': .1,
            'online_B_delta_nll': .1, 'canonical_ppl_ratio': 1.0,
        } for step in (34000, 36000)]
        self.assertEqual(decide(report)['new_profile'], 'off')

    def test_missing_40k_evidence_ends_trial(self):
        report = {
            'schema_version': 'f30-final-v2', 'completed_step': 40000,
            'current_profile': 'late_trial', 'lineage_verified': True,
            'protocol_compatible': True, 'all_required_metrics_ready': False,
            'generation_health': False, 'risk_checks': [], 'metrics': {},
        }
        result = decide(report)
        self.assertEqual(result['new_profile'], 'uniform')
        self.assertEqual(result['action_first_update'], 40001)


if __name__ == '__main__':
    unittest.main()
