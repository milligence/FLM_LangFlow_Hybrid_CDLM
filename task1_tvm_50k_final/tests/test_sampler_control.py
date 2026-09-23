import unittest,json,copy
from pathlib import Path
import numpy as np
import yaml
from reference.samplers import sample_map_pairs,sample_local_times,draw_local_marginal_conditioned
from reference.control import learning_rates,rho_map,calibrated_scale,safety_recalibrate,ema_decay
ROOT=Path(__file__).resolve().parents[1]
# Synthetic monotone LUT for CONTRACT tests only. NOT the project's FLM LUT.
def tau(t):return t**8
def inv(u):return u**(1/8)

class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.f=yaml.safe_load((ROOT/'sampler_f.yaml').read_text())
        self.p=yaml.safe_load((ROOT/'sampler_p.yaml').read_text())
    def test_all_stages_and_boundaries(self):
        for cfg in [self.f,self.p]:
            for stage in cfg['map']['stages']:
                for k in [stage['start_step'],stage['end_step']-1]:
                    rows=sample_map_pairs(cfg,k,np.random.default_rng(k+17),tau,inv)
                    self.assertEqual(len(rows),stage['map_batch'])
                    for row in rows:
                        self.assertTrue(0<=row['r']<row['s']<=stage['terminal_cap']+1e-12)
                        self.assertAlmostEqual(row['eta'],(row['s']-row['r'])/(1-row['r']))
                    for name,n in stage['counts'].items():
                        self.assertEqual(sum(x['class']==name for x in rows),n)
    def test_final_exact_quotas_before_local_zero_atom(self):
        for cfg in [self.f,self.p]:
            for k in [5000,5001,8000,9999]:
                rows=sample_map_pairs(cfg,k,np.random.default_rng(k),tau,inv)
                self.assertEqual(sum(x['r']==0 for x in rows),18)
                self.assertEqual(sum(x['class']=='H' for x in rows),6)
                self.assertEqual(sum(x['class']=='D' and x['jittered'] for x in rows),12)
                self.assertEqual(sum(x['hard'] for x in rows),36)
    def test_no_chi_filter(self):
        found=False
        for k in [500,501,1000,1500]:
            rows=sample_map_pairs(self.f,k,np.random.default_rng(k),tau,inv)
            found |= any(x['chi']>.25 for x in rows)
        self.assertTrue(found)
    def test_f_patch_after_32k_is_exactly_eight_separate_analytical_D_rows(self):
        rows=sample_map_pairs(self.f,32000,np.random.default_rng(32000),tau,inv)
        counts={name:sum(x['class']==name for x in rows)
                for name in ('S','M','L','D','D_patch','Z','H')}
        self.assertEqual(counts,{'S':12,'M':24,'L':16,'D':24,'D_patch':8,'Z':6,'H':6})
        original=[row for row in rows if row['class']=='D']
        added=[row for row in rows if row['class']=='D_patch']
        self.assertEqual(sum(row['grid']=='g0' for row in original),12)
        self.assertEqual(sum(row['grid']=='gu' for row in original),12)
        self.assertEqual({row['grid'] for row in added},{'gstar'})
        self.assertEqual({row['source'] for row in added},{'analytical_data'})
        for interval in range(4):
            selected=[row for row in added if row['interval']==interval]
            self.assertEqual(len(selected),2)
            self.assertEqual(sum(row['jittered'] for row in selected),1)
    def test_f_dynamic_profiles_preserve_96_rows_and_declared_intervals(self):
        expected={
            'pre32':([0,0,0,0],24,0),
            'uniform':([2,2,2,2],16,8),
            'late_trial':([0,0,2,6],16,8),
            'late_fixed':([0,0,2,6],16,8),
            'off':([0,0,0,0],24,0),
        }
        for profile,(intervals,l_count,patch_count) in expected.items():
            rows=sample_map_pairs(
                self.f,36000,np.random.default_rng(36000),tau,inv,
                profile=profile)
            self.assertEqual(len(rows),96)
            self.assertEqual(sum(row['class']=='L' for row in rows),l_count)
            patch=[row for row in rows if row['class']=='D_patch']
            self.assertEqual(len(patch),patch_count)
            self.assertEqual(
                [sum(row['interval']==index for row in patch)
                 for index in range(4)],intervals)
            for index,count in enumerate(intervals):
                selected=[row for row in patch if row['interval']==index]
                self.assertEqual(sum(row['jittered'] for row in selected),count//2)
    def test_local_quota_and_rng_resume(self):
        for k,zeros in [(15000,16),(18000,8),(24000,8),(30000,6)]:
            x=sample_local_times(self.f,k,np.random.default_rng(k),tau,inv)
            self.assertEqual(len(x),256);self.assertEqual(sum(x==0),zeros)
        rng=np.random.default_rng(123)
        state=json.loads(json.dumps(rng.bit_generator.state))
        a=sample_map_pairs(self.f,18000,rng,tau,inv)
        rng2=np.random.default_rng();rng2.bit_generator.state=state
        b=sample_map_pairs(self.f,18000,rng2,tau,inv)
        self.assertEqual(a,b)
    def test_conditional_tau_law(self):
        rng=np.random.default_rng(8);upper=.8
        x=np.array([draw_local_marginal_conditioned(self.f['local'],0,upper,rng,tau,inv) for _ in range(3000)])
        u=x**8/upper**8
        self.assertLess(abs(u.mean()-.5),.02)
        self.assertLess(abs((u<.5).mean()-.5),.03)
    def test_lr_and_calibration(self):
        train=yaml.safe_load((ROOT/'train_f.yaml').read_text())
        self.assertEqual(learning_rates(train,self.f,100)['finite_G'],0)
        self.assertAlmostEqual(learning_rates(train,self.f,7999)['legacy'],.0006)
        self.assertAlmostEqual(learning_rates(train,self.f,8000)['finite_G'],.0006)
        self.assertAlmostEqual(learning_rates(train,self.f,8000)['finite_B'],.00006)
        self.assertLessEqual(calibrated_scale([1,1],[1e-20,1e-20]),10)
        self.assertAlmostEqual(safety_recalibrate(4,.1,1,2),.75)
        self.assertAlmostEqual(ema_decay(1,.99),.5)

if __name__=='__main__':unittest.main()
