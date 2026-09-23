#!/usr/bin/env python3
"""Validate static contract consistency; does not claim real-model/GPU correctness."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import yaml

HASH_FILES=['train_f.yaml','train_p.yaml','sampler_f.yaml','sampler_p.yaml','eval.yaml',
 'resolved_delta_v2.yaml','diagnostic_record_v2.schema.json',
 'reference/objectives.py','reference/control.py','reference/samplers.py',
 'tools/decision_policy.py','tools/paired_statistics.py',
 'tools/build_patch_decision_input.py']
def contract_hash(root:Path)->str:
    h=hashlib.sha256()
    for name in HASH_FILES:
        h.update(name.encode()+b'\0');h.update((root/name).read_bytes());h.update(b'\0')
    return h.hexdigest()

def validate(root:Path)->dict:
    errors=[];checks=[]
    def check(ok,msg):
        (checks if ok else errors).append(msg)
    for line in ['f','p']:
        t=yaml.safe_load((root/f'train_{line}.yaml').read_text())
        s=yaml.safe_load((root/f'sampler_{line}.yaml').read_text())
        check(t['global_batch']==256 and t['seq_len']==128,f'{line}: local batch and seq')
        check(t['microbatch']*t['grad_accum']==256,f'{line}: accumulation')
        check(t['objective']['map_batch']==96,f'{line}: map subset')
        check(t['initialize_from_checkpoint'] is None,f'{line}: from scratch')
        check(not s['map']['source_law']['chi_hard_filter'],f'{line}: no chi hard gate')
        check(not t['precision']['detach_jvp'] and not t['precision']['finite_difference_training'],f'{line}: exact connected JVP')
        for part in ['local','map']:
            stages=s[part]['stages'];prev=0
            for z in stages:
                check(z['start_step']==prev and z['end_step']>prev,f'{line}: {part} continuous stage {prev}')
                prev=z['end_step']
                if part=='local' and z['quotas'] is not None:
                    check(sum(z['quotas'])==256,f'{line}: local quota sum at {z["start_step"]}')
                if part=='map':
                    check(sum(z['counts'].values())==z['map_batch'],f'{line}: map quota sum at {z["start_step"]}')
                    check(z['map_batch'] in [0,96],f'{line}: no hidden map batch changes')
                    if z['start_step']>=5000:
                        expected = (
                            {'S':12,'M':24,'L':16,'D':24,'D_patch':8,'Z':6,'H':6}
                            if line == 'f' and z['start_step'] >= 32000 else
                            {'S':12,'M':24,'L':24,'D':24,'Z':6,'H':6})
                        check(z['counts']==expected,f'{line}: final quota at {z["start_step"]}')
            check(prev==50000,f'{line}: {part} ends at 50k')
        check(s['map']['hard_weight']['end_step']==(8000 if line=='f' else 10000),f'{line}: hard deadline')
        check(t['objective']['terminal_T']==.95,f'{line}: terminal')
        check(t['optimizer']['lr_schedule']==[[0,0.],[2500,.0006],[30000,.0006],[30500,.0003],[50000,.0003]],f'{line}: LR')
        check(t['optimizer']['automatic_lr_changes'] is False,f'{line}: no adaptive LR')
        check(t['stop']['no_automatic_strategy_forks'],f'{line}: no strategy forks')
    f=yaml.safe_load((root/'sampler_f.yaml').read_text());p=yaml.safe_load((root/'sampler_p.yaml').read_text())
    expected_profiles={
        'pre32':({'S':12,'M':24,'L':24,'D':24,'D_patch':0,'Z':6,'H':6},[0,0,0,0]),
        'uniform':({'S':12,'M':24,'L':16,'D':24,'D_patch':8,'Z':6,'H':6},[2,2,2,2]),
        'late_trial':({'S':12,'M':24,'L':16,'D':24,'D_patch':8,'Z':6,'H':6},[0,0,2,6]),
        'late_fixed':({'S':12,'M':24,'L':16,'D':24,'D_patch':8,'Z':6,'H':6},[0,0,2,6]),
        'off':({'S':12,'M':24,'L':24,'D':24,'D_patch':0,'Z':6,'H':6},[0,0,0,0]),
    }
    if 'profiles' in f['map']:
        for name,(counts,intervals) in expected_profiles.items():
            profile=f['map']['profiles'].get(name,{})
            check(profile.get('counts')==counts,f'F v2 profile counts {name}')
            check(profile.get('D_patch_interval_rows')==intervals,f'F v2 intervals {name}')
            check(sum(counts.values())==96,f'F v2 budget {name}')
    else:
        # Archived pre-30k contract predates the D_patch profile switch.
        check(f['map']['final_exact_quota_atom_r_zero']==18,
              'F pre30 exact r-zero quota')
        check(f['map']['stages'][-1]['counts']==
              {'S':12,'M':24,'L':24,'D':24,'Z':6,'H':6},
              'F pre30 final map quota')
    check(f['local']==p['local'],'identical fixed local sampler contracts')
    ev=yaml.safe_load((root/'eval.yaml').read_text())
    check(ev['formal_steps']==[10000,20000,30000,50000],'formal eval milestones')
    for name,nodes in ev['finite']['grids'].items():
        check(nodes[0]==0 and nodes[-1]==.95 and all(a<b for a,b in zip(nodes,nodes[1:])),f'eval grid {name}')
    return {'scope':'static_contract_only','passed':not errors,'contract_sha256':contract_hash(root),
            'checks_count':len(checks),'errors':errors,'checks':checks,
            'not_tested':['real_repository_integration','real_DDiT_hand_JVP','CUDA_runtime','training','generation_quality']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--write',type=Path)
    a=ap.parse_args();report=validate(a.root)
    text=json.dumps(report,ensure_ascii=False,indent=2)
    if a.write:a.write.write_text(text+'\n')
    print(text);sys.exit(0 if report['passed'] else 1)
