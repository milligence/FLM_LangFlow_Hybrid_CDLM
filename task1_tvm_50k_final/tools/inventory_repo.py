#!/usr/bin/env python3
"""Read-only initial inventory. Candidate ranking is not automatic science resolution."""
from __future__ import annotations
import argparse,subprocess,json,hashlib
from pathlib import Path
SKIP={'.git','.venv','venv','node_modules','__pycache__','checkpoints','wandb','.cache','data','datasets'}
KEYS=['tvm','jvp','langflow','flm','train','sampler','self_condition','tokenizer','metric','ema','config']

def git(root:Path,*args:str)->str:
    p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
    return p.stdout if p.returncode==0 else p.stderr

def main():
    p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.repo.resolve();rows=[]
    # Sources/configs only. Never serialize .env, credentials, model bytes or datasets.
    def walk(d:Path):
        for f in sorted(d.iterdir()):
            if f.name in SKIP or f.is_symlink():continue
            if f.is_dir():
                yield from walk(f)
            elif f.suffix in {'.py','.yaml','.yml','.toml'} and f.stat().st_size<=2_000_000:
                yield f
    for f in walk(root):
        rel=str(f.relative_to(root));score=sum(x in rel.lower() for x in KEYS)
        if score:rows.append({'path':rel,'candidate_score':score,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
        if len(rows)>=5000:break
    report={'repo_root':str(root),'git_head':git(root,'rev-parse','HEAD').strip(),
            'git_status_porcelain':git(root,'status','--porcelain=v1'),
            'git_worktrees':git(root,'worktree','list','--porcelain'),
            'candidates':sorted(rows,key=lambda x:(-x['candidate_score'],x['path'])),
            'possibly_truncated':len(rows)>=5000,
            'does_not_resolve':['successful_baseline_identity','entrypoint_semantics','LUT_identity','metric_definitions']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(a.output)
if __name__=='__main__':main()
