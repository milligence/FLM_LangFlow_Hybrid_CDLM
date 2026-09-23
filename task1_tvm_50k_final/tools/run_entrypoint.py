#!/usr/bin/env python3
"""Fail-closed bridge to repository entrypoints resolved by Codex.

This script intentionally does not invent a Transformer or training entrypoint.
It launches only a bound adapter, in the existing repository, with verified
contract/provenance. No shell evaluation and no hyperparameters on the CLI.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from validate_contract import validate,contract_hash


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','evaluate','precheck'])
    p.add_argument('--line',choices=['f','p'],required=True)
    p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--gpu',required=True)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--resume',default='auto',help='auto, never, or absolute full-state checkpoint')
    p.add_argument('--steps',default=None,help='Evaluation checkpoint subset only; never controls training length')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    static=validate(root)
    if not static['passed']:raise RuntimeError('Static contract validation failed: '+str(static['errors']))
    bindings=json.loads(a.bindings.read_text())
    if bindings.get('status')!='RESOLVED':
        raise RuntimeError('Repository bindings are unresolved. Codex must complete inventory and adapter integration first.')
    repo=Path(bindings['repo_root']).resolve()
    if not repo.is_dir():raise RuntimeError('Missing existing repository')
    py=Path(bindings['python_executable'])
    if not py.is_file():raise RuntimeError('Missing resolved environment Python')
    if ',' in a.gpu or not a.gpu.strip():raise ValueError('One GPU per independent line; no multi-GPU list')
    runroot=(a.output_root or Path(bindings['run_root'])).resolve()
    runroot.mkdir(parents=True,exist_ok=True)
    manifest=Path(bindings['provenance']['source_manifest_path'])
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest!=bindings['provenance']['source_sha256_manifest']:
        raise RuntimeError('Source-manifest file digest changed; re-audit before launch')
    # Verify current source bytes, not just the manifest's own hash.
    sources=json.loads(manifest.read_text())
    for item in sources['files']:
        path=(repo/item['path']).resolve()
        if not path.is_relative_to(repo):raise RuntimeError('Source manifest escapes repo root')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError(f'Source changed after precheck: {path}')
    if a.action in ['train','evaluate']:
        report_path=bindings.get('precheck_report')
        if not report_path:raise RuntimeError('Missing real-repository precheck report')
        report=json.loads(Path(report_path).read_text())
        if not(report.get('scope')=='repository_full_model' and report.get('passed') is True):
            raise RuntimeError('CPU reference tests are not a real-model acceptance report')
        if report.get('contract_sha256')!=contract_hash(root) or report.get('source_manifest_sha256')!=digest:
            raise RuntimeError('Precheck was run against different code/contract')
        if a.line.upper() not in report.get('accepted_lines',[]):
            raise RuntimeError('Requested line did not pass full-model acceptance')
    command=bindings['commands'].get(a.action)
    if not isinstance(command,list) or not command or not all(isinstance(s,str) for s in command):
        raise RuntimeError('Resolved command must be an argv-prefix list')
    command=[s.replace('{python}',str(py)) for s in command]
    if any('<RESOLVED_' in s for s in command):raise RuntimeError('Unresolved entrypoint placeholder')
    output=runroot/f'task1_tvm_{a.line}_scratch_50k_final_v1'
    if a.action=='evaluate' and not output.exists():raise RuntimeError('No output directory to evaluate')
    output.mkdir(parents=True,exist_ok=True);logs=output/'logs';logs.mkdir(exist_ok=True)
    if a.resume not in ['auto','never'] and not Path(a.resume).is_absolute():
        raise ValueError('Explicit resume checkpoint must be absolute')
    argv=command+['--contract-dir',str(root),'--line',a.line,'--bindings',str(a.bindings.resolve()),
                  '--output',str(output)]
    if a.action=='train':argv+=['--resume',a.resume]
    if a.action=='evaluate' and a.steps:argv+=['--steps',a.steps]
    lockdir=runroot/'.locks';lockdir.mkdir(exist_ok=True)
    safe_gpu=''.join(c for c in a.gpu if c.isalnum() or c in '-_')
    with (lockdir/f'gpu_{safe_gpu}.lock').open('a+') as gpu_lock, \
         (output/'.process.lock').open('a+') as run_lock:
        for handle in [gpu_lock,run_lock]:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('GPU/run already held by another contract process; do not steal it')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu
        env['TASK1_CONTRACT_SHA256']=contract_hash(root)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        meta={'argv':argv,'cwd':str(repo),'gpu':a.gpu,'action':a.action,'line':a.line,
              'contract_sha256':contract_hash(root),'source_manifest_sha256':digest}
        (logs/f'{stamp}_{a.action}_launch.json').write_text(json.dumps(meta,indent=2)+'\n')
        with (logs/f'{stamp}_{a.action}.log').open('w',buffering=1) as logfile:
            proc=subprocess.Popen(argv,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,bufsize=1)
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line);sys.stdout.flush();logfile.write(line)
                return proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(2)
                return proc.wait()

if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,ValueError,KeyError,FileNotFoundError) as e:
        print(f'BLOCKED: {e}',file=sys.stderr);sys.exit(2)
