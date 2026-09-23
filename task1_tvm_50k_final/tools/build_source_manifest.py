#!/usr/bin/env python3
"""Build the exact source-byte manifest for the bound FLM worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXPLICIT_UNTRACKED = (
    'task1_tvm_50k_final.py',
    'task1_tvm_50k_final_adapter.py',
    'task1_tvm_50k_precheck.py',
    'tests/test_task1_tvm_50k_final.py',
    'configs/algo/task1_tvm_50k_final_f.yaml',
    'configs/algo/task1_tvm_50k_final_p.yaml',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    tracked = subprocess.check_output(
        ['git', 'ls-files'], cwd=repo, text=True).splitlines()
    paths = {
        value for value in tracked
        if value.endswith(('.py', '.yaml', '.yml'))
        or value in {'requirements.txt', 'pyproject.toml'}
    }
    paths.update(EXPLICIT_UNTRACKED)
    files = []
    for relative in sorted(paths):
        path = repo / relative
        if not path.is_file():
            continue
        files.append({
            'path': relative,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    payload = {
        'schema': 'task1-source-manifest-v1',
        'repo_root_at_build': str(repo),
        'git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'files': files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
