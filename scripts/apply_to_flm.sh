#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?Usage: $0 /path/to/flm-with-frozen-release-applied}"
base_commit="a1918d5164e5038e37d0b7a4fb2010ce75b863b3"

[[ -e "$target/.git" ]] || {
  echo "Target is not a Git checkout: $target" >&2
  exit 2
}
[[ "$(git -C "$target" rev-parse HEAD)" == "$base_commit" ]] || {
  echo "Target must use the FrozenMSE upstream base $base_commit" >&2
  exit 2
}
for required in \
  langflow_hybrid/model.py \
  configs/task1_m40_stable_to_50k.env \
  scripts/train_task1_a.sh; do
  [[ -f "$target/$required" ]] || {
    echo "Apply codex/frozen-mse-50k-release first; missing $required" >&2
    exit 2
  }
done
while IFS= read -r path; do
  [[ ! -e "$target/$path" ]] || {
    echo "Refusing to overwrite an existing small-step file: $path" >&2
    exit 2
  }
done < "$release_root/manifest/code-files.txt"

git -C "$target" apply --check \
  "$release_root/patches/frozen-release-integration.patch"
git -C "$target" apply \
  "$release_root/patches/frozen-release-integration.patch"
cp -R "$release_root/code/." "$target/"

while IFS= read -r path; do
  [[ -e "$target/$path" ]] || {
    echo "Small-step file was not copied: $path" >&2
    exit 1
  }
done < "$release_root/manifest/code-files.txt"

echo "Small-step TVM code applied to: $target"
