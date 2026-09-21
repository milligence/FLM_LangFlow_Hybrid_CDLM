#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?Usage: $0 /path/to/flm}"
base_commit="a1918d5164e5038e37d0b7a4fb2010ce75b863b3"

[[ -e "$target/.git" ]] || {
  echo "Target is not a Git checkout: $target" >&2
  exit 2
}
[[ "$(git -C "$target" rev-parse HEAD)" == "$base_commit" ]] || {
  echo "Target must be checked out at $base_commit" >&2
  exit 2
}
[[ -z "$(git -C "$target" status --porcelain)" ]] || {
  echo "Target working tree must be clean before applying the release" >&2
  exit 2
}

git -C "$target" apply --check "$release_root/patches/upstream-integration.patch"
git -C "$target" apply "$release_root/patches/upstream-integration.patch"
cp -R "$release_root/code/." "$target/"

while IFS= read -r path; do
  [[ -e "$target/$path" ]] || {
    echo "Release file was not copied: $path" >&2
    exit 1
  }
done < "$release_root/manifest/code-files.txt"

echo "FrozenMSE-50k code applied to: $target"
