#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/autodl_env.sh"
if [[ -f /etc/network_turbo ]]; then source /etc/network_turbo; fi
destination="$FLM_STORAGE_DIR/data/models/gpt2-large"
revision=32b71b12589c2f8d625668d2335a01cac3249519
mkdir -p "$destination"
for file in config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json; do
  if [[ ! -f "$destination/$file" ]]; then
    curl -fL --connect-timeout 15 --max-time 900 --speed-limit 1024 --speed-time 60 --retry 1 \
      -o "$destination/$file.part" \
      "https://huggingface.co/openai-community/gpt2-large/resolve/$revision/$file"
    mv "$destination/$file.part" "$destination/$file"
  fi
done
cd "$destination"
sha256sum -c "$FLM_PROJECT_DIR/scripts/gpt2_large_frozen.sha256"
