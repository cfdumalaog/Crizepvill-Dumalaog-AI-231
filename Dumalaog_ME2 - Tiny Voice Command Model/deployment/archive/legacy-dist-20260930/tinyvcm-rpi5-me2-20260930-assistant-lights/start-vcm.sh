#!/usr/bin/env bash
set -euo pipefail

app_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv="$HOME/.venvs/tinyvcm-rpi5"
if [[ ! -x "$venv/bin/python" ]]; then
  echo "Tiny VCM runtime is missing: $venv/bin/python" >&2
  echo "Use the existing Pi runtime setup documented in README.md." >&2
  exit 1
fi
cd "$app_root"
exec "$venv/bin/python" "$app_root/antigrav_demo.py" \
  --host 127.0.0.1 \
  --port 7860 \
  --model "$app_root/models/intent_31_int8.onnx" \
  --binary-wake-model "$app_root/models/wake_binary_int8.onnx" \
  --wake-threshold 0.95 \
  --vad-threshold 0.006 \
  --inference-interval 0.12 \
  --timeout 10.0 \
  --gpio \
  --live-weather \
  "$@"
