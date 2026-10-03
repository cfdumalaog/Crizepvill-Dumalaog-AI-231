#!/usr/bin/env bash
set -euo pipefail

app_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv="$HOME/.venvs/tinyvcm-rpi5"
if [[ ! -x "$venv/bin/python" ]]; then
  echo "Tiny VCM runtime is missing: $venv/bin/python" >&2
  echo "Install the Pi runtime by following README.md." >&2
  exit 1
fi
cd "$app_root"
mapfile -t settings < <(/usr/bin/python3 - "$app_root/deployment.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("port", "wake_threshold", "vad_threshold", "inference_interval_sec",
            "post_wake_timeout_sec", "gpio", "live_weather", "wake_confirmations"):
    print(str(cfg[key]).lower() if isinstance(cfg[key], bool) else cfg[key])
PY
)
args=(
  --host 127.0.0.1
  --port "${settings[0]}"
  --model "$app_root/models/intent_31_int8.onnx"
  --binary-wake-model "$app_root/models/wake_binary_int8.onnx"
  --wake-threshold "${settings[1]}"
  --vad-threshold "${settings[2]}"
  --inference-interval "${settings[3]}"
  --timeout "${settings[4]}"
  --wake-confirmations "${settings[7]}"
)
[[ "${settings[5]}" == "true" ]] && args+=(--gpio)
[[ "${settings[6]}" == "true" ]] && args+=(--live-weather)
exec "$venv/bin/python" "$app_root/vcm_app.py" "${args[@]}" "$@"
