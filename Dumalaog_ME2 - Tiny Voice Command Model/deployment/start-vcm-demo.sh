#!/usr/bin/env bash
set -euo pipefail

app_root="/home/dalmacio/Desktop/dandan"
url="http://127.0.0.1:7860/assistant"

if ! curl -fsS http://127.0.0.1:7860/api/state >/dev/null 2>&1; then
  nohup "$app_root/launch-vcm.sh" >"$app_root/vcm.log" 2>&1 </dev/null &
  ready=false
  for _ in $(seq 1 20); do
    if curl -fsS http://127.0.0.1:7860/api/state >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    echo "VCM did not start. See $app_root/vcm.log" >&2
    exit 1
  fi
fi

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$url" >/dev/null 2>&1 || true
fi
