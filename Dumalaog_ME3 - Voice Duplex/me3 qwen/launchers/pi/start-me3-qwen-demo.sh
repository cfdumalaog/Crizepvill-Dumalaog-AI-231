#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT="me3-agent.service"
WEB_URL="http://127.0.0.1:8126/api/state"

printf '\nME3 Qwen Instruct demo startup\n'
printf 'Project: %s\n' "$PROJECT"
printf 'Host:    %s.local\n' "$(hostname -s)"
printf 'Starting the Pi systemd service...\n'

systemctl --user enable "$UNIT"
systemctl --user restart "$UNIT"

state=""
loop_running="false"
for _ in $(seq 1 40); do
  if state="$(curl -fsS --max-time 3 "$WEB_URL" 2>/dev/null)"; then
    loop_running="$(printf '%s' "$state" | "$PROJECT/.venv/bin/python" -c \
      'import json,sys; print("true" if json.load(sys.stdin).get("loop") else "false")' 2>/dev/null || printf 'false')"
    if [[ "$loop_running" == "true" ]]; then
      break
    fi
  fi
  sleep 1
done

printf '\nService: '
systemctl --user is-active "$UNIT" || true
printf 'Pi addresses: '
hostname -I | xargs
printf 'Stable local name: %s.local\n' "$(hostname -s)"
printf 'Voice loop: %s\n' "$loop_running"

if systemctl is-active --quiet openvpn-client@sharchpc.service; then
  printf 'SHARC VPN service: active\n'
else
  printf 'SHARC VPN service: inactive\n'
fi

port_check() {
  local port="$1"
  timeout 3 bash -c "</dev/tcp/10.153.60.233/$port" >/dev/null 2>&1
}
if port_check 8000 && port_check 8001; then
  printf 'DGX3 Qwen ASR/TTS ports: reachable from Pi\n'
else
  printf 'DGX3 Qwen ASR/TTS ports: not reachable from Pi; check VPN or DILNET\n'
fi

if [[ -z "$state" ]]; then
  printf '\nERROR: ME3 web API did not become ready.\n' >&2
  exit 1
fi
if [[ "$loop_running" != "true" ]]; then
  printf '\nERROR: the UI is responding, but the voice loop did not start.\n' >&2
  printf 'Check the Pi microphone connection, then use the UI restart control.\n' >&2
  exit 2
fi

printf '\nPi side is ready. On the laptop, open “ME3 Qwen Laptop Demo” to\n'
printf 'tunnel the UI/audio ports and use the laptop speaker.\n'
printf 'The laptop UI is http://127.0.0.1:8100/ (Spotify callback uses this tunnel).\n'
