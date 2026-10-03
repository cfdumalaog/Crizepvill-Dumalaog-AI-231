#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This bundle requires 64-bit Raspberry Pi OS (aarch64); found $(uname -m)." >&2
  exit 1
fi
if [[ "$bundle_root" =~ [[:space:]] ]]; then
  echo "Install under a path without spaces (for example ~/tinyvcm-rpi5)." >&2
  exit 1
fi
if [[ "$(id -u)" == "0" ]]; then
  echo "Run this script as your regular Pi user; it uses sudo only for OS setup." >&2
  exit 1
fi

python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
venv="$HOME/.venvs/tinyvcm-rpi5"
sudo apt-get update
sudo apt-get install -y python3-venv python3-gpiozero python3-lgpio libportaudio2 alsa-utils
mkdir -p "$(dirname -- "$venv")"
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$venv"
fi
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install --only-binary=:all: -r "$bundle_root/requirements-runtime.txt"

for group in audio gpio; do
  if getent group "$group" >/dev/null; then
    sudo usermod -aG "$group" "$(id -un)"
  fi
done

cd "$bundle_root"
"$venv/bin/python" verify_release.py

service_tmp="$(mktemp)"
trap 'rm -f "$service_tmp"' EXIT
cat >"$service_tmp" <<EOF
[Unit]
Description=Tiny VCM Antigrav voice command assistant
After=sound.target
Wants=sound.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$bundle_root
Environment=PYTHONUNBUFFERED=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
ExecStart=$venv/bin/python $bundle_root/antigrav_demo.py --host 127.0.0.1 --port 7860
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
NoNewPrivileges=yes
ProtectSystem=full
ProtectKernelTunables=yes
ProtectControlGroups=yes

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 0644 "$service_tmp" /etc/systemd/system/tiny-vcm-antigrav.service
sudo systemctl daemon-reload

cat <<EOF
Runtime setup and offline model smoke checks passed.
Bundle: $bundle_root
Python: $venv/bin/python
Microphone check: $venv/bin/python -m tinyvcm.microphone
Foreground demo: $venv/bin/python antigrav_demo.py --host 127.0.0.1 --port 7860
Local UI: http://127.0.0.1:7860/assistant
After microphone and wake/command checks pass, enable startup with:
  sudo systemctl enable --now tiny-vcm-antigrav.service
GPIO remains disabled until you explicitly run the app with --gpio and verify wiring.
EOF
