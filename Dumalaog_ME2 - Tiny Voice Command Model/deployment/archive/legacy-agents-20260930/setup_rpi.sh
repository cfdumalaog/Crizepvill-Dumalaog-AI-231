#!/usr/bin/env bash
set -euo pipefail
vcm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo 'Install Raspberry Pi OS 64-bit first (aarch64 required).' >&2
  exit 1
fi
if [[ "$(id -u)" == 0 ]]; then
  echo 'Run as your normal Pi user; this script calls sudo only for system setup.' >&2
  exit 1
fi
if [[ "$vcm_root" =~ [[:space:]] ]]; then
  echo 'Copy the release folder to ~/vcm (a path without spaces) first.' >&2
  exit 1
fi
vcm_user="$(id -un)"
vcm_env="$HOME/.venv"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-gpiozero python3-lgpio libportaudio2 alsa-utils
# One shared Python environment on the Pi. Never copy the Windows .venv.
if [[ ! -x "$vcm_env/bin/python" ]]; then
  python3 -m venv --system-site-packages "$vcm_env"
fi
"$vcm_env/bin/python" -m pip install --upgrade pip
"$vcm_env/bin/python" -m pip install --only-binary=:all: -r "$vcm_root/requirements-runtime.txt"
sudo usermod -aG audio,gpio "$vcm_user"
cd "$vcm_root"
"$vcm_env/bin/python" verify_release.py
"$vcm_env/bin/python" -m tinyvcm.runtime --output pi_benchmark.json
sudo tee /etc/systemd/system/tiny-vcm.service >/dev/null <<EOF
[Unit]
Description=Tiny VCM local voice command assistant
After=sound.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$vcm_user
WorkingDirectory=$vcm_root
Environment=PYTHONUNBUFFERED=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
EnvironmentFile=-$vcm_root/vcm.env
ExecStart=$vcm_env/bin/python $vcm_root/demo.py \$VCM_ARGS
Restart=on-failure
RestartSec=5
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
printf '\nInstalled. Plug in a USB microphone and USB/HDMI speaker.\n'
printf 'Test: cd %s && %s/bin/python demo.py --check-seconds 10\n' "$vcm_root" "$vcm_env"
printf 'After the microphone and wiring tests pass: sudo systemctl enable --now tiny-vcm\n'
printf 'Dashboard on Pi: http://127.0.0.1:7861  | Logs: journalctl -u tiny-vcm -f\n'
