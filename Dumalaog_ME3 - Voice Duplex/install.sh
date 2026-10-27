#!/usr/bin/env bash
# ME3 - Voice Duplex: idempotent installer for Raspberry Pi 4/5
# Usage: bash install.sh   (run on the Pi, as the dalmacio user)
# Safe to re-run: every step checks before acting.
set -euo pipefail

echo "== ME3 Voice Duplex installer =="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Raspberry Pi readiness: keep the source tree portable and select the ASR
# model at install time from available RAM.  The checked-in config remains the
# default; systemd receives the selection through a non-secret env file.
ARCH="$(uname -m 2>/dev/null || echo unknown)"
RAM_MB="$(awk '/MemTotal:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "  warning: expected 64-bit Pi (aarch64/arm64), detected $ARCH"
fi
if [ -n "${ME3_ASR_MODEL:-}" ]; then
  ASR_MODEL="$ME3_ASR_MODEL"
elif [ "$RAM_MB" -gt 0 ] && [ "$RAM_MB" -le 5120 ]; then
  ASR_MODEL="base.en"
else
  ASR_MODEL="small.en"
fi
OVERLAP_VOICE="${ME3_OVERLAP_VOICE:-en_US-lessac-medium}"
export ME3_ASR_MODEL="$ASR_MODEL"
export ME3_OVERLAP_VOICE="$OVERLAP_VOICE"
echo "  architecture: $ARCH | RAM: ${RAM_MB} MB | ASR: $ASR_MODEL"

# 1. System deps (audio + python build tools)
echo "[1/6] system packages..."
if command -v apt-get >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y python3-venv python3-dev portaudio19-dev \
       libsndfile1 libopenblas-dev ffmpeg
else
  echo "  apt-get not found (non-Debian?) - skipping system pkgs"
fi

# 2. venv
echo "[2/6] virtualenv..."
VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# 3. Python deps
echo "[3/6] python dependencies (pinned)..."
pip install -r requirements.txt

# 4. Models (download once)
echo "[4/6] models..."
mkdir -p models
python - <<'PY'
import os
# faster-whisper: pulls the RAM-selected CTranslate2 model on first use; prefetch it now
try:
    from faster_whisper import WhisperModel
    WhisperModel(os.environ["ME3_ASR_MODEL"], device="cpu", compute_type="int8")
    print("  faster-whisper", os.environ["ME3_ASR_MODEL"], "cached")
except Exception as e:
    print("  faster-whisper prefetch skipped:", e)
# piper voice
try:
    import subprocess, sys
    for voice in ("en_GB-alba-medium", os.environ["ME3_OVERLAP_VOICE"]):
        subprocess.run([
            sys.executable, "-m", "piper.download_voices", voice,
            "--download-dir", "models/tts",
        ], check=True)
        print("  piper voice cached:", voice)
except Exception as e:
    print("  piper voice prefetch skipped:", e)
PY

# 5. Silero VAD ONNX is bundled by Pipecat; no model download is needed.
echo "[5/6] VAD model (Pipecat-bundled CPU ONNX)..."
echo "  no torch/CUDA dependency is required for the ME3 VAD path"

# 6. systemd units (optional autostart)
echo "[6/6] systemd units..."
UNITS_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNITS_DIR"
ENV_DIR="$HOME/.config/me3"
mkdir -p "$ENV_DIR"
# VLLM_API_KEY: the tool brain (Qwen @ vLLM) needs it.  The key lives on the
# Windows dev box as a DPAPI blob (~/.codex/vllm-api-key.dpapi), which the Pi
# cannot decrypt.  Export it once on Windows (powershell -File
# ~/.codex/Get-VllmApiKey.ps1) and place the plaintext in
# ~/.codex/vllm-api-key.txt on the Pi, or export VLLM_API_KEY in the shell.
VLLM_KEY="${VLLM_API_KEY:-}"
if [ -z "$VLLM_KEY" ] && [ -f "$HOME/.codex/vllm-api-key.txt" ]; then
  VLLM_KEY="$(cat "$HOME/.codex/vllm-api-key.txt")"
fi
# Key files copied between Windows and Linux may carry a trailing CR/LF. Keep
# the secret in memory, but normalize transport whitespace before writing the
# runtime env file so it cannot corrupt an HTTP Authorization header.
VLLM_KEY="${VLLM_KEY//$'\r'/}"
VLLM_KEY="${VLLM_KEY//$'\n'/}"
ELEVENLABS_KEY="${ELEVENLABS_API_KEY:-}"
if [ -z "$ELEVENLABS_KEY" ] && [ -f "$ENV_DIR/elevenlabs-api-key.txt" ]; then
  ELEVENLABS_KEY="$(cat "$ENV_DIR/elevenlabs-api-key.txt")"
fi
ELEVENLABS_KEY="${ELEVENLABS_KEY//$'\r'/}"
ELEVENLABS_KEY="${ELEVENLABS_KEY//$'\n'/}"
ELEVENLABS_VOICE="${ELEVENLABS_VOICE_ID:-}"
TTS_ENGINE="${ME3_TTS_ENGINE:-auto}"
# Preserve channel credentials already provisioned in the protected runtime
# file.  This is line-based rather than `source`-based so a malformed env file
# cannot execute arbitrary shell code during an upgrade.  Values never reach
# stdout; only configured/not-configured summaries are printed below.
for env_name in \
  TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_FROM_NUMBER \
  TWILIO_MESSAGING_SERVICE_SID TWILIO_VOICE_URL RESEND_API_KEY RESEND_FROM_EMAIL; do
  if [ -z "${!env_name:-}" ] && [ -f "$ENV_DIR/me3.env" ]; then
    old_line="$(grep -m1 -E "^${env_name}=" "$ENV_DIR/me3.env" || true)"
    old_value="${old_line#*=}"
    old_value="${old_value//$'\r'/}"
    printf -v "$env_name" '%s' "$old_value"
  fi
done
cat > "$ENV_DIR/me3.env" <<EOF
ME3_ASR_MODEL=$ASR_MODEL
ME3_OVERLAP_VOICE=$OVERLAP_VOICE
ME3_TTS_ENGINE=$TTS_ENGINE
ELEVENLABS_API_KEY=${ELEVENLABS_KEY:-}
ELEVENLABS_VOICE_ID=${ELEVENLABS_VOICE:-}
VLLM_API_KEY=${VLLM_KEY:-${VLLM_API_KEY:-}}
TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID:-}
TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN:-}
TWILIO_FROM_NUMBER=${TWILIO_FROM_NUMBER:-}
TWILIO_MESSAGING_SERVICE_SID=${TWILIO_MESSAGING_SERVICE_SID:-}
TWILIO_VOICE_URL=${TWILIO_VOICE_URL:-}
RESEND_API_KEY=${RESEND_API_KEY:-}
RESEND_FROM_EMAIL=${RESEND_FROM_EMAIL:-}
EOF
chmod 600 "$ENV_DIR/me3.env"
if [ -n "$VLLM_KEY" ] || [ -n "${VLLM_API_KEY:-}" ]; then
  echo "  VLLM_API_KEY: set (tool brain enabled)"
else
  echo "  VLLM_API_KEY: EMPTY -> agent will use the stub brain."
  echo "  Export the key from Windows (see README 'Pi deployment') and put it"
  echo "  in ~/.codex/vllm-api-key.txt or export VLLM_API_KEY before running."
fi
if [ -n "$ELEVENLABS_KEY" ]; then
  echo "  ELEVENLABS_API_KEY: set (value hidden; streaming voice enabled when TTS=auto)"
else
  echo "  ELEVENLABS_API_KEY: not configured -> Piper fallback remains active"
fi
if [ -n "${TWILIO_ACCOUNT_SID:-}" ] && [ -n "${TWILIO_AUTH_TOKEN:-}" ]; then
  echo "  Twilio credentials: set (value hidden)"
else
  echo "  Twilio credentials: not configured -> call/SMS tools remain gated"
fi
if [ -n "${RESEND_API_KEY:-}" ] && [ -n "${RESEND_FROM_EMAIL:-}" ]; then
  echo "  Resend credentials: set (value hidden)"
else
  echo "  Resend credentials: not configured -> email tool remains gated"
fi
for svc in me3-asr me3-tts me3-vad me3-agent; do
  if [ -f "$SCRIPT_DIR/deploy/$svc.service" ]; then
    cp "$SCRIPT_DIR/deploy/$svc.service" "$UNITS_DIR/$svc.service"
    sed -i "s|__ME3_DIR__|$SCRIPT_DIR|g" "$UNITS_DIR/$svc.service"
    sed -i "s|__VENV__|$VENV|g" "$UNITS_DIR/$svc.service"
    echo "  installed $svc.service"
  fi
done
systemctl --user daemon-reload || true
echo "  enable with: systemctl --user enable --now me3-agent"

echo ""
echo "== install complete =="
echo "Runtime env: $ENV_DIR/me3.env"
echo "Start agent:  source $VENV/bin/activate && python -m me3.agent"
echo "Run bench:    python -m me3.benchmark --trials 20"
