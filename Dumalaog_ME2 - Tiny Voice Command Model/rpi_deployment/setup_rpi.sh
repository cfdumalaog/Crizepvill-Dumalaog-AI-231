#!/usr/bin/env bash
# ==============================================================================
# Tiny Voice Command Model (VCM) - 1-Command Raspberry Pi 4/5 Setup Script
# Automatically installs system packages, enables I2C, creates virtualenv,
# installs PyTorch/audio dependencies, and registers the systemd service.
# ==============================================================================

set -e

echo "=================================================================="
echo "    Configuring Raspberry Pi 4/5 for Tiny Voice Command Model"
echo "=================================================================="

# 1. Update APT Repositories & Install Required Linux Packages
echo "[1/6] Installing Linux audio, I2C, and build packages..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-tk \
    libasound2-dev \
    portaudio19-dev \
    libatlas-base-dev \
    i2c-tools \
    alsa-utils \
    espeak-ng \
    sox \
    libsox-fmt-all

# 2. Enable I2C interface for OLED Display
echo "[2/6] Enabling Raspberry Pi I2C hardware bus..."
if command -v raspi-config &> /dev/null; then
    sudo raspi-config nonint do_i2c 0
    echo "I2C interface enabled."
else
    echo "raspi-config not found (skipping automatic I2C toggle)."
fi

# Add current user to gpio, audio, and i2c groups
sudo usermod -a -G gpio,audio,i2c "$USER" || true

# 3. Create Dedicated Python Virtual Environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv_rpi"

echo "[3/6] Setting up Python virtual environment at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 4. Install Edge Python Libraries
echo "[4/6] Installing Python packages (PyTorch, TorchAudio, GPIO, I2C OLED)..."
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU wheels and audio processing
pip install \
    torch \
    torchaudio \
    sounddevice \
    soundfile \
    numpy \
    gpiozero \
    adafruit-circuitpython-ssd1306 \
    pillow \
    gradio

# Install appropriate GPIO backend (RPi 5 uses RP1 controller via rpi-lgpio; RPi 4 can use RPi.GPIO)
if grep -q "Raspberry Pi 5" /proc/device-tree/model 2>/dev/null; then
    echo "Detected Raspberry Pi 5: Installing rpi-lgpio backend for RP1 chip..."
    pip install rpi-lgpio || true
else
    echo "Detected Raspberry Pi 4 / earlier: Installing standard GPIO backend..."
    pip install rpi-lgpio || pip install RPi.GPIO || true
fi

# 5. Install & Enable Systemd Service (Appliance Auto-Start)
echo "[5/6] Registering systemd appliance service (vcm.service)..."
SERVICE_FILE="/etc/systemd/system/vcm.service"
sudo bash -c "cat <<EOF > $SERVICE_FILE
[Unit]
Description=Tiny Voice Command Model (VCM) Smart Assistant
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$VENV_DIR/bin/python $SCRIPT_DIR/run_vcm.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
echo "Systemd service installed. To enable auto-start on boot, run:"
echo "    sudo systemctl enable --now vcm.service"

# 6. Audio Device Diagnostics
echo "[6/6] Probing connected audio hardware..."
echo "--- Connected Recording Microphones ---"
arecord -l || true
echo "--- Connected Playback Speakers ---"
aplay -l || true

echo "=================================================================="
echo " SETUP COMPLETE! Available Run Modes on Raspberry Pi 5:"
echo " 1. Native HDMI Monitor Screen (Voiced + Kiosk Display):"
echo "    python monitor_display.py"
echo " 2. Zero-Button Hands-Free Always-Listening (Console):"
echo "    python live_listen.py"
echo " 3. Interactive Web Dashboard (Local Network):"
echo "    python app.py"
echo " 4. Automatic Systemd Service:"
echo "    sudo systemctl start vcm.service"
echo "=================================================================="
