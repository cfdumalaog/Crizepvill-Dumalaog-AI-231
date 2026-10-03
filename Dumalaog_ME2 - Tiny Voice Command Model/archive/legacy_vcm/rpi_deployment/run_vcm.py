#!/usr/bin/env python3
"""
Standalone Edge Runner for Tiny Voice Command Model (VCM).
Target: Raspberry Pi 4 Model B / Raspberry Pi 5.
Continuously listens on USB/I2S microphone, runs sliding-window BC-ResNet inference,
and actuates RGB LED, Buzzer, I2C OLED display, and audio speaker in real time.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add parent directory to module search path
DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

import numpy as np
import torch

from src.config import (
    SAMPLE_RATE,
    STREAM_HOP_SEC,
    CONFIDENCE_THRESHOLD,
    DETECTION_HOLDOFF_SEC,
    MODELS_DIR,
    ASSETS_DIR
)
from src.model import build_model
from src.hal import get_hardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine


def main():
    print("=" * 72)
    print("      TINY VOICE COMMAND MODEL (VCM) - EDGE SMART NODE")
    print("=" * 72)
    print("Device Target: Raspberry Pi 4 / 5 (ARM64)")
    print("Model Architecture: BC-ResNet-1 (INT8 Quantized / Real-Time)")
    print("Capabilities: Standalone, 100% On-Device, Zero Cloud Calls, Zero LLMs\n")

    # 1. Initialize Hardware Abstraction Layer
    hw = get_hardware(verbose=True)
    controller = SmartDeviceController(hardware=hw)

    # 2. Load Model Weights
    int8_model_path = DEPLOY_DIR / "models" / "bc_resnet_int8.pt"
    fp32_model_path = DEPLOY_DIR / "models" / "bc_resnet_best.pt"

    model = build_model("bc_resnet")
    if int8_model_path.exists():
        print(f"[EDGE] Loading quantized INT8 weights from {int8_model_path.name}...")
        try:
            # Load dynamic quantized model
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            quantized_model.load_state_dict(torch.load(int8_model_path, map_location="cpu"))
            model = quantized_model
        except Exception as e:
            print(f"[EDGE] Fallback to standard weights ({e})")
            model.load_state_dict(torch.load(fp32_model_path, map_location="cpu"))
    elif fp32_model_path.exists():
        print(f"[EDGE] Loading FP32 weights from {fp32_model_path.name}...")
        model.load_state_dict(torch.load(fp32_model_path, map_location="cpu"))
    else:
        print(f"[ERROR] No model weights found in {DEPLOY_DIR / 'models'}!")
        return

    model.eval()

    # 3. Instantiate Real-Time Streaming Engine with Wake-Word Detection
    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        holdoff_seconds=DETECTION_HOLDOFF_SEC,
        device="cpu",
        require_wake_word=True
    )

    # 4. Graceful Shutdown Handler
    running = True

    def _sig_handler(sig, frame):
        nonlocal running
        print("\n[EDGE] Shutdown signal received. Cleaning up...")
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # 5. Connect Audio Input Stream
    hop_samples = int(SAMPLE_RATE * STREAM_HOP_SEC)
    use_live_mic = False

    try:
        import sounddevice as sd
        # Probe available input devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if len(input_devices) > 0:
            print(f"[EDGE] Audio input device detected: '{input_devices[0]['name']}'")
            use_live_mic = True
        else:
            print("[EDGE] No hardware microphone detected. Starting in standby mode.")
    except Exception as e:
        print(f"[EDGE] Notice: sounddevice library or audio input device not ready ({e}).")

    controller.reset_to_standby()
    time.sleep(0.5)

    if use_live_mic:
        print("\n[EDGE] >>> Microphone stream OPEN. Always listening for wake phrase ('Hi Dandan' / 'Hello Dandan')... <<<")
        print("Press Ctrl+C to terminate.\n")

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[AUDIO STREAM STATUS] {status}", file=sys.stderr)
            audio_chunk = indata[:, 0].copy()
            event = engine.feed_audio_chunk(audio_chunk)
            if event:
                evt_type = event.get("event")
                if evt_type == "WAKE_WORD_DETECTED":
                    print(f"[WAKE WORD DETECTED]: '{event['command']}' ({event['confidence']*100:.1f}%) | Latency: {event['latency_ms']:.1f} ms -> Listening...")
                elif evt_type == "COMMAND_EXECUTED":
                    print(f"[COMMAND EXECUTED]: '{event['command']}' ({event['confidence']*100:.1f}%) | Latency: {event['latency_ms']:.1f} ms -> {event['action_message']}")
                elif evt_type == "TIMEOUT":
                    print("[TIMEOUT]: Listening window expired -> Standby.")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=hop_samples,
                callback=audio_callback
            ):
                while running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[AUDIO ERROR] Live stream failed: {e}")
    else:
        print("\n[EDGE] Running in automated standby / loopback verification mode...")
        print("To feed speech without a mic, use simulation/test_all_commands.py or plug in a USB mic.")
        while running:
            time.sleep(1.0)

    hw.cleanup()
    print("[EDGE] System shut down cleanly.")


if __name__ == "__main__":
    main()
