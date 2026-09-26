#!/usr/bin/env python3
"""
Hands-Free, Zero-Button Always-Listening Voice Command Assistant.
Target: Windows Workstation & Raspberry Pi 5.
Uses sounddevice to continuously stream audio from the microphone.
No buttons required:
1. Speak "Hi Dandan" or "Hello Dandan" to wake up.
2. Hear the wake chime and see the status change to LISTENING.
3. Speak any smart device command (e.g., "play music", "turn on lights").
4. Assistant executes the action and returns to always-listening standby.
"""

import os
import sys
import time
import signal
from pathlib import Path

# Setup project root import
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import sounddevice as sd

from src.config import (
    SAMPLE_RATE,
    STREAM_HOP_SEC,
    CONFIDENCE_THRESHOLD,
    DETECTION_HOLDOFF_SEC,
    MODELS_DIR,
    WAKE_TIMEOUT_SEC,
    WAKE_CONFIDENCE
)
from src.model import build_model
from src.hal import get_hardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine

from src.mic_stream import RobustMicrophoneStreamer

# Sound chime on Windows
def play_system_chime(kind="wake"):
    try:
        if sys.platform == "win32":
            import winsound
            if kind == "wake":
                winsound.Beep(1200, 80)
                time.sleep(0.04)
                winsound.Beep(1600, 120)
            elif kind == "command":
                winsound.Beep(1000, 100)
            elif kind == "timeout":
                winsound.Beep(600, 150)
    except Exception:
        pass


def main():
    print("=" * 76)
    print("      TINY VOICE COMMAND ASSISTANT — HANDS-FREE ALWAYS-LISTENING")
    print("=" * 76)
    print("  Trigger Phrase : 'Hi Dandan' or 'Hello Dandan'")
    print("  Interaction    : 100% Hands-Free (Zero Buttons Required)")
    print("  Edge Target    : Raspberry Pi 5 & Windows PC")
    print("=" * 76)

    # 1. Initialize Hardware Abstraction Layer
    hw = get_hardware(verbose=False)
    controller = SmartDeviceController(hardware=hw)

    # 2. Load Model Weights
    int8_path = MODELS_DIR / "bc_resnet_int8.pt"
    best_path = MODELS_DIR / "bc_resnet_best.pt"

    model = build_model("bc_resnet")
    if int8_path.exists():
        print(f"[MODEL] Loading INT8 model from {int8_path.name}...")
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            quantized.load_state_dict(torch.load(int8_path, map_location="cpu"))
            model = quantized
        except Exception:
            model.load_state_dict(torch.load(best_path, map_location="cpu"))
    elif best_path.exists():
        print(f"[MODEL] Loading model from {best_path.name}...")
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    else:
        print("[ERROR] No trained model weights found! Train model first.")
        return

    model.eval()

    # 3. Initialize Streaming Engine with Wake-Word State Machine
    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=0.50,
        wake_confidence=0.48,
        wake_timeout_sec=WAKE_TIMEOUT_SEC,
        holdoff_seconds=0.35,
        device="cpu",
        require_wake_word=True
    )

    controller.reset_to_standby()

    # 4. Probe and Open Microphone Streamer
    def audio_callback(chunk_16k: np.ndarray):
        event = engine.feed_audio_chunk(chunk_16k)
        if event:
            evt = event.get("event")
            cmd = event.get("command", "")
            conf = event.get("confidence", 0.0) * 100
            lat = event.get("latency_ms", 0.0)
            msg = event.get("action_message", "")

            if evt == "COMPOUND_COMMAND_EXECUTED":
                print(f"\n\n[⚡ COMPOUND ONE-SHOT COMMAND] '{cmd}' ({conf:.1f}%) | Latency: {lat:.1f} ms")
                print(f"       -> Mode: Single-Breath Siri/Alexa One-Shot Utterance")
                print(f"       -> Action: {msg}")
                print(f"       -> 🔊 Voiced Output: \"{hw.get_state().get('last_speech', '')}\"")
                print(f"       -> Returning to STANDBY (Awaiting 'Hi Dandan')...\n")
                play_system_chime("command")
            elif evt == "WAKE_WORD_DETECTED":
                trigger_type = event.get("trigger", "KWS")
                print(f"\n\n[👋 WAKE WORD DETECTED] '{cmd}' ({conf:.1f}%) | Latency: {lat:.1f} ms | Trigger: {trigger_type}")
                print(f"       -> Assistant is AWAKE! Listening for command (Dynamic Endpointing Active)...")
                print(f"       -> 🔊 Voiced Output: \"{hw.get_state().get('last_speech', '')}\"\n")
                play_system_chime("wake")
            elif evt == "COMMAND_EXECUTED":
                trigger_type = event.get("trigger", "KWS")
                print(f"\n\n[✅ COMMAND EXECUTED] '{cmd}' ({conf:.1f}%) | Latency: {lat:.1f} ms | Trigger: {trigger_type}")
                print(f"       -> Action: {msg}")
                print(f"       -> 🔊 Voiced Output: \"{hw.get_state().get('last_speech', '')}\"")
                print(f"       -> Assistant returning to STANDBY (Awaiting 'Hi Dandan')...\n")
                play_system_chime("command")
            elif evt == "TIMEOUT":
                print(f"\n\n[⏱️ TIMEOUT] Inactivity window expired -> Returning to STANDBY...")
                print(f"       -> 🔊 Voiced Output: \"{hw.get_state().get('last_speech', '')}\"\n")
                play_system_chime("timeout")

    streamer = RobustMicrophoneStreamer(
        target_sample_rate=SAMPLE_RATE,
        hop_seconds=STREAM_HOP_SEC,
        callback=audio_callback,
        apply_agc=True
    )

    print(f"[MIC] Active Microphone: '{streamer.device_name}' (Native: {streamer.native_sample_rate} Hz, {streamer.channels} ch)")
    print(f"[MIC] Host API: {streamer.host_api_name}")
    print("[MIC] Continuous audio stream initialized at 16,000 Hz Mono.")
    print("\n>>> System is ALWAYS LISTENING. Just speak into the room: 'Hi Dandan' <<<")
    print("Press Ctrl+C to stop.\n")

    running = True
    def _sig_handler(sig, frame):
        nonlocal running
        print("\n[SHUTDOWN] Stopping always-listening engine...")
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        streamer.start()
        while running:
            # Display real-time VU meter
            vu_str, energy = streamer.get_vu_meter(num_bars=10)
            state_str = "STANDBY [Say 'Hi Dandan']" if engine.state == StreamingVCMEngine.STATE_STANDBY else "LISTENING [Speak Command now!]"
            print(f"\r  Mic: {vu_str} (Energy: {energy:.4f}) | State: {state_str}   ", end="", flush=True)

            to_evt = engine.check_timeout()
            if to_evt:
                print(f"\n\n[⏱️ TIMEOUT] Listening window expired -> Returning to STANDBY...\n")
                play_system_chime("timeout")
            time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[STREAM ERROR] {e}")
    finally:
        streamer.stop()

    hw.cleanup()
    print("\n[SYSTEM] Clean shutdown complete.")


if __name__ == "__main__":
    main()
