"""
Interactive Virtual Smart Assistant Simulator.
Allows testing the complete edge smart speaker pipeline on a PC without physical hardware.
Supports feeding synthetic spoken commands or live PC microphone, rendering the OLED screen,
RGB LED PWM states, buzzer alerts, and audio responses in real time.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.config import (
    COMMAND_CLASSES,
    MODELS_DIR,
    SAMPLE_RATE,
    STREAM_HOP_SEC
)
from src.model import build_model
from src.dataset import AcousticCommandSynthesizer
from src.hal import get_hardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine


def run_interactive_simulator():
    print("\n" + "=" * 78)
    print("      VIRTUAL RASPBERRY PI SMART ASSISTANT - INTERACTIVE SIMULATOR")
    print("=" * 78)
    print("Running in Virtual Simulation Mode (Mock GPIO, Virtual OLED, Software Audio)")
    print("Zero physical hardware needed on PC — exact same code that runs on Raspberry Pi!\n")

    # 1. Hardware & Model Setup
    hw = get_hardware(force_virtual=True, verbose=True)
    controller = SmartDeviceController(hardware=hw)

    checkpoint_path = MODELS_DIR / "bc_resnet_best.pt"
    if not checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Train the model first.")
        return

    model = build_model("bc_resnet")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=0.65,
        holdoff_seconds=0.8,
        device="cpu"
    )

    synthesizer = AcousticCommandSynthesizer()

    # Available demo commands
    demo_commands = {
        "1": ("play_music", "Command 1: Play Music ('play music')"),
        "2": ("question_weather", "Command 2: Search/Weather ('what's the weather')"),
        "3": ("question_time", "Command 2: Search/Time ('what time is it')"),
        "4": ("lights_on", "Command 3: Control Lights ('turn on lights')"),
        "5": ("dim_lights_50", "Command 4: Dim Lights ('dim lights to 50 percent')"),
        "6": ("lights_off", "Command 3: Control Lights ('turn off lights')"),
        "7": ("timer_5min", "Command 5: Set a Timer ('set a timer for 5 minutes')"),
        "8": ("alarm_set", "Command 6: Set an Alarm ('set an alarm for 7 am')"),
        "9": ("temp_cooler", "Command 7: Thermostat Cooler ('make it cooler')"),
        "10": ("temp_warmer", "Command 7: Thermostat Warmer ('make it warmer')"),
        "11": ("volume_up", "Command 8: Media Volume Up ('volume up')"),
        "12": ("media_pause", "Command 8: Media Pause ('pause')"),
        "13": ("reminders_check", "Command 9: Check Reminders ('what are my reminders')"),
        "14": ("call_mom", "Command 10: Calls & Messaging ('call mom')"),
    }

    def _feed_command(cmd_key: str):
        if cmd_key not in demo_commands:
            print(f"Unknown option '{cmd_key}'")
            return

        cmd_name, desc = demo_commands[cmd_key]
        print(f"\n[SPEAKING INTO VIRTUAL MIC]: '{desc}'...")
        utterance = synthesizer.synthesize_utterance(cmd_name)
        silence_pad = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
        stream = np.concatenate([silence_pad, utterance, silence_pad])

        chunk_size = int(SAMPLE_RATE * STREAM_HOP_SEC)
        engine.ring_buffer.reset()
        for offset in range(0, len(stream), chunk_size):
            chunk = stream[offset:offset + chunk_size]
            engine.feed_audio_chunk(chunk)
            time.sleep(0.02)  # brief pacing

    print("\nSelect a command to simulate voice input into the microphone:")
    for k, (_, desc) in demo_commands.items():
        print(f"  [{k:>2}] {desc}")
    print("  [ A] Run Automatic Showcase (all 10 commands in sequence)")
    print("  [ Q] Quit Simulation\n")

    # If run in non-interactive environment (or automated pipeline), demonstrate 3 actions then exit
    if not sys.stdin.isatty():
        print("[Notice] Running automated non-interactive demonstration:")
        for test_key in ["1", "4", "2", "14"]:
            _feed_command(test_key)
            time.sleep(0.5)
        print("\nAutomated demonstration finished.")
        hw.cleanup()
        return

    while True:
        try:
            choice = input("\nEnter selection (1-14, A for all, Q to quit): ").strip().upper()
            if choice == "Q":
                break
            elif choice == "A":
                print("\n--- RUNNING ALL-COMMAND SHOWCASE ---")
                for k in demo_commands:
                    _feed_command(k)
                    time.sleep(1.0)
                print("\n--- SHOWCASE COMPLETE ---")
            elif choice in demo_commands:
                _feed_command(choice)
            else:
                print("Invalid selection. Enter 1-14, A, or Q.")
        except (KeyboardInterrupt, EOFError):
            break

    hw.cleanup()
    print("\nVirtual Simulation Terminated.")


if __name__ == "__main__":
    run_interactive_simulator()
