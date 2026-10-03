"""
Automated Integration Test for Tiny Voice Command Model (VCM).
Synthesizes test audio streams for all 10 smart device commands,
feeds them sequentially through the StreamingVCMEngine, and verifies
that each physical/virtual actuator and OLED screen state updates correctly.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf
from src.config import (
    COMMAND_CLASSES,
    MODELS_DIR,
    DATA_DIR,
    SAMPLE_RATE,
    NUM_SAMPLES,
    STREAM_HOP_SEC
)
from src.model import build_model
from src.hal import VirtualHardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine


def run_all_commands_test():
    print("=" * 78)
    print("      TINY VOICE COMMAND MODEL (VCM) - END-TO-END INTEGRATION TEST")
    print("=" * 78)

    # 1. Initialize Virtual Hardware and Smart Device Controller
    hw = VirtualHardware(verbose=False)
    controller = SmartDeviceController(hardware=hw)

    # 2. Load Trained BC-ResNet Model Checkpoint
    checkpoint_path = MODELS_DIR / "bc_resnet_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")

    model = build_model("bc_resnet")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # 3. Instantiate Streaming Engine
    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=0.50,
        holdoff_seconds=0.4,
        device="cpu",
        require_wake_word=False
    )

    # Define test suite covering all 10 smart device command categories
    test_cases = [
        # (Command Name, Category Description, Verification Lambda)
        (
            "play_music",
            "Command #1: Play Music",
            lambda s: controller.media_playing is True and "Track 1" in s["display"][1]
        ),
        (
            "question_weather",
            "Command #2: Ask Question / Weather",
            lambda s: "WEATHER REPORT" in s["display"][0]
        ),
        (
            "question_time",
            "Command #2: Ask Question / Time",
            lambda s: "CURRENT TIME" in s["display"][0]
        ),
        (
            "lights_on",
            "Command #3: Control Lights / Turn ON",
            lambda s: s["led"]["on"] is True and s["led"]["brightness"] > 0.9
        ),
        (
            "dim_lights_50",
            "Command #4: Dim Lights to 50%",
            lambda s: s["led"]["on"] is True and 0.45 <= s["led"]["brightness"] <= 0.55
        ),
        (
            "lights_off",
            "Command #3: Control Lights / Turn OFF",
            lambda s: s["led"]["on"] is False
        ),
        (
            "timer_5min",
            "Command #5: Set a Timer (5 min)",
            lambda s: "TIMER STARTED" in s["display"][0]
        ),
        (
            "alarm_set",
            "Command #6: Set an Alarm",
            lambda s: "ALARM SET" in s["display"][0]
        ),
        (
            "temp_cooler",
            "Command #7: Thermostat (Cooler)",
            lambda s: "COOLING" in s["display"][2] and controller.temperature < 72
        ),
        (
            "temp_warmer",
            "Command #7: Thermostat (Warmer)",
            lambda s: "HEATING" in s["display"][2]
        ),
        (
            "media_pause",
            "Command #8: Media Control (Pause)",
            lambda s: controller.media_playing is False and "PAUSED" in s["display"][0]
        ),
        (
            "volume_up",
            "Command #8: Media Control (Volume Up)",
            lambda s: controller.media_volume > 70 and "VOLUME UP" in s["display"][0]
        ),
        (
            "reminders_check",
            "Command #9: Reminders and Lists",
            lambda s: "YOUR REMINDERS" in s["display"][0]
        ),
        (
            "call_mom",
            "Command #10: Calls and Messaging",
            lambda s: "Calling Mom" in s["display"][1] and s["led"]["g"] > 0.8
        )
    ]

    print(f"[TEST] Starting evaluation of {len(test_cases)} distinct command scenarios...\n")
    passed_count = 0
    failed_cases = []

    # Hop chunk size in samples for sliding window (0.25 sec = 4,000 samples)
    chunk_size = int(SAMPLE_RATE * STREAM_HOP_SEC)

    for i, (cmd_name, desc, verify_fn) in enumerate(test_cases, 1):
        print(f"Test {i:02d}/{len(test_cases):02d} | {desc} ('{cmd_name}')...")

        # Load real recorded audio stream for this command
        cls_dir = DATA_DIR / "dataset" / cmd_name
        clean_files = list(cls_dir.glob("*clean.wav")) if cls_dir.exists() else []
        wav_files = clean_files if clean_files else (list(cls_dir.glob("*.wav")) if cls_dir.exists() else [])
        if wav_files:
            utterance, _ = sf.read(str(wav_files[0]))
            if utterance.ndim > 1:
                utterance = np.mean(utterance, axis=1)
        else:
            utterance = np.zeros(NUM_SAMPLES, dtype=np.float32)

        # Pad with 0.3s silence before and after to simulate live speech stream
        silence_pad = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
        stream_audio = np.concatenate([silence_pad, utterance, silence_pad])

        # Reset ring buffer and feed audio in chunks through streaming engine
        engine.ring_buffer.reset()
        engine.last_detected_command = None
        detected_event = None

        for offset in range(0, len(stream_audio), chunk_size):
            chunk = stream_audio[offset:offset + chunk_size]
            event = engine.feed_audio_chunk(chunk)
            if event is not None:
                detected_event = event

        state = hw.get_state()
        is_verified = verify_fn(state)

        if detected_event and detected_event["command"] == cmd_name and is_verified:
            print(f"  [PASS] Detected '{detected_event['command']}' (conf: {detected_event['confidence']:.2f}, latency: {detected_event['latency_ms']:.2f} ms)")
            print(f"         Actuator verified: {detected_event['action_message']}")
            print(f"         OLED Screen: {state['display'][:2]}\n")
            passed_count += 1
        else:
            print(f"  [FAIL] Failed detection or state verification!")
            print(f"         Detected: {detected_event}")
            print(f"         State: {state}\n")
            failed_cases.append(cmd_name)

        time.sleep(0.05)

    # 4. Negative Test: Verify background noise & silence do NOT trigger false positives
    print("Negative Test: Feeding 3.0 seconds of background noise + silence...")
    engine.ring_buffer.reset()
    noise_dir = DATA_DIR / "dataset" / "_background_noise_"
    noise_files = list(noise_dir.glob("*.wav")) if noise_dir.exists() else []
    if noise_files:
        noise_stream, _ = sf.read(str(noise_files[0]))
        if len(noise_stream) < SAMPLE_RATE * 3:
            noise_stream = np.tile(noise_stream, 2)
    else:
        noise_stream = np.random.normal(0, 0.02, SAMPLE_RATE * 3).astype(np.float32)

    triggered_false_alarm = False
    for offset in range(0, len(noise_stream), chunk_size):
        chunk = noise_stream[offset:offset + chunk_size]
        event = engine.feed_audio_chunk(chunk)
        if event is not None:
            triggered_false_alarm = True
            break

    if not triggered_false_alarm:
        print("  [PASS] Noise rejection verified (zero false alarms).\n")
        passed_count += 1
    else:
        print("  [FAIL] Background noise caused false trigger!\n")
        failed_cases.append("noise_rejection")

    # Final Summary
    total_tests = len(test_cases) + 1
    print("=" * 78)
    print(f"TEST RESULTS: {passed_count}/{total_tests} Tests Passed ({(passed_count/total_tests)*100:.1f}%)")
    latency_stats = engine.get_latency_profile()
    print(f"Inference Latency Profile: Mean = {latency_stats['mean_ms']:.2f} ms | p95 = {latency_stats['p95_ms']:.2f} ms")
    print("=" * 78)

    hw.cleanup()
    return len(failed_cases) == 0


if __name__ == "__main__":
    success = run_all_commands_test()
    sys.exit(0 if success else 1)
