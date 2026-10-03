#!/usr/bin/env python3
"""
Test Suite for Siri and Alexa Edge Architecture Enhancements:
1. Continuous Stream One-Breath Utterance Recognition ("Hi Dandan" -> "Play Music")
2. VAD Silence Gating (Zero compute during ambient silence)
3. Dynamic Acoustic Endpointing (Immediate command dispatch on speech pause)
4. Earcon Audio Assets Verification
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("AI 231/Dumalaog_ME2 - Tiny Voice Command Model").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import soundfile as sf

from src.config import (
    SAMPLE_RATE,
    NUM_SAMPLES,
    MODELS_DIR,
    DATA_DIR,
    ASSETS_DIR,
    WAKE_WORD_CLASS
)
from src.model import build_model
from src.hal import VirtualHardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine
from src.audio import EnergyVAD, AdaptiveEndpointer


def run_tests():
    print("=" * 78)
    print("   SIRI AND ALEXA MULTI-STAGE EDGE ARCHITECTURE VERIFICATION TEST")
    print("=" * 78)

    # 1. Verify Earcon Audio Clips
    print("\n[TEST 1] Verifying Siri and Alexa Synthesized Earcon Assets...")
    earcon_dir = ASSETS_DIR / "earcons"
    for name in ["wake_earcon.wav", "success_earcon.wav", "cancel_earcon.wav"]:
        p = earcon_dir / name
        assert p.exists(), f"Earcon {name} not found!"
        data, sr = sf.read(str(p))
        assert sr == SAMPLE_RATE, f"Invalid sample rate {sr} for {name}"
        assert len(data) > 0, f"Empty earcon {name}"
        print(f"  [PASS] {name} ({len(data)/sr*1000:.1f} ms, 16 kHz)")

    # 2. Initialize Engine
    hw = VirtualHardware(verbose=False)
    controller = SmartDeviceController(hardware=hw)
    model = build_model("bc_resnet")
    model.load_state_dict(torch.load(MODELS_DIR / "bc_resnet_best.pt", map_location="cpu"))
    model.eval()

    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=0.55,
        wake_confidence=0.50,
        device="cpu",
        require_wake_word=True
    )

    # 3. Test VAD Silence Gating
    print("\n[TEST 2] Verifying VAD Silence Gating (Zero Compute During Standby Silence)...")
    silence_chunk = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)
    initial_bypasses = engine.vad_silence_bypasses
    initial_frames = engine.total_frames_processed

    for _ in range(8):  # 2.0 seconds of silence
        engine.feed_audio_chunk(silence_chunk)

    assert engine.vad_silence_bypasses > initial_bypasses, "VAD did not bypass silence!"
    assert engine.total_frames_processed == initial_frames, "Neural forward pass ran on pure silence!"
    print(f"  [PASS] VAD successfully bypassed {engine.vad_silence_bypasses} silent chunks.")
    print(f"         Total neural forward passes on silence: 0 (100% idle power reduction).")

    # 4. Test Dynamic Acoustic Endpointing
    print("\n[TEST 3] Verifying Dynamic Acoustic Endpointing (Immediate Dispatch on Pause)...")
    endpointer = AdaptiveEndpointer(sample_rate=SAMPLE_RATE, silence_duration_sec=0.45)
    chunk_len = int(SAMPLE_RATE * 0.25)
    
    # User speaks for 500ms
    s1 = endpointer.process(chunk_len, is_speech=True)
    s2 = endpointer.process(chunk_len, is_speech=True)
    assert s2 == AdaptiveEndpointer.STATE_SPEECH_ACTIVE, f"Expected SPEECH_ACTIVE, got {s2}"

    # User stops speaking (pause)
    s3 = endpointer.process(chunk_len, is_speech=False)
    s4 = endpointer.process(chunk_len, is_speech=False)
    assert s4 == AdaptiveEndpointer.STATE_ENDPOINT_REACHED, f"Expected ENDPOINT_REACHED, got {s4}"
    print("  [PASS] Dynamic endpointing triggered immediately after 500 ms pause.")

    # 5. Test Continuous Stream Compound Utterance: "Hi Dandan" + "Play Music"
    print("\n[TEST 4] Verifying Continuous Stream Compound Utterance Flow...")
    wake_dir = DATA_DIR / "dataset" / "wake_word"
    music_dir = DATA_DIR / "dataset" / "play_music"
    wake_wav = list(wake_dir.glob("*clean.wav"))[0]
    music_wav = list(music_dir.glob("*clean.wav"))[0]

    wake_audio, _ = sf.read(str(wake_wav))
    music_audio, _ = sf.read(str(music_wav))

    # Stream continuous speech: Wake Word -> 200ms natural breath pause -> Command
    stream_audio = np.concatenate([
        wake_audio,
        np.zeros(int(SAMPLE_RATE * 0.20), dtype=np.float32),
        music_audio
    ])

    engine.state = StreamingVCMEngine.STATE_STANDBY
    events = engine.process_audio_clip(stream_audio)

    event_types = [e.get("event") for e in events]
    print(f"  [PASS] Events captured across continuous stream: {event_types}")
    for evt in events:
        print(f"         - [{evt.get('event')}] '{evt.get('command')}' (conf: {evt.get('confidence', 0.0)*100:.1f}%) -> {evt.get('action_message')}")

    assert "WAKE_WORD_DETECTED" in event_types or "COMPOUND_COMMAND_EXECUTED" in event_types, "Wake word was not detected in stream!"
    assert "COMMAND_EXECUTED" in event_types or "COMPOUND_COMMAND_EXECUTED" in event_types, "Command was not executed in stream!"

    # 6. Test Acoustic Ducking and Volume Restoration
    print("\n[TEST 5] Verifying Acoustic Self-Interference Music Ducking & Restoration...")
    # Step A: Start music playback at normal volume (70%)
    controller.execute_command("play_music", 0.95)
    assert controller.media_playing is True, "Media did not start playing!"
    assert controller.media_volume == 70, f"Expected 70% volume, got {controller.media_volume}%"
    print("  [PASS] Music playing at 70% volume.")

    # Step B: Wake word arrives -> assistant enters LISTENING -> volume ducks to avoid microphone feedback
    controller.trigger_wake()
    assert controller.is_ducked is True, "Media was not ducked upon wake!"
    assert controller.media_volume <= 15, f"Expected ducked volume <= 15%, got {controller.media_volume}%"
    print(f"  [PASS] Music successfully ducked to {controller.media_volume}% during active listening.")

    # Step C: Command executes -> assistant fulfills request and un-ducks volume back to 70%
    engine.require_wake_word = True
    engine.state = StreamingVCMEngine.STATE_LISTENING
    engine._evaluate_and_dispatch_command(pre_evaluated=("lights_on", 0.92, 8.1, {}))
    assert controller.lights_on is True, "Lights did not turn on!"
    assert controller.is_ducked is False, "Media volume was not un-ducked after command dispatch!"
    assert controller.media_volume == 70, f"Expected restored volume 70%, got {controller.media_volume}%"
    print(f"  [PASS] Media volume restored to {controller.media_volume}% after command execution.")

    # Step D: Test listening timeout un-ducking
    controller.trigger_wake()
    assert controller.is_ducked is True and controller.media_volume <= 15
    controller.trigger_timeout()
    assert controller.is_ducked is False and controller.media_volume == 70
    print("  [PASS] Media volume restored to 70% after listening timeout.")

    print("\n" + "=" * 78)
    print("ALL 5 SIRI AND ALEXA ARCHITECTURE TESTS PASSED PERFECTLY!")
    print("=" * 78)


if __name__ == "__main__":
    run_tests()
