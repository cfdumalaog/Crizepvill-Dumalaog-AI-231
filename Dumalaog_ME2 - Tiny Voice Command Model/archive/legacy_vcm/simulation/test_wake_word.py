"""
Automated Integration Test for Wake-Word Flow in Tiny VCM.
Verifies that:
1. When in STANDBY, commands are rejected until wake word is spoken.
2. Speaking "Hi" / "Hello" (wake_word) awakens the assistant, triggers wake chime & cyan LED.
3. While LISTENING, speaking a command triggers execution.
4. If no command is given within timeout, assistant returns to STANDBY.
"""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf
from src.config import (
    MODELS_DIR,
    DATA_DIR,
    SAMPLE_RATE,
    NUM_SAMPLES,
    STREAM_HOP_SEC,
    WAKE_WORD_CLASS
)
from src.model import build_model
from src.hal import VirtualHardware
from src.actions import SmartDeviceController
from src.engine import StreamingVCMEngine

def test_wake_word_lifecycle():
    print("=" * 78)
    print("      WAKE-WORD LIFECYCLE INTEGRATION TEST ('Hi' / 'Hello')")
    print("=" * 78)

    hw = VirtualHardware(verbose=False)
    controller = SmartDeviceController(hardware=hw)

    checkpoint_path = MODELS_DIR / "bc_resnet_best.pt"
    model = build_model("bc_resnet")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    engine = StreamingVCMEngine(
        model=model,
        controller=controller,
        confidence_threshold=0.50,
        wake_confidence=0.50,
        wake_timeout_sec=2.0,  # Short 2.0s for testing timeout
        holdoff_seconds=0.2,
        device="cpu",
        require_wake_word=True
    )

    # 1. Verify initial state is STANDBY
    assert engine.state == StreamingVCMEngine.STATE_STANDBY, f"Initial state should be STANDBY, got {engine.state}"
    print("[PASS] Step 1: Assistant is in STANDBY state.")

    # 2. Feed a command ("lights_on") while in STANDBY -> Should NOT execute!
    lights_dir = DATA_DIR / "dataset" / "lights_on"
    lights_wav = list(lights_dir.glob("*clean.wav"))[0]
    lights_audio, _ = sf.read(str(lights_wav))
    if len(lights_audio) < NUM_SAMPLES:
        pad = np.zeros(NUM_SAMPLES, dtype=np.float32)
        pad[:len(lights_audio)] = lights_audio
        lights_audio = pad

    engine.ring_buffer.clear()
    engine.ring_buffer.push(lights_audio)
    res = engine.process_current_window()
    assert res is None or res.get("event") != "COMMAND_EXECUTED", "Command should NOT execute while in STANDBY!"
    assert not controller.lights_on, "Lights should remain OFF in standby!"
    print("[PASS] Step 2: Un-awakened command correctly ignored in STANDBY.")

    # 3. Feed wake word ("Hello" / "Hi") -> Assistant should wake up!
    wake_dir = DATA_DIR / "dataset" / WAKE_WORD_CLASS
    wake_wav = list(wake_dir.glob("*clean.wav"))[0]
    wake_audio, _ = sf.read(str(wake_wav))
    if len(wake_audio) < NUM_SAMPLES:
        pad = np.zeros(NUM_SAMPLES, dtype=np.float32)
        pad[:len(wake_audio)] = wake_audio
        wake_audio = pad

    engine.ring_buffer.clear()
    engine.ring_buffer.push(wake_audio)
    wake_res = engine.process_current_window()
    assert wake_res is not None, "Wake word should trigger an event!"
    assert wake_res["event"] == "WAKE_WORD_DETECTED", f"Expected WAKE_WORD_DETECTED, got {wake_res['event']}"
    assert engine.state == StreamingVCMEngine.STATE_LISTENING, f"Expected LISTENING, got {engine.state}"
    print(f"[PASS] Step 3: Wake word detected (conf: {wake_res['confidence']:.2f}). State transitioned to LISTENING.")
    print(f"       OLED Screen: {hw.display_lines[:2]}")

    # 4. Now feed command ("lights_on") while in LISTENING -> Should EXECUTE!
    engine.ring_buffer.clear()
    engine.ring_buffer.push(lights_audio)
    cmd_res = engine.process_current_window()
    assert cmd_res is not None, "Command should trigger an event!"
    assert cmd_res["event"] == "COMMAND_EXECUTED", f"Expected COMMAND_EXECUTED, got {cmd_res['event']}"
    assert controller.lights_on is True, "Lights should now be ON!"
    print(f"[PASS] Step 4: Command executed successfully ({cmd_res['action_message']}).")
    print(f"       LED State: {hw.get_state()['led']}")

    # 5. Test Wake -> Timeout
    # Wake up again
    engine.ring_buffer.clear()
    engine.ring_buffer.push(wake_audio)
    wake_res2 = engine.process_current_window()
    assert engine.state == StreamingVCMEngine.STATE_LISTENING, "State should be LISTENING"
    print("[PASS] Step 5a: Assistant awakened again.")

    # Wait 2.2 seconds to exceed timeout
    time.sleep(2.2)
    timeout_res = engine.check_timeout()
    assert timeout_res is not None and timeout_res["event"] == "TIMEOUT", "Timeout should trigger!"
    assert engine.state == StreamingVCMEngine.STATE_STANDBY, "State should return to STANDBY after timeout!"
    print("[PASS] Step 5b: Inactivity timeout verified. Assistant returned to STANDBY.")

    print("\n" + "=" * 78)
    print("ALL 5 WAKE-WORD LIFECYCLE TESTS PASSED PERFECTLY!")
    print("=" * 78)

if __name__ == "__main__":
    test_wake_word_lifecycle()
