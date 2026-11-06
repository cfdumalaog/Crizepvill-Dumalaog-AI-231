from __future__ import annotations

import json
import time

import numpy as np

from me3.web_audio import BrowserAudioHub, BrowserOutputAudio


class _ConnectedHub:
    def __init__(self):
        self.published = []
        self.cleared = 0

    def snapshot(self):
        return {"clients": 1}

    def publish(self, pcm):
        self.published.append(np.asarray(pcm, dtype=np.int16).copy())

    def clear(self):
        self.cleared += 1


def test_browser_audio_hub_publishes_pcm_and_clear_event():
    from websockets.sync.client import connect

    hub = BrowserAudioHub("127.0.0.1", 0)
    hub.start()
    try:
        with connect(f"ws://127.0.0.1:{hub.port}/audio", open_timeout=3) as client:
            # Let the server handler register the client before publishing.
            deadline = time.time() + 3
            while hub.snapshot()["clients"] < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert hub.snapshot()["clients"] == 1
            samples = np.array([0, 1200, -1200, 42], dtype=np.int16)
            hub.publish(samples)
            assert client.recv(timeout=3) == samples.astype("<i2").tobytes()
            hub.clear()
            assert json.loads(client.recv(timeout=3))["event"] == "clear"
            hub.publish_event({"event": "music_action", "provider": "YouTube Music"})
            assert json.loads(client.recv(timeout=3))["event"] == "music_action"
            assert hub.snapshot()["sent_samples"] == 4
    finally:
        hub.stop()


def test_browser_output_registers_speaker_reference_for_echo_guard():
    cfg = {
        "audio": {
            "sample_rate": 16000,
            "frame_ms": 30,
            "echo_guard": {
                "enabled": True,
                "delay_ms": 80,
                "search_ms": 250,
                "search_step_ms": 5,
                "correlation_threshold": 0.20,
                "hard_gate_correlation": 0.75,
                "min_reference_rms": 20,
                "tail_ms": 250,
            },
        },
    }
    hub = _ConnectedHub()
    audio = BrowserOutputAudio(cfg, hub)
    t = np.arange(16000, dtype=np.float32) / 16000.0
    speaker = (12000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    audio.play(speaker)
    assert audio.capture._play_ref is not None
    assert audio.capture._playing is True
    audio.capture._play_started_at = time.perf_counter() - 0.08
    cleaned = audio.capture._remove_speaker_echo(speaker[:480])
    assert audio.capture._echo_cancelled_frames == 1
    assert float(np.sqrt(np.mean(cleaned.astype(np.float32) ** 2))) < 100.0
    audio.stop_playback(silent=True)
    assert audio.capture._playing is False
    audio.close()
