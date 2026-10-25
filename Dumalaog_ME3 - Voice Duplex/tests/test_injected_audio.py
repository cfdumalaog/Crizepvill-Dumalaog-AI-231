from __future__ import annotations

import time

import numpy as np

from me3.audio_io import InjectedDuplexAudio, RemoteMicAudio


def test_injected_audio_delivers_realtime_frames_without_portaudio():
    audio = InjectedDuplexAudio(sample_rate=1000, frame_ms=100)
    pcm = np.arange(250, dtype=np.int16)
    started = audio.inject(pcm, label="primary")
    first = audio.read_frame(timeout=0.5)
    second = audio.read_frame(timeout=0.5)
    assert started <= time.perf_counter()
    assert first is not None and len(first) == 100
    assert second is not None and len(second) == 100
    assert audio.inject_events[0]["label"] == "primary"
    audio.close()


def test_injected_audio_virtual_sink_records_and_stops_output():
    audio = InjectedDuplexAudio(sample_rate=1000, frame_ms=100)
    audio.play(np.ones(500, dtype=np.int16))
    assert audio.playback_done() is False
    assert audio.output_events[0]["sink"] == "virtual_discard"
    audio.stop_playback()
    assert audio.playback_done() is True
    assert audio.output_events[0]["stopped_abs"] is not None
    audio.close()


def test_remote_mic_command_uses_raw_pcm_without_secret_material():
    audio = RemoteMicAudio(
        config={"audio": {"sample_rate": 16000, "frame_ms": 30}},
        ssh_host="dalmacio@cfdfnjrpi5.local",
        remote_device="hw:2,0",
    )
    command = audio.remote_command()
    assert command[:3] == ["ssh", "-T", "dalmacio@cfdfnjrpi5.local"]
    assert "arecord" in command
    assert command[command.index("-D") + 1] == "hw:2,0"
    assert command[command.index("-c") + 1] == "2"
    assert command[command.index("-t") + 1] == "raw"
    assert all("key" not in item.lower() for item in command)
    audio.close()
