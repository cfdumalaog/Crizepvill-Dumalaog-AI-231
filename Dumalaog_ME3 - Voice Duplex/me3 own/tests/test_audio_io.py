from __future__ import annotations

import time

import numpy as np

from me3.audio_io import FullDuplexAudio
from me3.config import load_config


def _audio_with_reference() -> tuple[FullDuplexAudio, np.ndarray]:
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
                "digital_loopback_hard_mute": True,
                "min_reference_rms": 20,
                "tail_ms": 250,
            },
        },
    }
    audio = FullDuplexAudio(cfg)
    t = np.arange(16000, dtype=np.float32) / 16000.0
    reference = (12000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    with audio._play_lock:
        audio._play_ref = reference
        audio._play_started_at = time.perf_counter() - 0.08
        audio._playing = True
        audio._echo_release_until = time.perf_counter() + 0.25
    return audio, reference


def test_speaker_reference_is_removed_before_vad():
    audio, reference = _audio_with_reference()
    frame = reference[:480]
    cleaned = audio._remove_speaker_echo(frame)

    assert audio._echo_cancelled_frames == 1
    assert float(np.sqrt(np.mean(cleaned.astype(np.float32) ** 2))) < 100.0
    audio.close()


def test_unmatched_user_speech_survives_echo_cancellation():
    audio, reference = _audio_with_reference()
    t = np.arange(480, dtype=np.float32) / 16000.0
    user = (9000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    mixed = (0.4 * reference[:480].astype(np.float32) + user).astype(np.int16)
    cleaned = audio._remove_speaker_echo(mixed)

    # The correlated speaker component is cancelled, while the orthogonal
    # user barge-in remains available to the VAD/ASR stage.
    user_rms = float(np.sqrt(np.mean(user.astype(np.float32) ** 2)))
    cleaned_rms = float(np.sqrt(np.mean(cleaned.astype(np.float32) ** 2)))
    assert audio._echo_cancelled_frames == 1
    assert cleaned_rms > user_rms * 0.75
    audio.close()


def test_digital_loopback_is_muted_only_while_agent_output_is_active():
    audio, reference = _audio_with_reference()
    audio._digital_loopback = True
    assert np.all(audio._remove_speaker_echo(reference[:480]) == 0)
    audio._playing = False
    audio._echo_release_until = time.perf_counter() - 1.0
    restored = audio._remove_speaker_echo(reference[:480])
    assert np.array_equal(restored, reference[:480])
    audio.close()


def test_philippine_profile_uses_multilingual_asr_and_echo_guard():
    cfg = load_config()
    if cfg.get("runtime_variant") == "qwen_instruct":
        assert cfg["asr"]["model"] == "Qwen/Qwen2-Audio-7B-Instruct"
    else:
        assert cfg["asr"]["model"] in {"tiny", "base", "small", "medium"}
    assert cfg["asr"]["language"] is None
    assert cfg["audio"]["echo_guard"]["enabled"] is True
