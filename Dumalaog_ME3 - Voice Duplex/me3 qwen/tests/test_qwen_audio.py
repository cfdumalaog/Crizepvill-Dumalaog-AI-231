from __future__ import annotations

import io
import sys
import wave
from types import SimpleNamespace

import numpy as np

from me3 import qwen_audio


def _wav_bytes(sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.zeros(sample_rate // 10, dtype=np.int16).tobytes())
    return buf.getvalue()


def test_qwen_asr_uses_vllm_multimodal_chat_request(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "Magandang araw"}}]},
        )

    monkeypatch.setattr(qwen_audio.requests, "post", fake_post)
    asr = qwen_audio.QwenRemoteASR({
        "base_url": "http://audio.test/v1",
        "model": "Qwen/Qwen2-Audio-7B-Instruct",
        "api_key": "test-only",
        "protocol": "chat_completions",
        "timeout_s": 2,
    })
    segments, info = asr.transcribe(np.zeros(1600, dtype=np.float32))
    assert segments[0].text == "Magandang araw"
    assert info.duration == 0.1
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["model"] == "Qwen/Qwen2-Audio-7B-Instruct"
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["type"] == "audio_url"
    assert content[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert content[1]["text"].startswith("Transcribe the spoken audio")
    assert captured["headers"]["Authorization"] == "Bearer test-only"


def test_qwen_asr_can_use_legacy_transcription_protocol(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"text": "legacy transcript"},
        )

    monkeypatch.setattr(qwen_audio.requests, "post", fake_post)
    asr = qwen_audio.QwenRemoteASR({
        "base_url": "http://audio.test/v1",
        "model": "m",
        "api_key": "test-only",
        "protocol": "transcriptions",
    })
    segments, _ = asr.transcribe(np.zeros(1600, dtype=np.float32))
    assert segments[0].text == "legacy transcript"
    assert captured["url"].endswith("/audio/transcriptions")
    assert captured["data"]["model"] == "m"
    assert "file" in captured["files"]


def test_qwen_tts_accepts_pcm_and_reports_sample_rate(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            content=(np.zeros(2400, dtype=np.int16)).tobytes(),
            headers={"Content-Type": "audio/pcm"},
        )

    monkeypatch.setattr(qwen_audio.requests, "post", fake_post)
    tts = qwen_audio.QwenRemoteTTS({
        "base_url": "http://tts.test/v1",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "api_key": "test-only",
        "voice": "aiden",
        "language": "Filipino",
        "instruct": "formal professional delivery",
        "sample_rate": 24000,
        "timeout_s": 2,
    })
    chunks = list(tts.synthesize("Magandang araw"))
    assert len(chunks) == 1
    assert len(chunks[0].audio_int16_bytes) == 4800
    assert tts.config.sample_rate == 24000
    assert captured["url"].endswith("/audio/speech")
    assert captured["json"]["voice"] == "aiden"
    assert captured["json"]["language"] == "Filipino"
    assert captured["json"]["instruct"] == "formal professional delivery"
    assert captured["headers"]["Content-Type"] == "application/json"


def test_qwen_audio_requires_runtime_configuration():
    asr = qwen_audio.QwenRemoteASR({"model": "m"})
    try:
        asr.transcribe(np.zeros(10, dtype=np.float32))
    except RuntimeError as exc:
        assert "missing base URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing Qwen endpoint should fail closed")


def test_qwen_asr_uses_local_faster_whisper_fallback_when_remote_is_down(monkeypatch):
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            assert model == "base"
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, pcm, **kwargs):
            assert len(pcm) == 1600
            return [SimpleNamespace(text="fallback transcript")], SimpleNamespace(
                duration=0.1
            )

    monkeypatch.setitem(
        sys.modules, "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr(
        qwen_audio.requests, "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            qwen_audio.requests.ConnectionError("ASR port unavailable")
        ),
    )
    asr = qwen_audio.QwenRemoteASR({
        "base_url": "http://audio.test/v1",
        "model": "Qwen/Qwen2-Audio-7B-Instruct",
        "api_key": "test-only",
        "protocol": "chat_completions",
        "fallback_engine": "faster_whisper",
        "fallback_model": "base",
    })
    segments, info = asr.transcribe(np.zeros(1600, dtype=np.float32))
    assert segments[0].text == "fallback transcript"
    assert info.duration == 0.1
    assert asr.last_backend == "faster_whisper_fallback"
