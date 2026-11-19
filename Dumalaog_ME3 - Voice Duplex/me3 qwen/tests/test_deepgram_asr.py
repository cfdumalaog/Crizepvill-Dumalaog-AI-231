from __future__ import annotations

from types import SimpleNamespace

from me3 import deepgram_asr, qwen_audio


def test_deepgram_key_validation_uses_token_header_without_echoing_secret(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(deepgram_asr.requests, "get", fake_get)
    result = deepgram_asr.validate_api_key("local-test-key")

    assert result["ok"] is True
    assert result["provider"] == "deepgram"
    assert result["model"] == "nova-3"
    assert seen["url"].endswith("/auth/token")
    assert seen["headers"]["Authorization"] == "Token local-test-key"
    assert "local-test-key" not in str(result)


def test_deepgram_multilingual_transcription_uses_english_and_tagalog_hints(monkeypatch):
    seen = {}
    monkeypatch.setattr(deepgram_asr.secure_config, "get",
                        lambda name: "local-test-key")

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": {"channels": [{"alternatives": [
                {"transcript": "Magandang umaga, good morning."}
            ]}]}},
        )

    monkeypatch.setattr(deepgram_asr.requests, "post", fake_post)
    transcript, latency = deepgram_asr.transcribe_wav(b"test-wav")

    assert transcript == "Magandang umaga, good morning."
    assert latency >= 0
    assert seen["url"].endswith("/listen")
    assert seen["params"] == [
        ("model", "nova-3"), ("language", "multi"),
        ("language_hint", "en"), ("language_hint", "tl"),
        ("smart_format", "true"),
    ]
    assert seen["headers"]["Authorization"] == "Token local-test-key"


def test_qwen_failure_uses_deepgram_before_local_whisper(monkeypatch):
    monkeypatch.setattr(qwen_audio.requests, "post", lambda *a, **k: (_ for _ in ()).throw(
        qwen_audio.requests.ConnectionError("ASR port unavailable")))
    monkeypatch.setattr(deepgram_asr.secure_config, "configured", lambda _name: True)
    monkeypatch.setattr(deepgram_asr, "transcribe_wav",
                        lambda raw, timeout_s: ("backup transcript", 25.0))
    asr = qwen_audio.QwenRemoteASR({
        "base_url": "http://audio.test/v1",
        "model": "Qwen/Qwen2-Audio-7B-Instruct",
        "api_key": "local-test-key",
        "fallback_engine": "faster_whisper",
        "fallback_model": "base",
    })

    segments, info = asr.transcribe(qwen_audio.np.zeros(1600, dtype="float32"))

    assert segments[0].text == "backup transcript"
    assert info.duration == 0.1
    assert asr.last_backend == "deepgram_fallback"
