from __future__ import annotations

from types import SimpleNamespace

from me3.mcp_tts import _AutoFallbackTTS


class FailingTTS:
    config = SimpleNamespace(sample_rate=16000)

    def synthesize(self, text):
        raise RuntimeError("provider timeout")


class PiperTTS:
    config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text):
        from me3.elevenlabs_tts import AudioChunk
        yield AudioChunk(b"\x01\x02")


def test_auto_fallback_uses_piper_before_first_provider_chunk():
    primary = FailingTTS()
    fallback = PiperTTS()
    engine = _AutoFallbackTTS(primary, lambda: fallback)

    chunks = list(engine.synthesize("hello"))

    assert [c.audio_int16_bytes for c in chunks] == [b"\x01\x02"]
    assert engine.provider == "piper_fallback"
    assert engine.config.sample_rate == 22050
    assert engine.last_event["backend"] == "piper_fallback"


class PartialFailTTS:
    config = SimpleNamespace(sample_rate=16000)

    def synthesize(self, text):
        from me3.elevenlabs_tts import AudioChunk
        yield AudioChunk(b"\x01\x02")
        raise RuntimeError("mid-stream failure")


def test_auto_fallback_does_not_duplicate_partial_provider_audio():
    engine = _AutoFallbackTTS(PartialFailTTS(), PiperTTS)

    try:
        list(engine.synthesize("hello"))
    except RuntimeError as exc:
        assert str(exc) == "mid-stream failure"
    else:
        raise AssertionError("mid-stream provider failure must be surfaced")

    assert engine.last_event["fallback"] == "not_safe_after_partial_audio"
