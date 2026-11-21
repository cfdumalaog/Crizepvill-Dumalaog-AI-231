from __future__ import annotations

from types import SimpleNamespace

from me3.mcp_tts import _ProfileFallbackTTS


class FailingTTS:
    config = SimpleNamespace(sample_rate=16000)

    def synthesize(self, text):
        raise RuntimeError("provider timeout")


class PiperTTS:
    config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text):
        yield SimpleNamespace(audio_int16_bytes=b"\x01\x02")


def test_qwen_profile_falls_back_to_piper_before_first_audio_chunk():
    primary = FailingTTS()
    fallback = PiperTTS()
    engine = _ProfileFallbackTTS(primary, lambda: fallback)

    chunks = list(engine.synthesize("Good morning."))

    assert [c.audio_int16_bytes for c in chunks] == [b"\x01\x02"]
    assert engine.provider == "piper_fallback"
    assert engine.config.sample_rate == 22050
    assert engine.last_event["backend"] == "piper_fallback"
