from __future__ import annotations

import pytest

from me3.elevenlabs_tts import ElevenLabsTTS


class FakeResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        assert chunk_size == 4096
        yield from self.chunks

    def close(self):
        self.closed = True


def test_elevenlabs_stream_is_raw_pcm_and_closes_response():
    calls = []
    response = FakeResponse([b"\x01\x02", b"", b"\x03\x04"])

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    engine = ElevenLabsTTS(
        api_key="in-memory-test-only",
        voice_id="voice/test",
        request_fn=fake_post,
    )
    chunks = list(engine.synthesize("hello from ME3"))

    assert b"".join(c.audio_int16_bytes for c in chunks) == b"\x01\x02\x03\x04"
    assert response.closed is True
    assert calls[0][0][0].endswith("/text-to-speech/voice%2Ftest/stream")
    assert calls[0][1]["params"] == {
        "output_format": "pcm_16000",
        "enable_logging": "false",
    }
    assert calls[0][1]["json"] == {
        "text": "hello from ME3",
        "model_id": "eleven_flash_v2_5",
    }


def test_elevenlabs_requires_key_and_voice_id():
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        ElevenLabsTTS(api_key="", voice_id="voice")
    with pytest.raises(RuntimeError, match="ELEVENLABS_VOICE_ID"):
        ElevenLabsTTS(api_key="key", voice_id="")


def test_elevenlabs_config_reads_environment_without_logging(monkeypatch):
    monkeypatch.setenv("TEST_ELEVEN_KEY", "in-memory-test-only")
    monkeypatch.setenv("TEST_ELEVEN_VOICE", "voice123")
    engine = ElevenLabsTTS.from_config({
        "elevenlabs": {
            "api_key_env": "TEST_ELEVEN_KEY",
            "voice_id_env": "TEST_ELEVEN_VOICE",
        }
    })
    assert engine.voice_id == "voice123"
    assert engine.config.sample_rate == 16000
