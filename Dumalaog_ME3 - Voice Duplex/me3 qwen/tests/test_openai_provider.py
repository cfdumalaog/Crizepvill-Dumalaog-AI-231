from __future__ import annotations

import io
import json
import urllib.request
import wave
from types import SimpleNamespace

import numpy as np

from me3 import mcp_asr, mcp_tts, provider_choice, secure_config
from me3.brain import build_tool_brain, build_system_prompt
from me3.config import load_config
from me3.openai_audio import OpenAIASR, OpenAITTS
from me3.agent_status import LoopStatus, StatusServer


def test_provider_selection_is_private_and_composes_all_three_models(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.delenv("ME3_PROVIDER", raising=False)
    monkeypatch.setenv("ME3_VARIANT", "qwen_instruct")
    assert provider_choice.choose("openai")["error"] == "openai_api_key_required"
    result = provider_choice.save_openai_key("sk-test-only-not-a-real-key")
    assert result["ok"]
    assert provider_choice.choose("openai")["ok"]
    assert provider_choice.selected() == "openai"
    assert provider_choice.status()["openai_key_configured"]
    assert "sk-" not in str(provider_choice.status())
    load_config.cache_clear()
    mcp_asr._engine = mcp_asr._cfg = None
    mcp_tts._engine = mcp_tts._cfg = None
    try:
        cfg = load_config()
        assert cfg["asr"]["model"] == "gpt-transcribe"
        assert cfg["tts"]["model"] == "gpt-4o-mini-tts"
        assert cfg["tts"]["voice"] == cfg["tts"]["announcer_voice"] == "onyx"
        assert cfg["llm"]["openai"]["model"] == "gpt-4.1-mini"
        assert mcp_asr._get_engine()[0].__class__ is OpenAIASR
        assert mcp_tts._get_engine()[0].__class__ is OpenAITTS
        brain = build_tool_brain(cfg, "openai", build_system_prompt("You are ME3."))
        assert brain.model == "gpt-4.1-mini"
        assert brain.allow_tools
        assert brain.api_key == secure_config.get("OPENAI_API_KEY")
    finally:
        load_config.cache_clear()
        mcp_asr._engine = mcp_asr._cfg = None
        mcp_tts._engine = mcp_tts._cfg = None
    assert provider_choice.choose("qwen")["ok"]


def test_qwen_key_is_private_and_overrides_stale_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setenv("VLLM_API_KEY", "old-environment-key")
    key = "sk-vllm-test-only-private-key"
    result = provider_choice.save_qwen_key(key)
    assert result["ok"]
    assert provider_choice.qwen_key() == key
    assert secure_config.get("VLLM_API_KEY") == key
    assert provider_choice.status()["qwen_key_configured"]
    assert key not in str(result) + str(provider_choice.status())


def test_openai_audio_uses_transcription_and_fixed_pcm_voice(monkeypatch):
    monkeypatch.setattr("me3.openai_audio._key", lambda: "test-key")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/audio/transcriptions"):
            assert kwargs["data"]["model"] == "gpt-transcribe"
            with wave.open(io.BytesIO(kwargs["files"]["file"][1]), "rb") as wav:
                assert wav.getframerate() == 16000
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"text": "Hello ME3"})
        assert kwargs["json"]["model"] == "gpt-4o-mini-tts"
        assert kwargs["json"]["voice"] == "onyx"
        assert kwargs["json"]["response_format"] == "pcm"
        return SimpleNamespace(raise_for_status=lambda: None, content=b"\0\0\1\0")

    monkeypatch.setattr("me3.openai_audio.requests.post", fake_post)
    segments, info = OpenAIASR().transcribe(np.zeros(16000, dtype=np.float32))
    assert segments[0].text == "Hello ME3"
    assert info.duration == 1.0
    assert list(OpenAITTS().synthesize("Hello"))[0].audio_int16_bytes == b"\0\0\1\0"
    assert len(calls) == 2


def test_private_provider_http_setup_and_status(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.delenv("ME3_PROVIDER", raising=False)
    server = StatusServer(
        LoopStatus(), html_dir=tmp_path, get_state=lambda: "idle",
        get_stats=lambda: SimpleNamespace(), host="127.0.0.1", port=0,
        provider_status=provider_choice.status,
        qwen_key_set=provider_choice.save_qwen_key,
        openai_key_set=provider_choice.save_openai_key,
        provider_select=lambda payload: provider_choice.choose(payload.get("provider")),
    )
    assert server.start()
    base = f"http://127.0.0.1:{server.port}"

    def post(path, body):
        request = urllib.request.Request(
            base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.read().decode()

    try:
        key = "sk-test-only-not-a-real-key"
        saved = post("/api/integrations/openai-key", {"api_key": key})
        chosen = post("/api/provider/select", {"provider": "openai"})
        with urllib.request.urlopen(base + "/api/provider/status", timeout=3) as response:
            status = response.read().decode()
        assert json.loads(chosen)["provider"] == "openai"
        assert json.loads(status)["openai_key_configured"]
        assert key not in saved + chosen + status
        qwen_key = "sk-vllm-test-only-private-key"
        qwen_saved = post("/api/integrations/qwen-key", {"api_key": qwen_key})
        qwen_chosen = post("/api/provider/select", {"provider": "qwen"})
        assert json.loads(qwen_chosen)["provider"] == "qwen"
        assert qwen_key not in qwen_saved + qwen_chosen
    finally:
        server.stop()


def test_openai_asr_accepts_mcp_file_formats(monkeypatch, tmp_path):
    source = tmp_path / "turn.mp3"
    source.write_bytes(b"test-only-audio-bytes")
    monkeypatch.setattr("me3.openai_audio._key", lambda: "test-key")

    def fake_post(_url, **kwargs):
        name, data, mime = kwargs["files"]["file"]
        assert name == "turn.mp3"
        assert data == b"test-only-audio-bytes"
        assert mime == "audio/mpeg"
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"text": "File transcription"})

    monkeypatch.setattr("me3.openai_audio.requests.post", fake_post)
    segments, _info = OpenAIASR().transcribe(source)
    assert segments[0].text == "File transcription"
