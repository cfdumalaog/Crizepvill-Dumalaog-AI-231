from __future__ import annotations

from pathlib import Path

from me3.config import load_config


def test_qwen_instruct_variant_composes_audio_models_without_changing_llm(monkeypatch):
    monkeypatch.setenv("ME3_VARIANT", "qwen_instruct")
    monkeypatch.setenv("QWEN_INSTRUCT_ASR_BASE_URL", "http://asr.test/v1")
    monkeypatch.setenv("QWEN_INSTRUCT_TTS_BASE_URL", "http://tts.test/v1")
    monkeypatch.setenv("QWEN_INSTRUCT_API_KEY", "test-only")
    load_config.cache_clear()
    cfg = load_config()
    assert cfg["runtime_variant"] == "qwen_instruct"
    assert cfg["asr"]["engine"] == "qwen_remote"
    assert cfg["asr"]["model"] == "Qwen/Qwen2-Audio-7B-Instruct"
    assert cfg["tts"]["engine"] == "qwen_remote"
    assert cfg["tts"]["model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert cfg["tts"]["announcer_voice"] == cfg["tts"]["voice"] == "aiden"
    assert "formal, neutral, professional" in cfg["tts"]["instruct"]
    assert cfg["llm"]["remote"]["model"]
    load_config.cache_clear()


def test_invalid_audio_variant_fails_closed(monkeypatch):
    monkeypatch.setenv("ME3_VARIANT", "not-a-variant")
    load_config.cache_clear()
    try:
        load_config()
    except ValueError as exc:
        assert "ME3_VARIANT" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid audio variant should fail closed")
    finally:
        load_config.cache_clear()


def test_announcer_voice_covers_backchannel_and_completion():
    """Keep the two asynchronous spoken paths on the standard Qwen voice."""
    source = (Path(__file__).resolve().parents[1] / "me3" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "tts_engine.voice = self.announcer_voice" in source
    assert source.count('voice_role="announcer"') >= 2


def test_spoken_tone_is_formal_and_professional(monkeypatch):
    monkeypatch.delenv("ME3_VARIANT", raising=False)
    load_config.cache_clear()
    cfg = load_config()
    assert cfg["agent"]["backchannel"] == "I am processing your request now."
    prompt = cfg["agent"]["system_prompt"].lower()
    assert "formal, professional customer-service tone" in prompt
    assert "do not flirt" in prompt
    assert "pet names" in prompt
    load_config.cache_clear()


def test_qwen_ui_uses_the_final_audio_model_label():
    source = (Path(__file__).resolve().parents[1] / "web" / "assistant.html").read_text(
        encoding="utf-8"
    )
    assert "ME3 QWEN INSTRUCT" in source
    assert "ME3 OWN" not in source
    assert "ME3 FASTER-WHISPER + PIPER" not in source
