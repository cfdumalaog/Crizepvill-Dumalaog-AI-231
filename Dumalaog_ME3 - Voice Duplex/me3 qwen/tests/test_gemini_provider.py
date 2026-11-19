from __future__ import annotations

from types import SimpleNamespace

from me3.brain import ToolBrain
from me3 import layer_choice, provider_choice, setup_profiles
from me3.gemini_brain import defaults


def test_gemini_brain_is_private_and_visible_as_a_layer(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    key = "AIza" + "test-only-gemini-key-" + "x" * 24
    assert provider_choice.save_gemini_key(key)["ok"]
    assert provider_choice.choose("gemini")["ok"]
    status = provider_choice.status()
    assert status["selected"] == "gemini"
    assert status["gemini_key_configured"]
    assert key not in str(status)

    layers = layer_choice.status()
    assert layers["layers"]["brain"] == "gemini"
    assert "gemini" in layers["options"]["brain"]
    assert "custom_vllm" not in layers["options"]["brain"]
    assert layers["option_labels"]["gemini"] == "Gemini text/tool brain"

    profile = setup_profiles.save_profile({
        "profile_name": "Class vLLM",
        "role": "llm",
        "base_url": "https://example.invalid/v1",
        "model": "class/model",
        "api_key": "profile-test-key-12345678901234567890",
        "protocol": "chat_completions",
        "fallback": "none",
    })
    assert profile["ok"]
    refreshed = layer_choice.status()
    assert "profile:class-vllm" in refreshed["options"]["brain"]
    assert "Class vLLM" in refreshed["option_labels"]["profile:class-vllm"]


def test_gemini_adapter_uses_function_tools_and_fast_mode(monkeypatch):
    cfg = defaults()
    cfg.update({"api_key": "test-gemini-key", "stream": False})
    brain = ToolBrain(cfg, "You are ME3.")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        )

    monkeypatch.setattr("me3.brain.requests.post", fake_post)
    message, _usage = brain._request(
        [{"role": "user", "content": "hello"}], include_tools=True)
    assert message["content"] == "Hello"
    assert calls[0][0] == defaults()["base_url"] + "/chat/completions"
    body = calls[0][1]["json"]
    assert body["tools"]
    assert body["tool_choice"] == "auto"
    assert body["reasoning_effort"] == "none"
