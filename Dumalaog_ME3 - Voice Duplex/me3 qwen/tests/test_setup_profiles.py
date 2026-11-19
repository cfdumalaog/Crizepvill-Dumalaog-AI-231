from __future__ import annotations

import json

from me3 import setup_profiles
from me3 import tools


def test_setup_plan_generates_vllm_fields():
    plan = setup_profiles.setup_plan(
        "use another vLLM endpoint for the tool model"
    )
    names = {field["name"] for field in plan["fields"]}
    assert plan["ok"] is True
    assert {"base_url", "model", "api_key", "protocol"}.issubset(names)
    assert next(field for field in plan["fields"] if field["name"] == "role")["default"] == "llm"


def test_toolbrain_setup_form_generates_fields_from_confirmed_request():
    result = setup_profiles.setup_form("configure Gemini as the tool brain")
    assert result["ok"] is True
    form = result["setup_form"]
    assert form["form_kind"] == "gemini"
    assert [field["name"] for field in form["fields"]] == ["api_key"]
    assert form["private_save"]["endpoint"] == "/api/integrations/gemini-key"


def test_toolbrain_setup_form_generates_custom_provider_fields():
    result = setup_profiles.setup_form(
        "use this OpenAI-compatible vLLM endpoint for the tool model"
    )
    assert result["ok"] is True
    form = result["setup_form"]
    assert form["form_kind"] == "custom_vllm"
    names = {field["name"] for field in form["fields"]}
    assert {"base_url", "model", "api_key", "protocol"}.issubset(names)


def test_setup_form_is_registered_as_a_tool():
    result = tools.dispatch("setup_form", {"request": "set up Spotify music"})
    assert result["ok"] is True
    assert result["setup_form"]["form_kind"] == "spotify"
    assert any(item["function"]["name"] == "setup_form"
               for item in tools.tool_definitions())


def test_custom_profile_is_private_and_status_is_redacted(monkeypatch, tmp_path):
    credential_path = tmp_path / "credentials.json"
    monkeypatch.setenv("ME3_CREDENTIALS_FILE", str(credential_path))
    result = setup_profiles.save_profile({
        "profile_name": "class-vllm",
        "role": "llm",
        "base_url": "https://models.example/v1",
        "model": "class/model",
        "api_key": "local-test-secret",
        "protocol": "chat_completions",
        "fallback": "none",
    })
    assert result["ok"] is True
    profile_id = result["activate_profile_id"]
    assert setup_profiles.activate_profile(profile_id)["ok"] is True
    stored = json.loads(setup_profiles._path().read_text(encoding="utf-8"))
    assert stored["profiles"][profile_id]["api_key"] == "local-test-secret"
    public = setup_profiles.status()
    encoded = json.dumps(public)
    assert "api_key" not in encoded
    assert public["active"]["llm"]["profile_id"] == profile_id
