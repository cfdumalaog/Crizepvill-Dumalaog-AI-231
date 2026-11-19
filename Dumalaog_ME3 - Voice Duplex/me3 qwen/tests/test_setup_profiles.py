from __future__ import annotations

import json

from me3 import setup_profiles


def test_setup_plan_generates_vllm_fields():
    plan = setup_profiles.setup_plan(
        "use another vLLM endpoint for the tool model"
    )
    names = {field["name"] for field in plan["fields"]}
    assert plan["ok"] is True
    assert {"base_url", "model", "api_key", "protocol"}.issubset(names)
    assert next(field for field in plan["fields"] if field["name"] == "role")["default"] == "llm"


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
