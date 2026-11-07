from __future__ import annotations

from types import SimpleNamespace

import pytest

from me3 import channels, tools


def _response(status: int, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, json=lambda: payload)


def _clear_channel_env(monkeypatch):
    for name in (
        "RESEND_API_KEY", "RESEND_FROM_EMAIL", "TELEGRAM_BOT_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


def test_channel_health_is_network_free_and_redacts_credentials(monkeypatch):
    _clear_channel_env(monkeypatch)
    sentinel = "telegram-token-health-sentinel"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", sentinel)
    result = channels.channel_health()
    assert result["ok"] is True
    assert result["side_effects_require_confirm"] is True
    assert set(result["channels"]) == {
        "telegram", "email", "calendar", "music", "phone_calls",
    }
    assert result["channels"]["telegram"]["configured"] is True
    assert result["channels"]["email"]["configured"] is False
    assert "phone-provider" not in str(result).lower()
    assert sentinel not in str(result)


def test_external_side_effects_stop_before_network_without_confirmation():
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(200, {"ok": True, "result": {"message_id": 42}})

    assert channels.send_email(["customer@example.com"], "Subject", "Body",
                               request_fn=post)["error"] == "confirmation_required"
    assert channels.send_telegram_message("12345", "Hello",
                                         request_fn=post)["error"] == "confirmation_required"
    assert calls == []


def test_resend_email_uses_json_and_bounded_recipient_list(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("RESEND_API_KEY", "re-test-secret")
    monkeypatch.setenv("RESEND_FROM_EMAIL", "ME3 <agent@example.com>")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(200, {"id": "email_123"})

    result = channels.send_email(
        ["customer@example.com", "agent@example.com"], "Update", "Your ticket is ready.",
        confirm=True, request_fn=post,
    )
    assert result == {"ok": True, "provider": "resend", "status_code": 200,
                      "provider_id": "email_123", "channel": "email"}
    assert calls[0][0][0] == "https://api.resend.com/emails"
    assert calls[0][1]["json"] == {
        "from": "ME3 <agent@example.com>",
        "to": ["customer@example.com", "agent@example.com"],
        "subject": "Update", "text": "Your ticket is ready.",
    }
    assert calls[0][1]["headers"]["Authorization"] == "Bearer re-test-secret"
    assert "re-test-secret" not in str(result)


def test_channel_validation_rejects_invalid_email_and_chat_id():
    with pytest.raises(ValueError, match="email"):
        channels.send_email(["not-an-email"], "Subject", "Body")
    with pytest.raises(ValueError, match="chat_id"):
        channels.send_telegram_message("bad chat id", "hello")


def test_telegram_send_message_uses_bot_api_and_returns_safe_metadata(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token-not-returned")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(200, {"ok": True, "result": {"message_id": 42}})

    result = channels.send_telegram_message("-10012345", "Kumusta",
                                           confirm=True, request_fn=post)
    assert result == {"ok": True, "provider": "telegram", "status_code": 200,
                      "channel": "text", "provider_id": "42"}
    assert calls[0][0][0].endswith("/botbot-token-not-returned/sendMessage")
    assert calls[0][1]["json"] == {
        "chat_id": "-10012345", "text": "Kumusta",
        "disable_web_page_preview": True,
    }
    assert "bot-token-not-returned" not in str(result)


def test_channel_health_reports_telegram_text_and_call_boundary(monkeypatch):
    _clear_channel_env(monkeypatch)
    result = channels.channel_health()
    assert result["channels"]["telegram"]["configured"] is False
    assert result["channels"]["telegram"]["text"] == "sendMessage"
    assert "unsupported" in result["channels"]["telegram"]["voice_calls"]


def test_tool_registry_exposes_telegram_without_phone_tools():
    names = {item["function"]["name"] for item in tools.tool_definitions()}
    assert {"channel_health", "send_email", "send_telegram_message"} <= names
    assert "send_sms" not in names
    assert "make_call" not in names
    assert {"channel_health", "send_email", "send_telegram_message"} <= set(tools.health()["tools"])
