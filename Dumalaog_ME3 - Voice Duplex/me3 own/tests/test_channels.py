from __future__ import annotations

from types import SimpleNamespace

import pytest

from me3 import channels, tools


def _response(status: int, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, json=lambda: payload)


def _clear_channel_env(monkeypatch):
    for name in (
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
        "TWILIO_MESSAGING_SERVICE_SID", "TWILIO_VOICE_URL",
        "RESEND_API_KEY", "RESEND_FROM_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_channel_health_is_network_free_and_reports_only_presence(monkeypatch):
    _clear_channel_env(monkeypatch)
    result = channels.channel_health()
    assert result["ok"] is True
    assert result["side_effects_require_confirm"] is True
    assert result["channels"]["sms"]["configured"] is False
    assert result["channels"]["call"]["configured"] is False
    assert result["channels"]["email"]["configured"] is False
    assert "TWILIO_AUTH_TOKEN" in result["channels"]["sms"]["missing_env"]
    assert "secret" not in str(result).lower()


def test_call_health_requires_a_caller_id_even_when_sms_uses_messaging_service(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth-material-not-returned")
    monkeypatch.setenv("TWILIO_MESSAGING_SERVICE_SID", "MG-test")
    monkeypatch.setenv("TWILIO_VOICE_URL", "https://example.com/twilio/answer")
    result = channels.channel_health()
    assert result["channels"]["sms"]["configured"] is True
    assert result["channels"]["call"]["configured"] is False
    assert "TWILIO_FROM_NUMBER" in result["channels"]["call"]["missing_env"]


def test_external_side_effects_stop_before_network_without_confirmation():
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(201, {"sid": "SM-test"})

    assert channels.send_sms("+639171234567", "Hello", request_fn=post)["error"] == (
        "confirmation_required"
    )
    assert channels.make_call("+639171234567", request_fn=post)["error"] == (
        "confirmation_required"
    )
    assert channels.send_email(["customer@example.com"], "Subject", "Body",
                               request_fn=post)["error"] == "confirmation_required"
    assert calls == []


def test_twilio_sms_uses_form_data_and_returns_safe_metadata(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth-material-not-returned")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+639171111111")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(201, {"sid": "SM123"})

    result = channels.send_sms("+639171234567", "Kumusta", confirm=True,
                               request_fn=post)
    assert result == {"ok": True, "provider": "twilio", "status_code": 201,
                      "provider_id": "SM123", "channel": "sms"}
    args, kwargs = calls[0]
    assert args[0].endswith("/Accounts/AC-test/Messages.json")
    assert kwargs["data"] == {"To": "+639171234567", "Body": "Kumusta",
                               "From": "+639171111111"}
    assert kwargs["auth"] == ("AC-test", "auth-material-not-returned")
    assert "auth-material-not-returned" not in str(result)


def test_twilio_call_requires_https_webhook_and_uses_call_resource(monkeypatch):
    _clear_channel_env(monkeypatch)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC-test")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth-material-not-returned")
    monkeypatch.setenv("TWILIO_FROM_NUMBER", "+639171111111")
    calls = []

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return _response(201, {"sid": "CA123"})

    with pytest.raises(ValueError, match="HTTPS"):
        channels.make_call("+639171234567", confirm=True,
                           twiml_url="http://localhost/answer", request_fn=post)
    assert calls == []
    result = channels.make_call(
        "+639171234567", confirm=True,
        twiml_url="https://example.com/twilio/answer", request_fn=post,
    )
    assert result["ok"] is True and result["provider_id"] == "CA123"
    assert calls[0][0][0].endswith("/Accounts/AC-test/Calls.json")
    assert calls[0][1]["data"] == {
        "To": "+639171234567", "From": "+639171111111",
        "Url": "https://example.com/twilio/answer",
    }


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


def test_channel_validation_rejects_non_e164_and_invalid_email():
    with pytest.raises(ValueError, match="E.164"):
        channels.send_sms("09171234567", "hello")
    with pytest.raises(ValueError, match="email"):
        channels.send_email(["not-an-email"], "Subject", "Body")


def test_tool_registry_exposes_shared_channel_tools():
    names = {item["function"]["name"] for item in tools.tool_definitions()}
    assert {"channel_health", "send_sms", "make_call", "send_email"} <= names
    assert {"channel_health", "send_sms", "make_call", "send_email"} <= set(tools.health()["tools"])
