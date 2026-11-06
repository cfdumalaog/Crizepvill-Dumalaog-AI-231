"""Credential-gated external communication adapters for ME3.

The voice agent can prepare or send an SMS, start an outbound call, or send a
transactional email.  The adapters deliberately use small REST requests rather
than provider SDKs so the Pi deployment stays lightweight.  They share one
policy boundary:

* ``confirm=True`` is required before any network side effect;
* phone numbers are E.164; email recipients and message sizes are bounded;
* provider response bodies, headers, and credentials are never returned; and
* missing configuration is reported as names/status only.

The call adapter starts an outbound Twilio call using a reviewed HTTPS TwiML
URL.  The URL must point to the operator's call-flow webhook; it is not a
complete media-stream bridge by itself.  That separation prevents an agent
turn from inventing call-control XML or silently exposing a local service.
"""
from __future__ import annotations

import os
import re
from email.utils import parseaddr
from typing import Any, Callable
from urllib.parse import urlparse

import requests


TWILIO_BASE_URL = "https://api.twilio.com/2010-04-01"
RESEND_BASE_URL = "https://api.resend.com"
TELEGRAM_BASE_URL = "https://api.telegram.org"
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_TELEGRAM_CHAT = re.compile(r"^[A-Za-z0-9_@:-]{1,128}$")

RequestFn = Callable[..., Any]


def _present(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _missing(names: list[str]) -> list[str]:
    return [name for name in names if not _present(name)]


def _phone(value: str, field: str) -> str:
    value = str(value or "").strip()
    if not _E164.fullmatch(value):
        raise ValueError(f"{field} must be an E.164 phone number such as +639171234567")
    return value


def _email(value: str, field: str) -> str:
    raw = str(value or "").strip()
    _, address = parseaddr(raw)
    if not _EMAIL.fullmatch(address):
        raise ValueError(f"{field} must be a valid email address")
    return raw


def _telegram_chat(value: str) -> str:
    raw = str(value or "").strip()
    if not _TELEGRAM_CHAT.fullmatch(raw):
        raise ValueError(
            "chat_id must be a Telegram numeric chat id or @channel username"
        )
    return raw


def _safe_id(value: Any) -> str | None:
    value = str(value or "").strip()
    return value if _SAFE_ID.fullmatch(value) else None


def _response_json(response: Any) -> dict:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def _result(provider: str, response: Any, id_key: str) -> dict:
    status_code = int(getattr(response, "status_code", 0) or 0)
    payload = _response_json(response)
    provider_id = _safe_id(payload.get(id_key))
    out = {"ok": 200 <= status_code < 300, "provider": provider,
           "status_code": status_code}
    if provider_id:
        out["provider_id"] = provider_id
    if not out["ok"]:
        # Never return provider text: it can contain request data or account
        # details.  The status code is enough for the operator-side audit.
        out["error"] = "provider_rejected_request"
    return out


def _request_error(provider: str, exc: Exception) -> dict:
    return {"ok": False, "provider": provider,
            "error": "provider_request_failed",
            "error_type": type(exc).__name__}


def channel_health() -> dict:
    """Report safe, network-free configuration status for all channels."""
    twilio_core = _missing(["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"])
    sms_sender = _present("TWILIO_FROM_NUMBER") or _present(
        "TWILIO_MESSAGING_SERVICE_SID")
    call_ready = _present("TWILIO_FROM_NUMBER") and _present("TWILIO_VOICE_URL")
    resend = _missing(["RESEND_API_KEY", "RESEND_FROM_EMAIL"])
    telegram = _missing(["TELEGRAM_BOT_TOKEN"])
    call_missing = twilio_core + ([] if _present("TWILIO_FROM_NUMBER") else [
        "TWILIO_FROM_NUMBER"]) + ([] if _present("TWILIO_VOICE_URL") else [
            "TWILIO_VOICE_URL"])
    return {
        "ok": True,
        "channels": {
            "sms": {
                "provider": "twilio",
                "configured": not twilio_core and sms_sender,
                "missing_env": twilio_core + ([] if sms_sender else [
                    "TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID"]),
            },
            "call": {
                "provider": "twilio",
                "configured": not twilio_core and call_ready,
                "missing_env": call_missing,
            },
            "email": {
                "provider": "resend",
                "configured": not resend,
                "missing_env": resend,
            },
            "telegram": {
                "provider": "telegram_bot_api",
                "configured": not telegram,
                "missing_env": telegram,
                "text": "sendMessage",
                "voice_calls": "unsupported_by_telegram_bot_api_use_twilio",
            },
        },
        "side_effects_require_confirm": True,
    }


def send_sms(to: str, body: str, *, confirm: bool = False,
             from_number: str | None = None,
             messaging_service_sid: str | None = None,
             request_fn: RequestFn | None = None) -> dict:
    """Send one SMS through Twilio after explicit confirmation."""
    to = _phone(to, "to")
    body = str(body or "").strip()
    if not body:
        raise ValueError("body is required")
    if len(body) > 3200:
        raise ValueError("body must be at most 3200 characters")
    if not confirm:
        return {"ok": False, "provider": "twilio", "channel": "sms",
                "error": "confirmation_required", "network_called": False}

    missing = _missing(["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"])
    sender = (messaging_service_sid or _env("TWILIO_MESSAGING_SERVICE_SID"))
    number = from_number or _env("TWILIO_FROM_NUMBER")
    if not sender and not number:
        missing.append("TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID")
    if missing:
        return {"ok": False, "provider": "twilio", "channel": "sms",
                "error": "configuration_missing", "missing_env": missing,
                "network_called": False}
    if number:
        number = _phone(number, "from_number")

    data = {"To": to, "Body": body}
    if sender:
        data["MessagingServiceSid"] = sender
    else:
        data["From"] = number
    url = f"{TWILIO_BASE_URL}/Accounts/{_env('TWILIO_ACCOUNT_SID')}/Messages.json"
    post = request_fn or requests.post
    try:
        response = post(url, data=data,
                        auth=(_env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")),
                        timeout=20)
        result = _result("twilio", response, "sid")
        result["channel"] = "sms"
        return result
    except Exception as exc:  # noqa: BLE001
        return _request_error("twilio", exc) | {"channel": "sms"}


def make_call(to: str, *, confirm: bool = False,
              twiml_url: str | None = None,
              from_number: str | None = None,
              request_fn: RequestFn | None = None) -> dict:
    """Start one outbound Twilio call using a reviewed HTTPS webhook URL."""
    to = _phone(to, "to")
    if not confirm:
        return {"ok": False, "provider": "twilio", "channel": "call",
                "error": "confirmation_required", "network_called": False}

    missing = _missing(["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"])
    number = from_number or _env("TWILIO_FROM_NUMBER")
    voice_url = twiml_url or _env("TWILIO_VOICE_URL")
    if not number:
        missing.append("TWILIO_FROM_NUMBER")
    if not voice_url:
        missing.append("TWILIO_VOICE_URL")
    if missing:
        return {"ok": False, "provider": "twilio", "channel": "call",
                "error": "configuration_missing", "missing_env": missing,
                "network_called": False}
    number = _phone(number, "from_number")
    parsed = urlparse(voice_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("twiml_url must be an absolute HTTPS URL")

    data = {"To": to, "From": number, "Url": voice_url}
    url = f"{TWILIO_BASE_URL}/Accounts/{_env('TWILIO_ACCOUNT_SID')}/Calls.json"
    post = request_fn or requests.post
    try:
        response = post(url, data=data,
                        auth=(_env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")),
                        timeout=20)
        result = _result("twilio", response, "sid")
        result["channel"] = "call"
        return result
    except Exception as exc:  # noqa: BLE001
        return _request_error("twilio", exc) | {"channel": "call"}


def send_email(to: str | list[str], subject: str, text: str, *,
               confirm: bool = False, from_email: str | None = None,
               request_fn: RequestFn | None = None) -> dict:
    """Send a bounded transactional email through Resend."""
    recipients = [to] if isinstance(to, str) else list(to or [])
    if not recipients or len(recipients) > 10:
        raise ValueError("to must contain between 1 and 10 recipients")
    recipients = [_email(value, "to") for value in recipients]
    subject = str(subject or "").strip()
    text = str(text or "").strip()
    if not subject or len(subject) > 200:
        raise ValueError("subject must contain 1-200 characters")
    if not text or len(text) > 20000:
        raise ValueError("text must contain 1-20000 characters")
    if not confirm:
        return {"ok": False, "provider": "resend", "channel": "email",
                "error": "confirmation_required", "network_called": False}

    missing = _missing(["RESEND_API_KEY", "RESEND_FROM_EMAIL"])
    sender = from_email or _env("RESEND_FROM_EMAIL")
    if missing:
        return {"ok": False, "provider": "resend", "channel": "email",
                "error": "configuration_missing", "missing_env": missing,
                "network_called": False}
    sender = _email(sender, "from_email")
    post = request_fn or requests.post
    try:
        response = post(f"{RESEND_BASE_URL}/emails",
                        headers={"Authorization": f"Bearer {_env('RESEND_API_KEY')}",
                                 "Content-Type": "application/json"},
                        json={"from": sender, "to": recipients,
                              "subject": subject, "text": text},
                        timeout=20)
        result = _result("resend", response, "id")
        result["channel"] = "email"
        return result
    except Exception as exc:  # noqa: BLE001
        return _request_error("resend", exc) | {"channel": "email"}


def send_telegram_message(chat_id: str, text: str, *,
                          confirm: bool = False,
                          bot_token: str | None = None,
                          request_fn: RequestFn | None = None) -> dict:
    """Send one Telegram Bot API text message after explicit confirmation.

    The Bot API supports text messages but does not expose ordinary Telegram
    voice-call initiation.  Phone calls remain a Twilio capability in ME3;
    this adapter deliberately does not pretend that ``sendMessage`` is a call.
    """
    chat_id = _telegram_chat(chat_id)
    text = str(text or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > 4096:
        raise ValueError("text must be at most 4096 characters")
    if not confirm:
        return {"ok": False, "provider": "telegram", "channel": "text",
                "error": "confirmation_required", "network_called": False}

    token = str(bot_token or _env("TELEGRAM_BOT_TOKEN")).strip()
    if not token:
        return {"ok": False, "provider": "telegram", "channel": "text",
                "error": "configuration_missing",
                "missing_env": ["TELEGRAM_BOT_TOKEN"],
                "network_called": False}
    post = request_fn or requests.post
    try:
        response = post(
            f"{TELEGRAM_BASE_URL}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text,
                  "disable_web_page_preview": True},
            timeout=20,
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        payload = _response_json(response)
        result = payload.get("result") if isinstance(payload, dict) else None
        message_id = _safe_id(result.get("message_id")) if isinstance(result, dict) else None
        ok = 200 <= status_code < 300 and payload.get("ok") is True
        out = {"ok": ok, "provider": "telegram", "status_code": status_code,
               "channel": "text"}
        if message_id:
            out["provider_id"] = message_id
        if not ok:
            out["error"] = "provider_rejected_request"
        return out
    except Exception as exc:  # noqa: BLE001
        return _request_error("telegram", exc) | {"channel": "text"}

