"""Credential-gated Telegram and email adapters for ME3.

The active ME3 communication path is Telegram Bot API text messaging. Email is
retained as an independent optional adapter; phone calls and SMS are deliberately
not part of this project because no phone-provider account is required or configured.
The adapters deliberately use small REST requests rather than provider SDKs so
the Pi deployment stays lightweight. They share one policy boundary:

* ``confirm=True`` is required before any network side effect;
* Telegram chat IDs and email recipients/message sizes are bounded;
* provider response bodies, headers, and credentials are never returned; and
* missing configuration is reported as names/status only.

The Telegram Bot API does not initiate ordinary voice calls. A future Telegram
voice-call feature would require a separately reviewed MTProto/TDLib user
account bridge and is intentionally outside the active demo.
"""
from __future__ import annotations

import os
import re
from email.utils import parseaddr
from typing import Any, Callable

import requests


RESEND_BASE_URL = "https://api.resend.com"
TELEGRAM_BASE_URL = "https://api.telegram.org"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
_TELEGRAM_CHAT = re.compile(r"^[A-Za-z0-9_@:-]{1,128}$")

RequestFn = Callable[..., Any]


def _present(name: str) -> bool:
    return bool(_env(name))


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    from . import secure_config
    return secure_config.get(name)


def _missing(names: list[str]) -> list[str]:
    return [name for name in names if not _present(name)]


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
    resend = _missing(["RESEND_API_KEY", "RESEND_FROM_EMAIL"])
    telegram = _missing(["TELEGRAM_BOT_TOKEN"])
    from . import google_calendar, me2_commands, spotify_player
    return {
        "ok": True,
        "channels": {
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
                "voice_calls": "unsupported_by_telegram_bot_api",
            },
            "calendar": google_calendar.health(),
            "music": spotify_player.health(),
            "phone_calls": me2_commands.phone_call_status(),
        },
        # `confirm` is derived from a clear user command or its direct answer
        # to a missing-detail question, not from a separate UI click.
        "side_effects_require_confirm": True,
        "side_effects_require_user_request": True,
    }


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
    """Send one Telegram Bot API text message after a clear user request.

    The Bot API supports text messages but does not expose ordinary Telegram
    voice-call initiation; this adapter deliberately does not pretend that
    ``sendMessage`` is a call.
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

