"""Google Calendar event creation for the ME3 voice agent.

Uses an OAuth refresh token owned by the operator.  Google credentials are
read only from the Pi service environment; token bodies and attendee details
are never returned in error messages or logs.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import threading
import time
from urllib.parse import urlencode
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from . import secure_config

TOKEN_URL = "https://oauth2.googleapis.com/token"
CALENDAR_API = "https://www.googleapis.com/calendar/v3"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_LOCK = threading.Lock()
_TOKEN: tuple[str, float] | None = None
_OAUTH_LOCK = threading.Lock()
_OAUTH_PENDING: dict | None = None
_SCOPES = "https://www.googleapis.com/auth/calendar.events"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _credentials() -> tuple[str, str, str]:
    return tuple(secure_config.get(name) for name in (
        "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET",
        "GOOGLE_CALENDAR_REFRESH_TOKEN"))


def redirect_uri() -> str:
    return os.getenv("GOOGLE_CALENDAR_REDIRECT_URI", "http://127.0.0.1:8100/api/google/callback").strip()


def configure_client(client_id: str, client_secret: str) -> dict:
    client_id = str(client_id or "").strip()
    client_secret = str(client_secret or "").strip()
    if not client_id or not client_secret:
        return {"ok": False, "error": "google_oauth_client_id_and_secret_required"}
    if len(client_id) > 300 or len(client_secret) > 300:
        return {"ok": False, "error": "invalid_google_oauth_client_values"}
    return secure_config.set_values({
        "GOOGLE_OAUTH_CLIENT_ID": client_id,
        "GOOGLE_OAUTH_CLIENT_SECRET": client_secret,
    })


def begin_oauth() -> tuple[str | None, dict | None]:
    """Start OAuth with minimal event-edit access and state/PKCE protection."""
    global _OAUTH_PENDING
    client_id, client_secret, _refresh = _credentials()
    if not client_id or not client_secret:
        return None, {"ok": False, "error": "google_oauth_client_credentials_required"}
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(32)
    with _OAUTH_LOCK:
        _OAUTH_PENDING = {"state": state, "verifier": verifier,
                          "created": time.time()}
    params = {
        "client_id": client_id, "redirect_uri": redirect_uri(),
        "response_type": "code", "scope": _SCOPES,
        "access_type": "offline", "prompt": "consent",
        "include_granted_scopes": "true", "state": state,
        "code_challenge_method": "S256", "code_challenge": challenge,
    }
    return AUTH_URL + "?" + urlencode(params), None


def complete_oauth(code: str, state: str) -> dict:
    """Exchange an approved code and store its refresh token privately on Pi."""
    global _OAUTH_PENDING, _TOKEN
    with _OAUTH_LOCK:
        pending = _OAUTH_PENDING
        _OAUTH_PENDING = None
    if (not pending or not secrets.compare_digest(str(state or ""), pending["state"])
            or time.time() - float(pending["created"]) > 600):
        return {"ok": False, "error": "google_oauth_state_invalid_or_expired"}
    client_id, client_secret, _refresh = _credentials()
    try:
        response = requests.post(TOKEN_URL, data={
            "client_id": client_id, "client_secret": client_secret,
            "grant_type": "authorization_code", "code": str(code or ""),
            "redirect_uri": redirect_uri(), "code_verifier": pending["verifier"],
        }, timeout=15)
        if response.status_code != 200:
            return {"ok": False, "error": "google_oauth_exchange_failed",
                    "status_code": response.status_code}
        data = response.json()
        refresh = str(data.get("refresh_token") or "")
        access = str(data.get("access_token") or "")
        if not refresh or not access:
            return {"ok": False, "error": "google_oauth_refresh_token_missing"}
        saved = secure_config.set_values({"GOOGLE_CALENDAR_REFRESH_TOKEN": refresh})
        if not saved.get("ok"):
            return saved
        with _TOKEN_LOCK:
            _TOKEN = (access, time.monotonic() + max(60, int(data.get("expires_in", 3600))))
        return {"ok": True, "connected": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "google_oauth_request_failed",
                "error_type": type(exc).__name__}


def health() -> dict:
    values = _credentials()
    names = ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET",
             "GOOGLE_CALENDAR_REFRESH_TOKEN")
    missing = [name for name, value in zip(names, values) if not value]
    return {"provider": "google_calendar_api", "configured": not missing,
            "missing_env": missing,
            "client_configured": bool(values[0] and values[1]),
            "authorized": bool(values[2]), "redirect_uri": redirect_uri()}


def _access_token() -> tuple[str | None, dict | None]:
    global _TOKEN
    client_id, client_secret, refresh = _credentials()
    if not client_id or not client_secret or not refresh:
        return None, {"ok": False, "error": "google_calendar_not_configured",
                      "missing_env": health()["missing_env"]}
    now = time.monotonic()
    with _TOKEN_LOCK:
        if _TOKEN and _TOKEN[1] - 30 > now:
            return _TOKEN[0], None
        try:
            response = requests.post(TOKEN_URL, data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh,
                "grant_type": "refresh_token",
            }, timeout=12)
            if response.status_code != 200:
                return None, {"ok": False, "error": "google_calendar_auth_failed",
                              "status_code": response.status_code}
            data = response.json()
            token = str(data.get("access_token") or "")
            if not token:
                return None, {"ok": False, "error": "google_calendar_empty_token"}
            _TOKEN = (token, now + max(60, int(data.get("expires_in", 3600))))
            return token, None
        except Exception as exc:  # noqa: BLE001 - provider body may contain secrets
            return None, {"ok": False, "error": "google_calendar_auth_request_failed",
                          "error_type": type(exc).__name__}


def _datetime(value: str, timezone: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("start and end times are required")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("use an ISO date/time such as 2026-11-08T15:00:00") from exc
    if parsed.tzinfo is None:
        try:
            parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
    return parsed


def create_event(title: str, start: str, end: str, *,
                 attendees: list[str] | None = None, timezone: str = "Asia/Manila",
                 location: str = "", description: str = "",
                 create_meet: bool = True) -> dict:
    title = str(title or "").strip()
    if not title:
        return {"ok": False, "error": "meeting_title_required"}
    try:
        start_dt = _datetime(start, timezone)
        end_dt = _datetime(end, timezone)
    except ValueError as exc:
        return {"ok": False, "error": "invalid_meeting_time", "message": str(exc)}
    if end_dt <= start_dt:
        return {"ok": False, "error": "meeting_end_must_follow_start"}
    if (end_dt - start_dt).total_seconds() > 12 * 3600:
        return {"ok": False, "error": "meeting_duration_exceeds_12_hours"}
    guests = [str(item).strip() for item in (attendees or []) if str(item).strip()]
    if len(guests) > 50 or any(not _EMAIL.fullmatch(item) for item in guests):
        return {"ok": False, "error": "attendees_must_be_valid_emails_max_50"}
    token, error = _access_token()
    if error:
        return error
    tz_name = timezone
    event = {
        "summary": title[:200],
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": tz_name},
    }
    if guests:
        event["attendees"] = [{"email": email} for email in guests]
    if location.strip():
        event["location"] = location.strip()[:500]
    if description.strip():
        event["description"] = description.strip()[:4000]
    params = {"sendUpdates": "all" if guests else "none"}
    if create_meet:
        event["conferenceData"] = {"createRequest": {
            "requestId": uuid.uuid4().hex,
            "conferenceSolutionKey": {"type": "hangoutsMeet"},
        }}
        params["conferenceDataVersion"] = 1
    try:
        response = requests.post(
            f"{CALENDAR_API}/calendars/primary/events",
            headers={"Authorization": f"Bearer {token}"},
            params=params, json=event, timeout=20,
        )
        if response.status_code not in (200, 201):
            return {"ok": False, "error": "google_calendar_event_rejected",
                    "status_code": response.status_code}
        data = response.json()
        conference = data.get("conferenceData") or {}
        meet_url = data.get("hangoutLink") or next((
            entry.get("entryPoints", [{}])[0].get("uri")
            for entry in conference.get("entryPoints", [])
            if entry.get("entryPointType") == "video"), None)
        return {"ok": bool(data.get("id")), "provider": "Google Calendar",
                "event_id": data.get("id"), "title": data.get("summary"),
                "start": (data.get("start") or {}).get("dateTime"),
                "end": (data.get("end") or {}).get("dateTime"),
                "event_url": data.get("htmlLink"), "meet_url": meet_url,
                "invite_count": len(guests), "provider_created": bool(data.get("id"))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "google_calendar_request_failed",
                "error_type": type(exc).__name__}


def create_reminder_event(text: str, due: datetime) -> dict:
    """Write a reminder to Google Calendar; ME3 handles the spoken alert."""
    title = str(text or "").strip()
    if not title:
        return {"ok": False, "error": "reminder_text_required"}
    if due.tzinfo is None:
        due = due.astimezone()
    end = due.timestamp() + 60
    end_dt = datetime.fromtimestamp(end, due.tzinfo)
    token, error = _access_token()
    if error:
        return error
    event = {
        "summary": ("Reminder: " + title)[:200],
        "start": {"dateTime": due.isoformat()},
        "end": {"dateTime": end_dt.isoformat()},
        # ME3 provides the audible notification; avoid a duplicate Calendar
        # popup while still keeping the reminder on the user's calendar.
        "reminders": {"useDefault": False},
        "extendedProperties": {"private": {"me3Type": "reminder"}},
    }
    try:
        response = requests.post(
            f"{CALENDAR_API}/calendars/primary/events",
            headers={"Authorization": f"Bearer {token}"},
            params={"sendUpdates": "none"}, json=event, timeout=20,
        )
        if response.status_code not in (200, 201):
            return {"ok": False, "error": "google_calendar_reminder_rejected",
                    "status_code": response.status_code}
        data = response.json()
        return {"ok": bool(data.get("id")), "provider": "Google Calendar",
                "event_id": data.get("id"), "title": data.get("summary"),
                "start": (data.get("start") or {}).get("dateTime"),
                "event_url": data.get("htmlLink")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "google_calendar_reminder_request_failed",
                "error_type": type(exc).__name__}


def delete_event(event_id: str) -> dict:
    """Delete an ME3-created calendar event by its provider ID."""
    event_id = str(event_id or "").strip()
    if not event_id:
        return {"ok": False, "error": "google_calendar_event_id_required"}
    token, error = _access_token()
    if error:
        return error
    try:
        response = requests.delete(
            f"{CALENDAR_API}/calendars/primary/events/{event_id}",
            headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if response.status_code not in (200, 204, 404):
            return {"ok": False, "error": "google_calendar_delete_rejected",
                    "status_code": response.status_code}
        return {"ok": True, "deleted": response.status_code != 404,
                "already_missing": response.status_code == 404}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "google_calendar_delete_failed",
                "error_type": type(exc).__name__}
