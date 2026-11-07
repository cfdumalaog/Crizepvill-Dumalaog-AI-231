"""ME2 command parity adapters for the model-driven ME3 agent.

ME3 does not train a replacement command classifier.  Qwen (or another
OpenAI-compatible model) selects these explicit tools from natural language,
while this module owns the deterministic action and safety boundary.  Local
state is always available for a demo; optional Home Assistant and channel
credentials turn the device adapters into live integrations.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

from . import channels, google_calendar, scheduled_actions, secure_config, spotify_player

_LOCK = threading.RLock()
_MEDIA = {
    "state": "stopped", "track": "", "volume": 50,
    "service": "spotify", "browser_url": "",
}
_MUSIC_EVENT_SINK = None
_DEVICE_ACTION_DISPATCHER = None
_LIGHTS: dict[str, dict] = {}
_THERMOSTATS: dict[str, float] = {}
_ALARMS: dict[str, dict] = {}
_REMINDERS: dict[str, dict] = {}
_SEQ = {"alarm": 0, "reminder": 0}


def simulated_light_state(room: str = "all") -> dict:
    """Return the persistent simulated bulb used by the ME3 demo UI.

    The state is deliberately kept in the same deterministic command adapter
    used by voice tool calls. A spoken command and a UI click therefore update
    the exact same device state. It is not a claim that a physical GPIO or
    Home Assistant light is connected.
    """
    room = str(room or "all").strip() or "all"
    with _LOCK:
        current = dict(_LIGHTS.get(room) or {})
    return {
        "id": "me3-demo-light-1",
        "name": "ME3 simulated light",
        "room": room,
        "on": bool(current.get("on", False)),
        "brightness": int(current.get("brightness", 100)),
        "color": str(current.get("color", "warm white")),
        "action": str(current.get("action", "off")),
        "value": current.get("value", current.get("brightness", 100)),
        "updated_at": current.get("updated_at"),
        "mode": "local_demo",
        "physical": False,
    }


def _next(kind: str) -> str:
    with _LOCK:
        _SEQ[kind] += 1
        return f"{kind[0]}{_SEQ[kind]}"


def _iso(value: dt.datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


def set_music_event_sink(sink) -> None:
    """Attach the Pi/browser event sink without importing the web server."""
    global _MUSIC_EVENT_SINK
    with _LOCK:
        _MUSIC_EVENT_SINK = sink if callable(sink) else None


def set_device_action_dispatcher(dispatcher) -> None:
    """Attach the active browser client as an execution device endpoint."""
    global _DEVICE_ACTION_DISPATCHER
    with _LOCK:
        _DEVICE_ACTION_DISPATCHER = dispatcher if callable(dispatcher) else None


def _store() -> scheduled_actions.ScheduledActionStore:
    return scheduled_actions.default_store()


def configure_schedule_delivery(on_due=None) -> scheduled_actions.ScheduledActionRunner:
    """Start the durable scheduler and route due items into voice output."""
    runner = scheduled_actions.ScheduledActionRunner(_store(), on_due=on_due)
    runner.start()
    return runner


def pending_notifications() -> dict:
    return {"ok": True, "notifications": _store().pending_notifications()}


def acknowledge_notification(action_id: str) -> dict:
    return _store().acknowledge(action_id)


def _emit_music(event: dict) -> None:
    with _LOCK:
        sink = _MUSIC_EVENT_SINK
    if sink is None:
        return
    try:
        sink({"event": "music_action", **event})
    except Exception:  # noqa: BLE001 - a disconnected browser must not fail a tool
        pass


def music_state() -> dict:
    """Read the actual Spotify player state from the connected account."""
    return spotify_player.state()


def play_music(query: str, service: str = "spotify") -> dict:
    """Find a track and start it on the user's active Spotify Connect device."""
    requested = str(service or "spotify").strip().lower().replace("-", "_")
    if requested in {"google", "google_music", "youtube", "youtube_music"}:
        return {"ok": False, "error": "provider_has_no_supported_autoplay_api",
                "provider": "YouTube Music",
                "note": "Use Spotify for autonomous playback; browser search handoff is disabled."}
    result = spotify_player.play(query, service="spotify")
    if result.get("ok"):
        with _LOCK:
            _MEDIA.update({"state": "playing", "track": result.get("track", query),
                           "service": "spotify", "browser_url": ""})
    return result


def media_control(action: str, value: int | None = None) -> dict:
    """Control the real active Spotify player, returning the provider result."""
    action = str(action or "").strip().lower().replace(" ", "_")
    action = {
        "louder": "volume_up", "increase_volume": "volume_up",
        "quieter": "volume_down", "decrease_volume": "volume_down",
        "next_track": "next", "next_song": "next",
    }.get(action, action)
    result = spotify_player.control(action, value)
    if result.get("ok"):
        with _LOCK:
            state = str((result.get("playback") or {}).get("playing", ""))
            if action in {"pause", "stop"}:
                _MEDIA["state"] = "paused" if action == "pause" else "stopped"
            elif action in {"play", "skip", "next"}:
                _MEDIA["state"] = "playing"
    return result


def get_weather(location: str) -> dict:
    """Return current weather from the no-key wttr.in JSON endpoint."""
    location = str(location or "").strip()
    if not location:
        return {"ok": False, "error": "location is required"}
    try:
        import requests
        response = requests.get(
            f"https://wttr.in/{quote(location)}?format=j1",
            headers={"User-Agent": "ME3-Voice-Agent/1.0"}, timeout=8)
        response.raise_for_status()
        data = response.json()
        current = (data.get("current_condition") or [{}])[0]
        desc = (current.get("weatherDesc") or [{"value": ""}])[0].get("value", "")
        return {"ok": True, "location": location,
                "temperature_c": current.get("temp_C"),
                "feels_like_c": current.get("FeelsLikeC"),
                "description": desc,
                "humidity": current.get("humidity"),
                "source": "wttr.in"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "location": location, "error": f"weather lookup failed: {exc}"}


def control_lights(action: str, room: str = "all", value: int | str | None = None) -> dict:
    """Control the simulated bulb; Home Assistant can replace it later."""
    action = str(action or "").strip().lower().replace(" ", "_")
    if action not in {"on", "off", "dim", "color"}:
        return {"ok": False, "error": "light action must be on, off, dim, or color"}
    room = str(room or "all").strip() or "all"
    if action == "dim":
        try:
            value = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            return {"ok": False, "error": "dim requires a percent from 0 to 100"}
    if action == "color" and not str(value or "").strip():
        return {"ok": False, "error": "color requires a color value"}
    with _LOCK:
        prior = simulated_light_state(room)
        state = {
            "action": action,
            "value": value,
            "on": prior["on"],
            "brightness": prior["brightness"],
            "color": prior["color"],
            "updated_at": _iso(dt.datetime.now().astimezone()),
        }
        if action == "on":
            state["on"] = True
        elif action == "off":
            state["on"] = False
        elif action == "dim":
            state["brightness"] = int(value)
            state["on"] = int(value) > 0
        elif action == "color":
            state["color"] = str(value).strip()
        _LIGHTS[room] = state
        device = simulated_light_state(room)
    return {"ok": True, "room": room, "lights": device,
            "mode": "local_demo", "live_integration": False,
            "note": "Simulated bulb updated. This project has no physical light controller connected."}


def set_thermostat(temperature_c: float, room: str = "home") -> dict:
    """Set demo thermostat state; a live climate adapter can be added via Home Assistant."""
    try:
        temperature_c = float(temperature_c)
    except (TypeError, ValueError):
        return {"ok": False, "error": "temperature_c must be numeric"}
    if not -10 <= temperature_c <= 45:
        return {"ok": False, "error": "temperature_c must be between -10 and 45"}
    room = str(room or "home").strip() or "home"
    with _LOCK:
        _THERMOSTATS[room] = temperature_c
    return {"ok": True, "room": room, "temperature_c": temperature_c,
            "mode": "local_demo", "live_integration": False,
            "note": "Simulated thermostat state updated; no physical climate controller is connected."}


def _parse_alarm_time(time_text: str) -> dt.datetime | None:
    raw = str(time_text or "").strip().lower()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", raw)
    if not match:
        return None
    hour, minute, meridian = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if minute > 59:
        return None
    if meridian:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridian == "pm" else 0)
    elif hour > 23:
        return None
    zone = ZoneInfo("Asia/Manila")
    now = dt.datetime.now(zone)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += dt.timedelta(days=1)
    return candidate


def set_alarm(time_text: str, label: str = "") -> dict:
    """Create a persistent next-occurrence alarm announced by ME3 when due."""
    when = _parse_alarm_time(time_text)
    if when is None:
        return {"ok": False, "error": "use an alarm time such as 7:30 AM or 19:30"}
    item = _store().create("alarm", {"label": str(label or "")}, when.timestamp())
    item["time"] = dt.datetime.fromtimestamp(item["due_at"], ZoneInfo("Asia/Manila")).isoformat(timespec="seconds")
    return {"ok": True, "alarm": item, "delivery": "persistent_me3_voice_and_ui"}


def list_alarms(include_cancelled: bool = False) -> dict:
    rows = _store().list("alarm", include_finished=include_cancelled)
    return {"ok": True, "alarms": rows}


def cancel_alarm(alarm_id: str) -> dict:
    return _store().finish(alarm_id, "cancelled")


def create_reminder(text: str, delay_seconds: float | None = None,
                    at_text: str = "") -> dict:
    """Create a reminder by delay or a local ISO/clock time."""
    text = str(text or "").strip()
    if not text:
        return {"ok": False, "error": "reminder text is required"}
    when = None
    if delay_seconds is not None:
        try:
            if float(delay_seconds) <= 0:
                raise ValueError
            when = dt.datetime.now().astimezone() + dt.timedelta(seconds=float(delay_seconds))
        except (TypeError, ValueError):
            return {"ok": False, "error": "delay_seconds must be positive"}
    elif at_text:
        try:
            when = dt.datetime.fromisoformat(str(at_text).strip())
            if when.tzinfo is None:
                when = when.replace(tzinfo=ZoneInfo("Asia/Manila"))
        except ValueError:
            when = _parse_alarm_time(at_text)
    if when is None:
        return {"ok": False, "error": "provide delay_seconds or at_text"}
    calendar_event = google_calendar.create_reminder_event(text, when)
    calendar_synced = bool(calendar_event.get("ok"))
    item = _store().create("reminder", {
        "text": text,
        "calendar_event_id": calendar_event.get("event_id") if calendar_synced else None,
        "calendar_event_url": calendar_event.get("event_url") if calendar_synced else None,
    }, when.timestamp())
    item["at"] = dt.datetime.fromtimestamp(item["due_at"], ZoneInfo("Asia/Manila")).isoformat(timespec="seconds")
    return {"ok": True, "reminder": item,
            "calendar_synced": calendar_synced,
            "calendar_sync_error": (None if calendar_synced else calendar_event.get("error")),
            "calendar_event_id": calendar_event.get("event_id") if calendar_synced else None,
            "calendar_event_url": calendar_event.get("event_url") if calendar_synced else None,
            "delivery": "persistent_me3_voice_and_ui",
            "note": ("Also added to Google Calendar." if calendar_synced else
                     "Local reminder is active; Google Calendar is not connected or did not confirm sync.")}


def list_reminders(include_completed: bool = False) -> dict:
    return {"ok": True, "reminders": _store().list(
        "reminder", include_finished=include_completed)}


def complete_reminder(reminder_id: str) -> dict:
    item = _store().get(reminder_id)
    if not item or item.get("kind") != "reminder":
        return {"ok": False, "error": "reminder_not_found"}
    event_id = str((item.get("payload") or {}).get("calendar_event_id") or "")
    if event_id:
        deleted = google_calendar.delete_event(event_id)
        if not deleted.get("ok"):
            return deleted
    return _store().finish(reminder_id, "completed")


def create_timer(seconds: float, label: str = "") -> dict:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return {"ok": False, "error": "seconds_must_be_numeric"}
    if seconds <= 0 or seconds > 30 * 24 * 3600:
        return {"ok": False, "error": "timer_must_be_between_0_and_30_days"}
    item = _store().create("timer", {"label": str(label or "")}, time.time() + seconds)
    return {"ok": True, "id": item["id"], "label": str(label or ""),
            "fires_at": dt.datetime.fromtimestamp(item["due_at"], dt.timezone.utc).isoformat(),
            "delivery": "persistent_me3_voice_and_ui"}


def list_scheduled_timers(include_finished: bool = False) -> dict:
    return {"ok": True, "timers": _store().list("timer", include_finished)}


def cancel_scheduled_timer(timer_id: str) -> dict:
    return _store().finish(timer_id, "cancelled")


def clock_schedule_snapshot() -> dict:
    rows = [item for item in _store().list(include_finished=False)
            if item.get("kind") in {"timer", "alarm"}]
    return {"ok": True, "now": time.time(), "items": rows}


def create_clock_action(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "schedule_payload_required"}
    kind = str(payload.get("kind") or "").strip().lower()
    if kind == "timer":
        try:
            seconds = float(payload.get("seconds"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "timer_seconds_required"}
        return create_timer(seconds, str(payload.get("label") or "web timer"))
    if kind == "alarm":
        return set_alarm(str(payload.get("time_text") or ""),
                         str(payload.get("label") or "ME3 alarm"))
    return {"ok": False, "error": "schedule_kind_must_be_timer_or_alarm"}


def cancel_clock_action(action_id: str) -> dict:
    item = _store().get(action_id)
    if not item or item.get("kind") not in {"timer", "alarm"}:
        return {"ok": False, "error": "clock_action_not_found"}
    return _store().finish(action_id, "cancelled")


def schedule_meeting(title: str, start: str, end: str,
                     attendees: list[str] | None = None,
                     timezone: str = "Asia/Manila", location: str = "",
                     description: str = "", create_meet: bool = True) -> dict:
    """Create a Google Calendar event and optional Google Meet conference."""
    return google_calendar.create_event(
        title, start, end, attendees=attendees, timezone=timezone,
        location=location, description=description, create_meet=create_meet)


def call_contact(contact: str, confirm: bool = False) -> dict:
    """Place a phone call through a connected ME3 Android companion device."""
    raw = str(contact or "").strip()
    if not raw:
        return {"ok": False, "error": "contact is required"}
    if not confirm:
        return {"ok": False, "error": "confirmation_required", "contact": raw,
                "network_called": False}
    with _LOCK:
        dispatcher = _DEVICE_ACTION_DISPATCHER
    if dispatcher is None:
        return {"ok": False, "error": "android_call_device_not_connected",
                "contact": raw, "network_called": False}
    try:
        result = dispatcher({"action": "call_phone", "contact": raw}, timeout_s=30)
    except TypeError:
        result = dispatcher({"action": "call_phone", "contact": raw})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "android_call_dispatch_failed",
                "error_type": type(exc).__name__}
    if not isinstance(result, dict):
        return {"ok": False, "error": "invalid_android_action_result"}
    return {"ok": bool(result.get("ok")), "contact": raw,
            "status": result.get("status"),
            "error": result.get("error"),
            "device": "ME3 Android companion"}


def send_message(contact: str, body: str, channel: str = "telegram",
                 confirm: bool = False) -> dict:
    """Send Telegram text using a protected contact mapping."""
    channel = str(channel or "telegram").strip().lower()
    raw = str(contact or "").strip()
    body = str(body or "").strip()
    if not raw or not body:
        return {"ok": False, "error": "contact and body are required"}
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    if channel != "telegram":
        return {"ok": False, "error": "telegram_only_channel", "channel": channel,
                "network_called": False}
    aliases = {"WIFE": "MNCFRNDZ", "WIFEY": "MNCFRNDZ", "WIFEEY": "MNCFRNDZ",
               "WIFEEYY_DUMALAOG_FERNANDEZ": "MNCFRNDZ",
               "MCFRNDZ": "MNCFRNDZ", "MNCFRNDZ": "MNCFRNDZ"}
    mapping_key = "ME3_TELEGRAM_CHAT_" + aliases.get(slug, slug)
    chat_id = raw if re.fullmatch(r"-?\d+", raw) else secure_config.get(mapping_key)
    if not chat_id:
        return {"ok": False, "error": "telegram_contact_not_configured", "contact": raw}
    return channels.send_telegram_message(chat_id, body, confirm=confirm)


def telegram_status() -> dict:
    token_ready = bool(secure_config.get("TELEGRAM_BOT_TOKEN"))
    mapped = bool(secure_config.get("ME3_TELEGRAM_CHAT_MNCFRNDZ"))
    return {"configured": token_ready and mapped, "bot_configured": token_ready,
            "target_mapped": mapped, "target": "@mncfrndz"}


def phone_call_status() -> dict:
    with _LOCK:
        dispatcher = _DEVICE_ACTION_DISPATCHER
    owner = getattr(dispatcher, "__self__", None) if dispatcher else None
    try:
        snapshot = owner.snapshot() if owner and hasattr(owner, "snapshot") else {}
        connected = int(snapshot.get("device_action_clients", 0))
    except Exception:
        connected = 0
    return {"provider": "android_companion", "configured": connected > 0,
            "connected_android_clients": connected,
            "requires_connected_android_client": True,
            "telegram_voice_calls": "unsupported_by_telegram_bot_api"}


def save_telegram_bot_token(token: str) -> dict:
    return secure_config.set_values({"TELEGRAM_BOT_TOKEN": token})


def connect_telegram_target(username: str = "mncfrndz") -> dict:
    """Map @mncfrndz to its Bot API chat ID after the user starts the bot."""
    import requests

    username = str(username or "mncfrndz").strip().lstrip("@").lower()
    if username != "mncfrndz":
        return {"ok": False, "error": "only_the_configured_target_is_supported"}
    token = secure_config.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"ok": False, "error": "telegram_bot_token_required"}
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"limit": 100, "allowed_updates": json.dumps(["message"])},
            timeout=15,
        )
        if response.status_code != 200:
            return {"ok": False, "error": "telegram_get_updates_rejected",
                    "status_code": response.status_code}
        data = response.json()
        if data.get("ok") is not True:
            return {"ok": False, "error": "telegram_get_updates_rejected"}
        chat_id = None
        for update in data.get("result") or []:
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            sender = message.get("from") or {}
            candidates = (chat, sender)
            if any(str(candidate.get("username") or "").lower() == username
                   for candidate in candidates):
                found = chat.get("id")
                if isinstance(found, (int, str)) and re.fullmatch(r"-?\d+", str(found)):
                    chat_id = str(found)
                    break
        if not chat_id:
            return {"ok": False, "error": "target_must_start_the_me3_bot_once",
                    "target": "@mncfrndz"}
        aliases = {
            "ME3_TELEGRAM_CHAT_MNCFRNDZ": chat_id,
            "ME3_TELEGRAM_CHAT_WIFE": chat_id,
            "ME3_TELEGRAM_CHAT_WIFEY": chat_id,
            "ME3_TELEGRAM_CHAT_WIFEEY": chat_id,
            "ME3_TELEGRAM_CHAT_WIFEEYY_DUMALAOG_FERNANDEZ": chat_id,
        }
        saved = secure_config.set_values(aliases)
        if not saved.get("ok"):
            return saved
        return {"ok": True, "mapped": True, "target": "@mncfrndz"}
    except Exception as exc:  # noqa: BLE001 - never expose token or response body
        return {"ok": False, "error": "telegram_connect_failed",
                "error_type": type(exc).__name__}

