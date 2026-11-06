"""ME2 command parity adapters for the model-driven ME3 agent.

ME3 does not train a replacement command classifier.  Qwen (or another
OpenAI-compatible model) selects these explicit tools from natural language,
while this module owns the deterministic action and safety boundary.  Local
state is always available for a demo; optional Home Assistant and channel
credentials turn the device adapters into live integrations.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import threading
import time
from urllib.parse import quote

from . import channels

_LOCK = threading.RLock()
_MEDIA = {
    "state": "stopped", "track": "", "volume": 50,
    "service": "youtube_music", "browser_url": "",
}
_MUSIC_EVENT_SINK = None
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
    """Return the deterministic media state and browser handoff URL."""
    with _LOCK:
        return {"ok": True, "media": dict(_MEDIA)}


def play_music(query: str, service: str = "youtube_music") -> dict:
    """Prepare a browser music handoff for YouTube Music or Spotify.

    Google Play Music no longer exists as a separate service; the Google
    choice is YouTube Music.  The Pi never reads browser cookies or credentials.
    The UI opens the returned URL only after the operator clicks its button, so
    an already-authenticated browser session can be used safely.
    """
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "error": "music query is required"}
    requested = str(service or "youtube_music").strip().lower().replace("-", "_")
    aliases = {
        "browser": "youtube_music", "google": "youtube_music",
        "google_music": "youtube_music", "youtube_music": "youtube_music",
        "youtube": "youtube", "spotify": "spotify", "local": "local",
    }
    service = aliases.get(requested)
    if service is None:
        return {"ok": False, "error": "unsupported music service"}
    if service == "youtube_music":
        browser_url = "https://music.youtube.com/search?q=" + quote(query)
        provider = "YouTube Music"
    elif service == "youtube":
        browser_url = "https://www.youtube.com/results?search_query=" + quote(query)
        provider = "YouTube"
    elif service == "spotify":
        browser_url = "https://open.spotify.com/search/" + quote(query, safe="")
        provider = "Spotify"
    else:
        browser_url = ""
        provider = "local demo"
    with _LOCK:
        _MEDIA.update({"state": "playing", "track": query, "service": service,
                       "browser_url": browser_url})
        state = dict(_MEDIA)
    event = {"action": "play", "provider": provider, "query": query,
             "url": browser_url, "media": state,
             "requires_user_click": bool(browser_url)}
    _emit_music(event)
    return {"ok": True, "action": "play_music", "provider": provider,
            "media": state, "browser_url": browser_url,
            "requires_user_click": bool(browser_url),
            "note": ("Click Open in browser in the ME3 UI to use the existing signed-in "
                      "browser session; ME3 does not copy cookies or credentials.")}


def media_control(action: str, value: int | None = None) -> dict:
    """Handle pause/stop/skip and volume commands used by the ME2 demo."""
    action = str(action or "").strip().lower().replace(" ", "_")
    aliases = {"next": "skip", "louder": "volume_up", "quieter": "volume_down",
               "volumeup": "volume_up", "volumedown": "volume_down"}
    action = aliases.get(action, action)
    allowed = {"play", "pause", "stop", "skip", "volume_up", "volume_down", "set_volume"}
    if action not in allowed:
        return {"ok": False, "error": f"unsupported media action: {action}"}
    with _LOCK:
        if action in {"play", "pause", "stop"}:
            _MEDIA["state"] = {"play": "playing", "pause": "paused", "stop": "stopped"}[action]
        elif action == "skip":
            _MEDIA["state"] = "playing"
            _MEDIA["track"] = "next track"
        else:
            current = int(_MEDIA["volume"])
            if action == "volume_up":
                current += 10
            elif action == "volume_down":
                current -= 10
            else:
                if value is None:
                    return {"ok": False, "error": "value is required for set_volume"}
                current = int(value)
            _MEDIA["volume"] = max(0, min(100, current))
        state = dict(_MEDIA)
    provider = {"youtube_music": "YouTube Music", "youtube": "YouTube",
                "spotify": "Spotify", "local": "local demo"}.get(
                    state.get("service"), state.get("service", "browser"))
    _emit_music({"action": action, "provider": provider,
                 "query": state.get("track", ""),
                 "url": state.get("browser_url", ""), "media": state,
                 "requires_user_click": bool(state.get("browser_url"))})
    return {"ok": True, "action": action, "media": state,
            "provider": provider,
            "browser_url": state.get("browser_url", ""),
            "requires_user_click": bool(state.get("browser_url"))}


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
    live = bool(os.getenv("HOME_ASSISTANT_URL") and os.getenv("HOME_ASSISTANT_TOKEN"))
    return {"ok": True, "room": room, "lights": device,
            "mode": "home_assistant_ready" if live else "local_demo",
            "live_integration": live,
            "note": ("Simulated bulb updated. Set HOME_ASSISTANT_URL, HOME_ASSISTANT_TOKEN, "
                     "and entity mapping before controlling physical lights.")}


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
    live = bool(os.getenv("HOME_ASSISTANT_URL") and os.getenv("HOME_ASSISTANT_TOKEN"))
    return {"ok": True, "room": room, "temperature_c": temperature_c,
            "mode": "home_assistant_ready" if live else "local_demo",
            "live_integration": live,
            "note": "Set HOME_ASSISTANT_URL, HOME_ASSISTANT_TOKEN, and climate entity mapping for a physical thermostat."}


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
    now = dt.datetime.now().astimezone()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += dt.timedelta(days=1)
    return candidate


def set_alarm(time_text: str, label: str = "") -> dict:
    """Create a next-occurrence local alarm (e.g. ``7:30 AM``)."""
    when = _parse_alarm_time(time_text)
    if when is None:
        return {"ok": False, "error": "use an alarm time such as 7:30 AM or 19:30"}
    alarm_id = _next("alarm")
    item = {"id": alarm_id, "time": _iso(when), "label": str(label or ""), "cancelled": False}
    with _LOCK:
        _ALARMS[alarm_id] = item
    return {"ok": True, "alarm": dict(item),
            "note": "The agent stores the alarm; connect a platform notification scheduler for audible firing."}


def list_alarms(include_cancelled: bool = False) -> dict:
    with _LOCK:
        return {"ok": True, "alarms": [dict(v) for v in _ALARMS.values()
                                         if include_cancelled or not v["cancelled"]]}


def cancel_alarm(alarm_id: str) -> dict:
    with _LOCK:
        item = _ALARMS.get(str(alarm_id))
        if not item:
            return {"ok": False, "error": "alarm_not_found"}
        item["cancelled"] = True
    return {"ok": True, "id": str(alarm_id)}


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
                when = when.astimezone()
        except ValueError:
            when = _parse_alarm_time(at_text)
    if when is None:
        return {"ok": False, "error": "provide delay_seconds or at_text"}
    reminder_id = _next("reminder")
    item = {"id": reminder_id, "text": text, "at": _iso(when), "completed": False}
    with _LOCK:
        _REMINDERS[reminder_id] = item
    return {"ok": True, "reminder": dict(item),
            "note": "The agent stores the reminder; connect a notification scheduler for delivery."}


def list_reminders(include_completed: bool = False) -> dict:
    with _LOCK:
        return {"ok": True, "reminders": [dict(v) for v in _REMINDERS.values()
                                             if include_completed or not v["completed"]]}


def complete_reminder(reminder_id: str) -> dict:
    with _LOCK:
        item = _REMINDERS.get(str(reminder_id))
        if not item:
            return {"ok": False, "error": "reminder_not_found"}
        item["completed"] = True
    return {"ok": True, "id": str(reminder_id)}


def call_contact(contact: str, confirm: bool = False) -> dict:
    """Resolve a contact from a number or ME3_CONTACT_<NAME> and call via Twilio."""
    raw = str(contact or "").strip()
    if not raw:
        return {"ok": False, "error": "contact is required"}
    if re.fullmatch(r"\+?[0-9][0-9 .()-]{6,}", raw):
        number = raw
    else:
        key = "ME3_CONTACT_" + re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
        number = os.getenv(key, "")
    if not number:
        return {"ok": False, "error": "contact_not_configured", "contact": raw,
                "note": "Set a protected ME3_CONTACT_<NAME> E.164 mapping."}
    return channels.make_call(number, confirm=confirm)


def send_message(contact: str, body: str, channel: str = "sms",
                 confirm: bool = False) -> dict:
    """Send SMS or Telegram text using a protected contact mapping."""
    channel = str(channel or "sms").strip().lower()
    raw = str(contact or "").strip()
    body = str(body or "").strip()
    if not raw or not body:
        return {"ok": False, "error": "contact and body are required"}
    slug = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    if channel == "telegram":
        chat_id = raw if re.fullmatch(r"-?\d+", raw) else os.getenv("ME3_TELEGRAM_CHAT_" + slug, "")
        if not chat_id:
            return {"ok": False, "error": "telegram_contact_not_configured", "contact": raw}
        return channels.send_telegram_message(chat_id, body, confirm=confirm)
    number = raw if re.fullmatch(r"\+?[0-9][0-9 .()-]{6,}", raw) else os.getenv("ME3_CONTACT_" + slug, "")
    if not number:
        return {"ok": False, "error": "contact_not_configured", "contact": raw}
    return channels.send_sms(number, body, confirm=confirm)

