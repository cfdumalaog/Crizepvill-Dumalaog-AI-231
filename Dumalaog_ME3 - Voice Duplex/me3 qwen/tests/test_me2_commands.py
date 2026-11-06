from __future__ import annotations

from types import SimpleNamespace

from me3 import me2_commands, tools


def test_me2_command_surface_is_exposed_to_the_model():
    expected = {
        "play_music", "media_control", "get_weather", "control_lights",
        "set_thermostat", "set_alarm", "list_alarms", "cancel_alarm",
        "create_reminder", "list_reminders", "complete_reminder",
        "call_contact", "send_message",
    }
    assert expected <= set(tools._FUNCTIONS)
    names = {item["function"]["name"] for item in tools.tool_definitions()}
    assert expected <= names


def test_media_and_device_commands_are_deterministic_demo_actions():
    result = me2_commands.play_music("Filipino jazz")
    assert result["media"]["state"] == "playing"
    assert result["provider"] == "YouTube Music"
    assert result["browser_url"].startswith("https://music.youtube.com/search?q=")
    assert me2_commands.media_control("louder")["media"]["volume"] == 60
    lights = me2_commands.control_lights("dim", "living room", 35)
    assert lights["ok"] is True and lights["lights"]["value"] == 35
    thermostat = me2_commands.set_thermostat(24, "home")
    assert thermostat["ok"] is True and thermostat["temperature_c"] == 24


def test_music_handoff_supports_youtube_music_and_spotify(monkeypatch):
    events = []
    me2_commands.set_music_event_sink(events.append)
    try:
        youtube = me2_commands.play_music("Eraserheads", "google_music")
        spotify = me2_commands.play_music("Eraserheads", "spotify")
    finally:
        me2_commands.set_music_event_sink(None)
    assert youtube["provider"] == "YouTube Music"
    assert youtube["requires_user_click"] is True
    assert spotify["provider"] == "Spotify"
    assert spotify["browser_url"].startswith("https://open.spotify.com/search/")
    assert [event["event"] for event in events] == ["music_action", "music_action"]


def test_weather_parser_uses_external_api_without_hiding_failures(monkeypatch):
    import requests

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"current_condition": [{
            "temp_C": "29", "FeelsLikeC": "31", "humidity": "70",
            "weatherDesc": [{"value": "Partly cloudy"}],
        }]},
    )
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    result = me2_commands.get_weather("Manila")
    assert result["ok"] is True
    assert result["temperature_c"] == "29"
    assert result["description"] == "Partly cloudy"


def test_alarm_and_reminder_lifecycle():
    alarm = me2_commands.set_alarm("23:59", "class")
    assert alarm["ok"] is True
    alarm_id = alarm["alarm"]["id"]
    assert me2_commands.cancel_alarm(alarm_id)["ok"] is True
    reminder = me2_commands.create_reminder("submit benchmark", delay_seconds=60)
    assert reminder["ok"] is True
    reminder_id = reminder["reminder"]["id"]
    assert me2_commands.complete_reminder(reminder_id)["ok"] is True


def test_named_calls_and_messages_require_protected_contact_mapping(monkeypatch):
    monkeypatch.delenv("ME3_CONTACT_MOM", raising=False)
    monkeypatch.delenv("ME3_TELEGRAM_CHAT_MOM", raising=False)
    assert me2_commands.call_contact("mom")["error"] == "contact_not_configured"
    assert me2_commands.send_message("mom", "hello")["error"] == "contact_not_configured"
    assert me2_commands.send_message("mom", "hello", channel="telegram")["error"] == "telegram_contact_not_configured"
