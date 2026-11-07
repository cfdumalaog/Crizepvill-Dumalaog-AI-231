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


def test_media_uses_spotify_and_devices_are_deterministic_demo_actions(monkeypatch):
    play_calls = []
    control_calls = []

    def fake_play(query, service="spotify"):
        play_calls.append((query, service))
        return {"ok": True, "provider": "Spotify", "track": "Filipino jazz",
                "artists": ["ME3 Test"], "device": "Laptop speaker",
                "playback_verified": True}

    def fake_control(action, value=None):
        control_calls.append((action, value))
        return {"ok": True, "provider": "Spotify", "action": action,
                "volume": 60}

    monkeypatch.setattr(me2_commands.spotify_player, "play", fake_play)
    monkeypatch.setattr(me2_commands.spotify_player, "control", fake_control)
    result = me2_commands.play_music("Filipino jazz")
    assert result["ok"] is True and result["provider"] == "Spotify"
    assert play_calls == [("Filipino jazz", "spotify")]
    assert me2_commands.media_control("louder")["volume"] == 60
    assert control_calls == [("volume_up", None)]
    lights = me2_commands.control_lights("dim", "living room", 35)
    assert lights["ok"] is True and lights["lights"]["value"] == 35
    thermostat = me2_commands.set_thermostat(24, "home")
    assert thermostat["ok"] is True and thermostat["temperature_c"] == 24


def test_simulated_light_state_is_shared_by_voice_tool_and_ui_adapter():
    me2_commands.control_lights("off", "all")
    initial = me2_commands.simulated_light_state()
    assert initial["physical"] is False
    assert initial["on"] is False

    turned_on = me2_commands.control_lights("on", "all")
    assert turned_on["lights"]["on"] is True
    assert me2_commands.simulated_light_state()["on"] is True

    dimmed = me2_commands.control_lights("dim", "all", 35)
    assert dimmed["lights"]["brightness"] == 35
    assert dimmed["lights"]["value"] == 35
    assert dimmed["lights"]["on"] is True


def test_music_uses_supported_spotify_playback_only(monkeypatch):
    monkeypatch.setattr(
        me2_commands.spotify_player, "play",
        lambda query, service="spotify": {
            "ok": True, "provider": "Spotify", "track": "Eraserheads",
            "playback_verified": True,
        },
    )
    youtube = me2_commands.play_music("Eraserheads", "google_music")
    spotify = me2_commands.play_music("Eraserheads", "spotify")
    assert youtube["ok"] is False
    assert youtube["error"] == "provider_has_no_supported_autoplay_api"
    assert spotify["provider"] == "Spotify"
    assert spotify["playback_verified"] is True


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


def test_alarm_and_reminder_lifecycle(tmp_path, monkeypatch):
    from me3 import google_calendar, scheduled_actions

    store = scheduled_actions.ScheduledActionStore(tmp_path / "actions.sqlite3")
    monkeypatch.setattr(me2_commands, "_store", lambda: store)
    monkeypatch.setattr(
        google_calendar, "create_reminder_event",
        lambda text, when: {"ok": False, "error": "google_calendar_not_configured"},
    )
    alarm = me2_commands.set_alarm("23:59", "class")
    assert alarm["ok"] is True
    alarm_id = alarm["alarm"]["id"]
    assert me2_commands.cancel_alarm(alarm_id)["ok"] is True
    reminder = me2_commands.create_reminder("submit benchmark", delay_seconds=60)
    assert reminder["ok"] is True
    assert reminder["calendar_synced"] is False
    assert reminder["delivery"] == "persistent_me3_voice_and_ui"
    reminder_id = reminder["reminder"]["id"]
    assert me2_commands.complete_reminder(reminder_id)["ok"] is True


def test_named_calls_and_messages_require_protected_contact_mapping(monkeypatch):
    monkeypatch.delenv("ME3_CONTACT_MOM", raising=False)
    monkeypatch.delenv("ME3_TELEGRAM_CHAT_MOM", raising=False)
    assert me2_commands.call_contact("mom")["error"] == "confirmation_required"
    assert me2_commands.call_contact("mom", confirm=True)["error"] == "android_call_device_not_connected"
    assert me2_commands.send_message("mom", "hello")["error"] == "telegram_contact_not_configured"
    assert me2_commands.send_message("mom", "hello", channel="telegram")["error"] == "telegram_contact_not_configured"
