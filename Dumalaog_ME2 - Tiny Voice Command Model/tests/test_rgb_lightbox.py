import pytest

from vcm_app import ASSISTANT_HTML, STUDIO_HTML, apply_light_demo_command
from tinyvcm.devices import Devices


def test_rgb_device_state_tracks_light_color_and_brightness():
    devices = Devices(gpio=False)
    try:
        initial = devices.snapshot()
        assert initial["lights_percent"] == 0
        assert initial["lights_color"] == "white"
        assert initial["lights_hex"] == "#000000"
        assert initial["rgb_hardware_enabled"] is False

        devices.execute("LIGHT_ON")
        assert devices.snapshot()["lights_rgb"] == [1.0, 1.0, 1.0]

        devices.execute("COLOR_BLUE")
        assert devices.snapshot()["lights_color"] == "blue"
        assert devices.snapshot()["lights_hex"] == "#0000ff"

        devices.execute("BRIGHTNESS_20")
        dimmed = devices.snapshot()
        assert dimmed["lights_percent"] == 20
        assert dimmed["lights_rgb"] == [0.0, 0.0, 0.2]
        assert dimmed["lights_hex"] == "#000033"

        devices.execute("LIGHT_OFF")
        off = devices.snapshot()
        assert off["lights_percent"] == 0
        assert off["lights_color"] == "blue"
        assert off["lights_hex"] == "#000000"
    finally:
        devices.close()


def test_studio_contains_large_live_rgb_lightbox():
    assert "Dedicated RGB lights" in STUDIO_HTML
    assert 'id="rgbStage"' in STUDIO_HTML
    assert 'id="lightBrightness"' in STUDIO_HTML
    assert "rgb_hardware_enabled" in STUDIO_HTML


@pytest.mark.parametrize('command', ['COLOR_RED', 'COLOR_GREEN', 'COLOR_BLUE'])
def test_light_on_after_color_and_off_restores_all_three_channels(command):
    devices = Devices(gpio=False)
    try:
        apply_light_demo_command(devices, command)
        assert sum(value > 0 for value in devices.snapshot()['lights_rgb']) == 1
        apply_light_demo_command(devices, 'LIGHT_OFF')
        assert devices.snapshot()['lights_rgb'] == [0.0, 0.0, 0.0]
        apply_light_demo_command(devices, 'LIGHT_ON')
        state = devices.snapshot()
        assert state['lights_color'] == 'white'
        assert state['lights_percent'] == 100
        assert state['lights_rgb'] == [1.0, 1.0, 1.0]
    finally:
        devices.close()


def test_assistant_contains_live_rgb_lightbox_and_local_demo_controls():
    assert 'id="assistantRgbPreview"' in ASSISTANT_HTML
    assert 'id="assistantLightBrightness"' in ASSISTANT_HTML
    assert "updateAssistantLight(devices)" in ASSISTANT_HTML
    assert "fetch('/api/light/demo'" in ASSISTANT_HTML
    assert "GPIO disabled" in ASSISTANT_HTML


def test_assistant_light_demo_controls_drive_only_supported_light_commands():
    devices = Devices(gpio=False)
    try:
        expected = {
            "COLOR_RED": ("red", "#ff0000"),
            "COLOR_GREEN": ("green", "#00ff00"),
            "COLOR_BLUE": ("blue", "#0000ff"),
        }
        for command, (color, hex_value) in expected.items():
            assert apply_light_demo_command(devices, command) == f"LED color set to {color}."
            state = devices.snapshot()
            assert state["lights_color"] == color
            assert state["lights_percent"] == 100
            assert state["lights_hex"] == hex_value
        assert apply_light_demo_command(devices, "LIGHT_OFF") == "Lights off."
        assert devices.snapshot()["lights_percent"] == 0
        assert devices.snapshot()["lights_hex"] == "#000000"
        with pytest.raises(ValueError, match="Unsupported"):
            apply_light_demo_command(devices, "WEATHER")
    finally:
        devices.close()


def test_assistant_has_a_single_cached_voice_and_live_playback_volume_control():
    assert ASSISTANT_HTML.count("new SpeechSynthesisUtterance(") == 1
    assert "assistantSpeechVoice" in ASSISTANT_HTML
    assert "utterance.voice = assistantSpeechVoice" in ASSISTANT_HTML
    assert "utterance.volume = Math.max(0, Math.min(1, currentVolume / 100))" in ASSISTANT_HTML
    assert 'id="assistantVolumeSlider"' in ASSISTANT_HTML
    assert "htmlAudio.volume = volume / 100" in ASSISTANT_HTML
    assert "fetch('/api/volume'" in ASSISTANT_HTML
    assert 'id="volumeMeter" type="range"' in STUDIO_HTML


def test_both_pages_explain_wake_only_standby_and_audio_expiry():
    assert "Wake-only standby" in STUDIO_HTML
    assert "remaining before sleep" in STUDIO_HTML
