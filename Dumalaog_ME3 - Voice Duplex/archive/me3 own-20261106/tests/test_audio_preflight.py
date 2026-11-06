from __future__ import annotations

from scripts.pi_audio_preflight import _alsa_has_hardware, _device_row


def test_audio_preflight_device_rows_are_redacted_to_safe_topology():
    row = _device_row(3, {
        "name": "USB microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000,
        "hostapi": 0,
    })

    assert row == {
        "index": 3,
        "name": "USB microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000,
        "hostapi": 0,
    }


def test_audio_preflight_detects_alsa_cards_without_exposing_audio_data():
    assert _alsa_has_hardware({"stdout": "**** List ****\ncard 1: usb [USB Audio]"})
    assert not _alsa_has_hardware({"stdout": "**** List ****"})
