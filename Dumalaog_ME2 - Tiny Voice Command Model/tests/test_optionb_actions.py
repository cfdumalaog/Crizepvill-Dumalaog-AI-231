from tinyvcm.devices import Devices
from tinyvcm_model.config import LABELS


def test_every_option_b_intent_has_a_deterministic_demo_action():
    devices = Devices(gpio=False, live_weather=False)
    try:
        for label in LABELS:
            response = devices.execute(label)
            assert response and response != "Command rejected.", label
    finally:
        devices.close()


def test_temperature_intents_update_celsius_without_corrupting_legacy_fahrenheit():
    devices = Devices(gpio=False)
    try:
        response = devices.execute("TEMPERATURE_18")
        state = devices.snapshot()
        assert state["thermostat_demo_c"] == 18
        assert state["thermostat_demo_f"] == 72
        assert "18 °C" in response
        assert "no HVAC connected" in response
    finally:
        devices.close()


def test_volume_intents_apply_to_output_adapter_and_update_reported_level():
    class FakeOutput:
        mode = "test-mixer"
        status = "test output"

        def __init__(self):
            self.levels = []

        def set_percent(self, percent):
            self.levels.append(percent)
            return True

    devices = Devices(gpio=False)
    output = FakeOutput()
    devices.volume_output = output
    try:
        assert devices.execute("VOLUME_UP") == "Volume set to 60%."
        assert devices.execute("VOLUME_DOWN") == "Volume set to 50%."
        assert output.levels == [60, 50]
        state = devices.snapshot()
        assert state["volume"] == 50
        assert state["volume_hardware_enabled"] is True
        assert state["volume_hardware_mode"] == "test-mixer"
    finally:
        devices.close()
