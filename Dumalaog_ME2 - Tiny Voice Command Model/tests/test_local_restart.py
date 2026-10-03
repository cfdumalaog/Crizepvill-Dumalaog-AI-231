import base64
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from vcm_app import arguments_with_selected_device
from tinyvcm.restart import build_supervisor_command


PROJECT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = PROJECT / "scripts" / "restart_local_app.py"
SUPERVISOR_SPEC = importlib.util.spec_from_file_location("restart_local_app", SUPERVISOR_PATH)
SUPERVISOR = importlib.util.module_from_spec(SUPERVISOR_SPEC)
SUPERVISOR_SPEC.loader.exec_module(SUPERVISOR)


def test_microphone_selection_replaces_only_device_argument():
    original = ["--port", "7863", "--device", "fifine", "--live-weather"]
    assert arguments_with_selected_device(original, "41") == [
        "--port", "7863", "--live-weather", "--device", "41"
    ]
    assert arguments_with_selected_device(original, "auto") == [
        "--port", "7863", "--live-weather"
    ]


def test_microphone_selection_rejects_non_device_text():
    with pytest.raises(ValueError, match="listed microphone"):
        arguments_with_selected_device(["--port", "7863"], "41;whoami")


def test_supervisor_command_transports_launch_arguments_as_data():
    arguments = ["--port", "7863", "--model", "C:\\models\\name with spaces.onnx", "--device", "41"]
    command = build_supervisor_command("vcm", 1234, 7863, arguments)
    assert command[0] == sys.executable
    assert command[1] == str(SUPERVISOR_PATH)
    encoded = command[-1]
    decoded = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    assert decoded == arguments
    assert all(";" not in token for token in command)


def test_supervisor_rejects_recorder_launch_arguments_before_waiting():
    encoded = base64.urlsafe_b64encode(json.dumps(["unexpected"]).encode()).decode()
    with pytest.raises(ValueError, match="cannot accept launch arguments"):
        SUPERVISOR.restart("recorder", 1234, 7862, encoded)


def test_supervisor_process_check_distinguishes_live_and_missing_pids():
    assert SUPERVISOR._parent_exited(os.getpid()) is False
    assert SUPERVISOR._parent_exited(2_147_483_647) is True
