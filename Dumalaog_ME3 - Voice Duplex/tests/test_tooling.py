from __future__ import annotations

from types import SimpleNamespace

import pytest

from me3 import brain, tools
from me3.task_runner import BackgroundTaskManager


def test_tool_definitions_are_openai_compatible_and_unique():
    definitions = tools.tool_definitions()
    names = [item["function"]["name"] for item in definitions]
    assert len(names) == len(set(names))
    assert {"write_file", "run_command", "start_background_task", "web_search"} <= set(names)
    for item in definitions:
        assert item["type"] == "function"
        assert item["function"]["parameters"]["type"] == "object"


def test_command_policy_blocks_shell_and_external_access():
    with pytest.raises(ValueError):
        tools._argv("rm -rf agent_workspace")
    with pytest.raises(ValueError):
        tools._argv("python -c 'print(1)' | sh")
    with pytest.raises(ValueError):
        tools._argv("ssh pi hostname")
    assert tools._argv("python -c 'print(1)'")[0] == "python"


def test_background_task_manager_runs_and_reports(tmp_path):
    manager = BackgroundTaskManager(tmp_path)
    task = manager.start(
        ["python", "-c", "print('me3-task-ok')"],
        tmp_path,
        timeout_s=30,
    )
    assert task["ok"] is True
    task_id = task["task_id"]
    for _ in range(30):
        status = manager.status(task_id)
        if status.get("status") in {"completed", "failed", "timed_out"}:
            break
        import time
        time.sleep(0.05)
    assert manager.status(task_id)["status"] == "completed"
    assert "me3-task-ok" in manager.output(task_id)["text"]


def test_qwen_tool_loop_executes_tool_then_returns_text(monkeypatch):
    monkeypatch.setenv("ME3_TEST_KEY", "in-memory-test-only")
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": None,
                                    "tool_calls": [{"id": "c1", "type": "function",
                                                    "function": {"name": "system_info",
                                                                 "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "The system check is complete."}}]},
    ]

    def fake_post(*args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: responses.pop(0),
        )

    monkeypatch.setattr(brain.requests, "post", fake_post)
    b = brain.ToolBrain(
        {"base_url": "http://qwen.test/v1", "model": "Qwen/test",
         "api_key_env": "ME3_TEST_KEY", "temperature": 0.1,
         "max_tokens": 128, "timeout_s": 2, "max_tool_steps": 2,
         "system_prompt": "test"},
        "test",
    )
    assert b.chat([], "check the system") == "The system check is complete."
    assert responses == []

