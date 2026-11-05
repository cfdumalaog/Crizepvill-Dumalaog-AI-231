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
    progress = []
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
        on_progress=lambda stage, task: progress.append((stage, task)),
    )
    assert b.chat([], "check the system") == "The system check is complete."
    assert responses == []
    assert any(stage == "thinking" for stage, _task in progress)
    assert any("Running tool: system_info" in task
               for _stage, task in progress)


def test_tool_budget_finishes_with_summary_instead_of_limit_error(monkeypatch):
    monkeypatch.setenv("ME3_TEST_KEY", "in-memory-test-only")
    captured = []
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": None,
                                    "tool_calls": [{"id": "c1", "type": "function",
                                                    "function": {"name": "system_info",
                                                                 "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"role": "assistant",
                                    "content": "The requested task is complete."}}]},
    ]

    def fake_post(*args, **kwargs):
        captured.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: responses.pop(0),
        )

    monkeypatch.setattr(brain.requests, "post", fake_post)
    b = brain.ToolBrain(
        {"base_url": "http://qwen.test/v1", "model": "Qwen/test",
         "api_key_env": "ME3_TEST_KEY", "max_tokens": 128, "timeout_s": 2,
         "max_tool_steps": 1, "system_prompt": "test"},
        "test",
    )
    assert b.chat([], "complete the system check") == "The requested task is complete."
    assert len(captured) == 2
    assert "tools" not in captured[-1]


def test_luna_tool_loop_disables_reasoning_and_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_TEST_KEY", "in-memory-test-only")
    captured = []

    def fake_post(*args, **kwargs):
        captured.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {
                "role": "assistant", "content": "Luna is ready."
            }}]},
        )

    monkeypatch.setattr(brain.requests, "post", fake_post)
    b = brain.ToolBrain(
        {"base_url": "https://api.openai.com/v1", "model": "gpt-5.6-luna",
         "provider": "luna", "api_key_env": "OPENAI_TEST_KEY",
         "reasoning_effort": "medium", "temperature": None,
         "max_completion_tokens": 256, "timeout_s": 2,
         "system_prompt": "test"},
        "test",
    )
    assert b.chat([], "say hello") == "Luna is ready."
    assert captured[0]["model"] == "gpt-5.6-luna"
    assert "reasoning_effort" not in captured[0]
    assert captured[0]["max_completion_tokens"] == 256
    assert "temperature" not in captured[0]


def test_openai_compatible_response_stream_is_assembled_without_reasoning(monkeypatch):
    monkeypatch.setenv("ME3_STREAM_TEST_KEY", "in-memory-test-only")
    captured = []

    class StreamResponse:
        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            del decode_unicode
            return iter([
                'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}',
                'data: {"choices":[{"delta":{"content":" there"}}]}',
                'data: [DONE]',
            ])

        def close(self):
            return None

    def fake_post(*args, **kwargs):
        captured.append(kwargs)
        return StreamResponse()

    monkeypatch.setattr(brain.requests, "post", fake_post)
    b = brain.ToolBrain(
        {"base_url": "http://qwen.test/v1", "model": "Qwen/test",
         "api_key_env": "ME3_STREAM_TEST_KEY", "stream": True,
         "max_tokens": 128, "timeout_s": 2, "system_prompt": "test"},
        "test",
    )
    assert b.chat([], "say hello") == "Hello there"
    assert captured[0]["json"]["stream"] is True
    assert captured[0]["stream"] is True
    assert "reasoning_effort" not in captured[0]["json"]


def test_auto_provider_prefers_qwen_then_luna(monkeypatch):
    cfg = {"llm": {
        "remote": {"model": "qwen", "api_key_env": "QWEN_KEY"},
        "openai": {"model": "gpt-5.6-luna", "api_key_env": "OPENAI_KEY"},
    }}
    monkeypatch.delenv("QWEN_KEY", raising=False)
    monkeypatch.setenv("OPENAI_KEY", "in-memory-test-only")
    provider, selected = brain.resolve_llm_config(cfg, "auto")
    assert provider == "luna"
    assert selected["model"] == "gpt-5.6-luna"
    monkeypatch.setenv("QWEN_KEY", "in-memory-test-only")
    provider, selected = brain.resolve_llm_config(cfg, "auto")
    assert provider == "qwen"
    assert selected["model"] == "qwen"

