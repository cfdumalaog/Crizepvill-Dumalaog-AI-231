from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

from me3 import brain, dev_workflow


def test_software_request_requires_an_action_not_a_general_question():
    assert dev_workflow.is_software_request("Build a website for my project")
    assert dev_workflow.is_software_request("Fix the bug in the app")
    assert dev_workflow.is_software_request("Gumawa ka ng website para sa tindahan")
    assert not dev_workflow.is_software_request("How do I build a website?")
    assert not dev_workflow.is_software_request("Paano gumawa ng website?")
    assert not dev_workflow.is_software_request("Set a timer")
    assert dev_workflow.verification_root("agent_workspace/shop/src/app.js") == "agent_workspace/shop"
    assert dev_workflow.verification_root("agent_workspace/helper.py") == "agent_workspace/helper.py"


def test_verifier_reports_source_errors_and_missing_assets(tmp_path):
    app = tmp_path / "shop"
    app.mkdir()
    (app / "index.html").write_text(
        '<html><link rel="stylesheet" href="missing.css"></html>', encoding="utf-8")
    (app / "data.json").write_text('{ broken', encoding="utf-8")
    failed = dev_workflow.verify_project(app)
    assert not failed["ok"]
    assert any("data.json" in item for item in failed["errors"])
    assert any("missing.css" in item for item in failed["errors"])
    (app / "data.json").write_text('{"ready": true}', encoding="utf-8")
    (app / "missing.css").write_text("body { color: navy; }", encoding="utf-8")
    passed = dev_workflow.verify_project(app)
    assert passed["ok"]
    assert passed["checks"]["html"] == 1
    assert passed["checks"]["json"] == 1
    single = dev_workflow.verify_project(app / "data.json")
    assert single["ok"] and single["files_scanned"] == 1


def test_verifier_checks_inline_webapp_javascript_when_node_is_available(tmp_path):
    if not shutil.which("node"):
        return
    page = tmp_path / "index.html"
    page.write_text("<script>const value = ;</script>", encoding="utf-8")
    failed = dev_workflow.verify_project(tmp_path)
    assert not failed["ok"]
    assert any("inline JavaScript" in item for item in failed["errors"])
    page.write_text("<script>const value = 1;</script>", encoding="utf-8")
    assert dev_workflow.verify_project(tmp_path)["ok"]


def test_dev_brain_repairs_failed_auto_check_before_claiming_success(monkeypatch, tmp_path):
    monkeypatch.setenv("ME3_DEV_TEST_KEY", "in-memory-test-only")
    target = tmp_path / "app" / "config.json"
    responses = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "write-1", "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({
                    "path": str(target), "content": "{ broken"}),
            }}]},
        {"role": "assistant", "content": "Finished the app."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "write-2", "type": "function", "function": {
                "name": "write_file", "arguments": json.dumps({
                    "path": str(target), "content": '{"ready": true}',
                    "overwrite": True}),
            }}]},
        {"role": "assistant", "content": "Finished the app."},
        {"role": "assistant", "content": "The source files now pass verification."},
    ]
    seen = []

    def fake_post(*_args, **_kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": responses.pop(0)}]},
        )

    def fake_dispatch(name, args):
        seen.append(name)
        if name == "write_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"], encoding="utf-8")
            return {"ok": True, "path": args["path"]}
        if name == "verify_project":
            return dev_workflow.verify_project(target.parent)
        raise AssertionError(name)

    monkeypatch.setattr(brain.requests, "post", fake_post)
    tool_brain = brain.ToolBrain({
        "base_url": "http://qwen.test/v1", "model": "Qwen/test",
        "api_key_env": "ME3_DEV_TEST_KEY", "max_tool_steps": 6,
        "max_development_steps": 8, "system_prompt": "test", "timeout_s": 2,
    }, "test", dispatch_fn=fake_dispatch)
    reply = tool_brain.chat([], "Build a web app")
    assert reply == "The source files now pass verification."
    assert seen == ["write_file", "verify_project", "write_file", "verify_project"]
    assert responses == []


def test_dev_brain_retries_when_model_only_promises_work(monkeypatch):
    monkeypatch.setenv("ME3_DEV_TEST_KEY", "in-memory-test-only")
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": "I will build it."}}]},
        {"choices": [{"message": {"role": "assistant", "content": "It is done."}}]},
    ]
    monkeypatch.setattr(brain.requests, "post", lambda *_args, **_kwargs:
                        SimpleNamespace(raise_for_status=lambda: None,
                                        json=lambda: responses.pop(0)))
    tool_brain = brain.ToolBrain({
        "base_url": "http://qwen.test/v1", "model": "Qwen/test",
        "api_key_env": "ME3_DEV_TEST_KEY", "max_tool_steps": 3,
        "system_prompt": "test", "timeout_s": 2,
    }, "test")
    assert tool_brain.chat([], "Create a website") == "I have not changed or tested the software yet."


def test_development_reply_does_not_claim_unrun_tests_or_deployment():
    reply = brain._ground_development_reply(
        "The tests passed and the website is live at the new URL.",
        "agent_workspace/shop", tests_passed=False,
        serving_url=None, github_published=False,
    )
    assert "not verified" in reply
    assert "tests passed" not in reply.lower()
    assert brain._ground_development_reply(
        "Tests passed.", "agent_workspace/shop", tests_passed=True,
        serving_url=None, github_published=False,
    ) == "Tests passed."
