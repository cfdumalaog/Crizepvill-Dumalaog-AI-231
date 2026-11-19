from __future__ import annotations

import json
import urllib.request

import pytest

from me3 import brain, permission_policy, tools
from me3.agent_status import LoopStatus, StatusServer


@pytest.fixture(autouse=True)
def isolated_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("ME3_PERMISSION_FILE", str(tmp_path / "permission_policy.json"))
    permission_policy.clear_pending()
    yield
    permission_policy.clear_pending()


def test_three_modes_are_persisted_and_have_expected_labels():
    assert permission_policy.set_mode("ask_approval")["label"] == "Ask for approval"
    assert permission_policy.status()["persisted"] is True
    assert permission_policy.set_mode("approve_for_me")["mode"] == "approve_for_me"
    assert permission_policy.set_mode("full_access")["label"] == "Full access"
    assert permission_policy.status()["mode"] == "full_access"
    assert permission_policy.set_mode("not-a-mode")["error"] == "invalid_permission_mode"


def test_mode_gates_internet_external_actions_and_shell():
    permission_policy.set_mode("ask_approval")
    assert permission_policy.requires_approval("web_search", {"query": "latest news"})
    assert permission_policy.requires_approval("run_command", {"command": "pytest", "cwd": "agent_workspace"})
    assert permission_policy.requires_approval("deploy_github_pages", {"repository": "owner/site"})
    assert not permission_policy.requires_approval("write_file", {"path": "agent_workspace/index.html"})

    permission_policy.set_mode("approve_for_me")
    assert not permission_policy.requires_approval("web_search", {"query": "latest news"})
    assert permission_policy.requires_approval("deploy_github_pages", {"repository": "owner/site"})
    assert permission_policy.requires_approval("write_file", {"path": "C:/demo/index.html"})
    assert not permission_policy.requires_approval("write_file", {"path": "agent_workspace/index.html"})

    permission_policy.set_mode("full_access")
    assert not permission_policy.requires_approval("deploy_github_pages", {"repository": "owner/site"})
    assert not permission_policy.requires_approval("run_command", {"command": "anything", "cwd": "C:/demo"})


def test_dispatch_returns_private_approval_request_then_executes_after_approval(monkeypatch):
    calls = []
    monkeypatch.setitem(tools._FUNCTIONS, "deploy_github_pages",
                        lambda **kwargs: calls.append(kwargs) or {"ok": True, "commit": "abc"})
    permission_policy.set_mode("ask_approval")
    blocked = tools.dispatch("deploy_github_pages", {"repository": "owner/site"})
    assert blocked["error"] == "approval_required"
    assert "repository" not in blocked["summary"] or "owner/site" in blocked["summary"]
    assert calls == []

    completed = tools.dispatch("deploy_github_pages", {"repository": "owner/site",
                                                        "_permission_approved": True})
    assert completed["ok"] is True
    assert calls == [{"repository": "owner/site"}]


def test_brain_resolves_approval_without_another_model_request(monkeypatch):
    permission_policy.set_mode("ask_approval")
    permission_policy.create_pending("deploy_github_pages", {"repository": "owner/site"})
    calls = []

    def fake_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True, "commit": "abc", "pages_url": "https://owner.github.io/site"}

    monkeypatch.setenv("ME3_PERMISSION_TEST_KEY", "in-memory-only")
    model = brain.ToolBrain({
        "base_url": "http://model.test/v1", "model": "test-model",
        "api_key_env": "ME3_PERMISSION_TEST_KEY", "system_prompt": "test",
    }, "test", dispatch_fn=fake_dispatch)
    reply = model.chat([], "Yes, go ahead.")
    assert "deployed" in reply.lower()
    assert calls[0][0] == "deploy_github_pages"
    assert calls[0][1]["_permission_approved"] is True


def test_permission_http_endpoints_round_trip():
    status = LoopStatus()
    server = StatusServer(
        status, html_dir=__import__("pathlib").Path(__file__).resolve().parents[1] / "web",
        get_state=lambda: "idle", get_stats=lambda: type("Stats", (), {})(), port=0,
        permission_status=permission_policy.status,
        permission_set=permission_policy.set_mode,
    )
    assert server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + "/api/permissions", timeout=3) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is True
        request = urllib.request.Request(
            base + "/api/permissions",
            data=json.dumps({"mode": "approve_for_me"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            updated = json.loads(response.read())
        assert updated["mode"] == "approve_for_me"
    finally:
        server.stop()
