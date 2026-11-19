from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

from me3 import brain, github_pages, secure_config, tools
from me3.agent import TurnStats
from me3.agent_status import LoopStatus, StatusServer


class _Response:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = {}

    def json(self):
        return self._payload


class _GitHubRequests:
    def __init__(self):
        self.calls = []
        self.blob_no = 0

    def request(self, method, url, *, headers, json, timeout):
        del headers, timeout
        self.calls.append((method, url, json))
        if url.endswith("/user"):
            return _Response(200, {"login": "test-owner"})
        if "/user/repos?" in url and method == "GET":
            return _Response(200, [{
                "full_name": "test-owner/site", "private": True,
                "default_branch": "main",
                "permissions": {"pull": True, "push": True, "admin": True},
                "html_url": "https://github.com/test-owner/site",
            }])
        if url.endswith("/repos/test-owner/site") and method == "GET":
            return _Response(200, {
                "full_name": "test-owner/site",
                "private": True,
                "default_branch": "main",
                "permissions": {"pull": True, "push": True, "admin": True},
                "html_url": "https://github.com/test-owner/site",
            })
        if url.endswith("/repos/alice/other-site") and method == "GET":
            return _Response(200, {
                "full_name": "alice/other-site",
                "private": False,
                "default_branch": "main",
                "permissions": {"pull": True, "push": True},
                "html_url": "https://github.com/alice/other-site",
            })
        if "/git/ref/heads/main" in url:
            return _Response(200, {"object": {"sha": "head-sha"}})
        if "/git/commits/head-sha" in url:
            return _Response(200, {"tree": {"sha": "base-tree"}})
        if url.endswith("/git/blobs"):
            self.blob_no += 1
            return _Response(201, {"sha": f"blob-{self.blob_no}"})
        if url.endswith("/git/trees"):
            return _Response(201, {"sha": "new-tree"})
        if url.endswith("/git/commits"):
            return _Response(201, {"sha": "new-commit"})
        if url.endswith("/git/refs/heads/main") and method == "PATCH":
            return _Response(200, {"object": {"sha": "new-commit"}})
        if url.endswith("/pages") and method == "GET":
            return _Response(404)
        if url.endswith("/pages") and method == "POST":
            return _Response(201, {"html_url": "https://test-owner.github.io/site/"})
        raise AssertionError(f"unexpected GitHub request: {method} {url}")


def _site(tmp_path: Path) -> Path:
    root = tmp_path / "agent_workspace" / "portfolio"
    root.mkdir(parents=True)
    (root / "index.html").write_text("<h1>ME3</h1>", encoding="utf-8")
    (root / "styles.css").write_text("body { color: white; }", encoding="utf-8")
    return root


def test_github_health_does_not_expose_token(monkeypatch):
    monkeypatch.setattr(secure_config, "get", lambda name: "secret" if name == "GITHUB_TOKEN" else "")
    result = github_pages.health()
    assert result["configured"] is True
    assert "secret" not in json.dumps(result)


def test_github_configure_token_validates_before_saving(monkeypatch):
    fake = _GitHubRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    saved = {}
    monkeypatch.setattr(secure_config, "set_values", lambda values: saved.update(values) or {"ok": True})
    token = "ghp_" + "A" * 36
    result = github_pages.configure_token(token)
    assert result == {"ok": True, "configured": True, "login": "test-owner"}
    assert saved == {"GITHUB_TOKEN": token}
    assert token not in json.dumps(result)


def test_github_deploy_runs_on_user_request_without_second_confirmation(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(github_pages, "AGENT_ROOT", tmp_path / "agent_workspace")
    monkeypatch.setattr(github_pages, "_publish_state_path",
                        lambda: tmp_path / "deployment" / "last_github_pages_publish.json")
    monkeypatch.setattr(secure_config, "get", lambda name: "ghp_" + "B" * 36)
    fake = _GitHubRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    result = github_pages.deploy(str(site), "test-owner", "site", confirm=False)
    assert result["ok"] is True
    assert result["commit"] == "new-commit"
    assert any(method == "PATCH" for method, _url, _body in fake.calls)


def test_requested_new_repository_is_created_before_publish(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(github_pages, "AGENT_ROOT", tmp_path / "agent_workspace")
    monkeypatch.setattr(github_pages, "_publish_state_path",
                        lambda: tmp_path / "deployment" / "last_github_pages_publish.json")
    monkeypatch.setattr(secure_config, "get", lambda name: "ghp_" + "B" * 36)

    class NewRepoRequests(_GitHubRequests):
        created = False

        def request(self, method, url, *, headers, json, timeout):
            if url.endswith("/repos/test-owner/new-site") and method == "GET":
                self.calls.append((method, url, json))
                if not self.created:
                    return _Response(404)
                return _Response(200, {"full_name": "test-owner/new-site",
                                       "permissions": {"pull": True, "push": True}})
            if url.endswith("/user/repos") and method == "POST":
                self.calls.append((method, url, json))
                self.created = True
                return _Response(201, {"full_name": "test-owner/new-site",
                                       "default_branch": "main", "private": False})
            return super().request(method, url, headers=headers, json=json,
                                   timeout=timeout)

    fake = NewRepoRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    result = github_pages.deploy(str(site), "test-owner", "new-site",
                                 create_if_missing=True)
    assert result["ok"] is True
    assert result["created_repository"] is True
    assert result["commit"] == "new-commit"
    assert any(method == "POST" and url.endswith("/user/repos")
               and body["auto_init"] is True and body["private"] is False
               for method, url, body in fake.calls)


def test_github_deploy_commits_files_and_enables_pages(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(github_pages, "AGENT_ROOT", tmp_path / "agent_workspace")
    monkeypatch.setattr(
        github_pages, "_publish_state_path",
        lambda: tmp_path / "deployment" / "last_github_pages_publish.json",
    )
    monkeypatch.setattr(secure_config, "get", lambda name: "ghp_" + "B" * 36)
    fake = _GitHubRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    result = github_pages.deploy(
        str(site), "test-owner", "site", branch="main", confirm=True,
    )
    assert result["ok"] is True
    assert result["commit"] == "new-commit"
    assert result["files"] == 2
    assert result["pages"] == "enabled"
    tree_call = next(call for call in fake.calls if call[0] == "POST" and call[1].endswith("/git/trees"))
    remote_paths = {entry["path"] for entry in tree_call[2]["tree"]}
    assert remote_paths == {"index.html", "styles.css"}
    blob_call = next(call for call in fake.calls if call[1].endswith("/git/blobs"))
    assert base64.b64decode(blob_call[2]["content"]) in {b"<h1>ME3</h1>", b"body { color: white; }"}


def test_tool_registry_and_voice_confirmation_cover_github_pages():
    names = {item["function"]["name"] for item in tools.tool_definitions()}
    assert {"github_pages_health", "github_repositories",
            "github_repository_access", "deploy_github_pages"} <= names
    assert brain.ToolBrain._explicit_action_request(
        "deploy_github_pages", "deploy my website to GitHub Pages", []
    ) is True


def test_repository_discovery_and_access_are_api_backed(monkeypatch):
    monkeypatch.setattr(secure_config, "get", lambda name: "ghp_" + "D" * 36)
    fake = _GitHubRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    listed = github_pages.list_repositories()
    access = github_pages.repository_access("test-owner", "site")
    assert listed["ok"] is True
    assert listed["repositories"][0]["repository"] == "test-owner/site"
    assert listed["repositories"][0]["writable"] is True
    assert access["ok"] is True
    assert access["readable"] is True
    assert access["writable"] is True


def test_short_github_approval_continues_the_pending_confirmation():
    history = [
        {"role": "user", "content": "publish the AI website"},
        {"role": "assistant", "content": (
            "The site is ready. Shall I publish it to GitHub Pages?"
        )},
    ]
    assert brain.ToolBrain._explicit_action_request(
        "deploy_github_pages", "Go ahead", history
    ) is True
    assert brain.ToolBrain._explicit_action_request(
        "deploy_github_pages", "mango and publish", history
    ) is False


def test_natural_spoken_github_approval_variants_are_bound_to_context():
    history = [
        {"role": "user", "content": "build an AI website"},
        {"role": "assistant", "content": (
            "The site is ready. Shall I publish it to GitHub Pages?"
        )},
    ]
    for phrase in ("go publish it", "yes publish it", "publish now",
                   "proceed with the deployment", "Yes, go ahead."):
        assert brain.ToolBrain._explicit_action_request(
            "deploy_github_pages", phrase, history
        ) is True
    assert brain.ToolBrain._explicit_action_request(
        "deploy_github_pages", "go publish it", []
    ) is False


def test_repeated_verified_publish_reuses_the_same_commit(tmp_path, monkeypatch):
    site = _site(tmp_path)
    monkeypatch.setattr(github_pages, "AGENT_ROOT", tmp_path / "agent_workspace")
    monkeypatch.setattr(
        github_pages, "_publish_state_path",
        lambda: tmp_path / "deployment" / "last_github_pages_publish.json",
    )
    monkeypatch.setattr(secure_config, "get", lambda name: "ghp_" + "C" * 36)
    fake = _GitHubRequests()
    monkeypatch.setattr(github_pages, "requests", fake)
    first = github_pages.deploy(
        str(site), "test-owner", "site", branch="main", confirm=True,
    )
    fake.calls.clear()
    second = github_pages.deploy(
        str(site), "test-owner", "site", branch="main", confirm=True,
    )
    assert first["ok"] is True
    assert second["ok"] is True
    assert second["idempotent"] is True
    assert second["commit"] == first["commit"]
    assert not any(url.endswith("/git/commits") for _, url, _ in fake.calls)


def test_direct_github_route_uses_configured_target_and_site(tmp_path, monkeypatch):
    project = tmp_path / "project"
    site = project / "agent_workspace" / "ai-website"
    site.mkdir(parents=True)
    (site / "index.html").write_text("<h1>AI</h1>", encoding="utf-8")
    monkeypatch.setattr(brain, "project_root", lambda: project)
    monkeypatch.setattr(brain, "load_config", lambda: {"github_pages": {
        "owner": "cfdumalaog",
        "repository": "Crizepvill-Dumalaog-AI-231",
        "branch": "main",
        "base_path": "",
        "pages_path": "/",
    }})
    history = [
        {"role": "user", "content": "build an AI website"},
        {"role": "assistant", "content": (
            "I found agent_workspace/ai-website. Shall I publish it to GitHub Pages?"
        )},
    ]
    name, args = brain._direct_github_pages_request(history, "Go ahead")
    assert name == "deploy_github_pages"
    assert args["site_dir"] == "agent_workspace/ai-website"
    assert args["owner"] == "cfdumalaog"
    assert args["repo"] == "Crizepvill-Dumalaog-AI-231"
    assert args["branch"] == "main"
    assert "confirm" not in args


def test_direct_github_route_accepts_an_explicit_non_default_repository(tmp_path, monkeypatch):
    project = tmp_path / "project"
    site = project / "agent_workspace" / "portfolio"
    site.mkdir(parents=True)
    (site / "index.html").write_text("<h1>Portfolio</h1>", encoding="utf-8")
    monkeypatch.setattr(brain, "project_root", lambda: project)
    monkeypatch.setattr(brain, "load_config", lambda: {"github_pages": {
        "owner": "test-owner", "repository": "configured-only",
    }})
    history = [
        {"role": "user", "content": "build a portfolio website"},
        {"role": "assistant", "content": (
            "The site is ready. Shall I publish it to GitHub Pages?"
        )},
    ]
    name, args = brain._direct_github_pages_request(
        history, "Deploy agent_workspace/portfolio to https://github.com/alice/other-site"
    )
    assert name == "deploy_github_pages"
    assert args["owner"] == "alice"
    assert args["repo"] == "other-site"
    assert "confirm" not in args


def test_short_minesweeper_publish_uses_latest_proposed_target(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for name in ("portfolio", "mind-reader", "minesweeper"):
        site = project / "agent_workspace" / name
        site.mkdir(parents=True)
        (site / "index.html").write_text(name, encoding="utf-8")
    monkeypatch.setattr(brain, "project_root", lambda: project)
    monkeypatch.setattr(brain, "load_config", lambda: {"github_pages": {
        "owner": "test-owner", "repository": "configured-only",
    }})
    history = [
        {"role": "user", "content": "build a mind reader game in agent_workspace/mind-reader"},
        {"role": "assistant", "content": "I created agent_workspace/mind-reader."},
        {"role": "user", "content": "build minesweeper in agent_workspace/minesweeper and use a new repository"},
        {"role": "assistant", "content": "I can create a new repository cfdumalaog/minesweeper. Shall I proceed with the publish?"},
    ]
    name, args = brain._direct_github_pages_request(history, "Yes, proceed.")
    assert name == "deploy_github_pages"
    assert args["site_dir"] == "agent_workspace/minesweeper"
    assert (args["owner"], args["repo"]) == ("cfdumalaog", "minesweeper")
    assert args["create_if_missing"] is True


def test_verified_github_result_is_not_replaced_by_unverified_model_prose(monkeypatch):
    monkeypatch.setenv("TEST_ME3_KEY", "test-key")
    cfg = {
        "provider": "qwen", "name": "qwen-tools", "model": "test-model",
        "base_url": "http://test", "api_key_env": "TEST_ME3_KEY",
        "system_prompt": "test", "allow_tools": True, "max_tool_steps": 3,
    }
    brain_instance = brain.ToolBrain(
        cfg, "test",
        dispatch_fn=lambda name, args: {
            "ok": True, "commit": "abcdef1234567890", "pages_url":
            "https://test-owner.github.io/site/",
        },
    )
    responses = iter([
        ({"content": "", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "deploy_github_pages", "arguments": "{}"},
        }]}, {"prompt_tokens": 1, "completion_tokens": 1}),
        ({"content": "The GitHub Pages site was deployed successfully."},
         {"prompt_tokens": 1, "completion_tokens": 1}),
    ])
    brain_instance._request = lambda messages, include_tools=True: next(responses)
    result = brain_instance.chat([], "publish the website to GitHub Pages")
    assert "cannot confirm" not in result.lower()
    assert "GitHub Pages site was deployed" in result


def test_unverified_github_completion_claim_is_blocked():
    assert brain._unverified_github_claim(
        "The repository was created and the GitHub Pages site is live at https://example.com"
    ) is True
    assert brain._unverified_github_claim(
        "I cannot confirm a GitHub deployment because the tool failed."
    ) is False


def test_github_setup_endpoint_is_loopback_only():
    st = LoopStatus()
    calls = []
    server = StatusServer(
        st, html_dir=Path(__file__).resolve().parents[1] / "web",
        get_state=lambda: "idle", get_stats=lambda: TurnStats(), port=0,
        github_status=lambda: {"ok": True, "configured": False},
        github_token_set=lambda token: calls.append(token) or {"ok": True},
    )
    assert server.start() is True
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + "/api/github/status", timeout=5) as response:
            assert json.loads(response.read())["configured"] is False
        request = urllib.request.Request(
            base + "/api/integrations/github-token",
            data=b'{"token":"ghp_test"}',
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["ok"] is True
        assert calls == ["ghp_test"]
    finally:
        server.stop()
