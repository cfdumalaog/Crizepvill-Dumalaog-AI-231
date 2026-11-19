"""Secure GitHub Pages publishing for the ME3 agent.

The voice agent may build a site inside ``agent_workspace`` and, after an
explicit confirmation, publish it to a selected GitHub repository.  The
publisher uses GitHub's REST Git-data API rather than a shell ``git push``:
the token stays in the Pi's private credential store, repository files are
committed atomically, and no credential or response body is returned to the
voice/UI event stream.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlencode

from . import secure_config
from .config import project_root

try:  # imported lazily in tests and on the Pi deployment
    import requests
except ImportError:  # pragma: no cover - requirements include requests
    requests = None  # type: ignore[assignment]


PROJECT = project_root()
AGENT_ROOT = (PROJECT / "agent_workspace").resolve()
API_ROOT = "https://api.github.com"
_OWNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9_./-]{1,200}$")
_PROTECTED_PARTS = {
    ".git", ".venv", ".tasks", "credentials", "secrets", "models",
    "__pycache__", "node_modules",
}
_ALLOWED_PAGES_PATHS = {"/", "/docs"}
_MAX_FILES = 160
_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_TOTAL_BYTES = 20 * 1024 * 1024


def _publish_state_path() -> Path:
    """Return the local, non-secret record of the last verified publish."""
    return PROJECT / "deployment" / "last_github_pages_publish.json"


def _files_digest(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for remote_path, data in files:
        digest.update(remote_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _read_publish_state() -> dict[str, Any] | None:
    try:
        payload = json.loads(_publish_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_publish_state(payload: dict[str, Any]) -> None:
    path = _publish_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ME3-GitHub-Pages/1.0",
    }


def _safe_api_error(status: int, path: str) -> str:
    if status == 401:
        return "github_token_rejected"
    if status == 403:
        return "github_forbidden_or_rate_limited"
    if status == 404:
        return "github_resource_not_found"
    if status == 409:
        return "github_conflict"
    if status == 422:
        return "github_validation_failed"
    return f"github_api_http_{status}"


def _request(method: str, path: str, token: str,
             payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if requests is None:
        return {"ok": False, "error": "requests_dependency_missing"}
    url = API_ROOT + path
    try:
        response = requests.request(
            method, url, headers=_headers(token), json=payload, timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 - never include token-bearing data
        return {"ok": False, "error": "github_network_error",
                "error_type": type(exc).__name__}
    status = int(response.status_code)
    if status < 200 or status >= 300:
        return {"ok": False, "status": status,
                "error": _safe_api_error(status, path)}
    try:
        data = response.json()
    except ValueError:
        data = {}
    return {"ok": True, "status": status, "data": data,
            "headers": {"etag": response.headers.get("ETag", "")}}


def _token() -> str:
    return secure_config.get("GITHUB_TOKEN")


def health(check: bool = False) -> dict[str, Any]:
    """Report GitHub configuration without exposing the credential.

    ``check=True`` performs a harmless ``GET /user`` validation.  The normal
    UI poll does not contact GitHub repeatedly.
    """
    token = _token()
    result: dict[str, Any] = {
        "ok": True,
        "provider": "github",
        "configured": bool(token),
        "missing": [] if token else ["GITHUB_TOKEN"],
    }
    if check and token:
        response = _request("GET", "/user", token)
        if not response.get("ok"):
            return {**result, "ok": False, "configured": False,
                    "error": response.get("error", "github_token_check_failed")}
        user = response.get("data") or {}
        result.update({"validated": True, "login": str(user.get("login") or "")})
    return result


def _permissions(data: dict[str, Any]) -> dict[str, bool]:
    """Return only the non-sensitive GitHub permission flags."""
    raw = data.get("permissions") or {}
    return {
        key: bool(raw.get(key))
        for key in ("pull", "triage", "push", "maintain", "admin")
        if key in raw
    }


def list_repositories(max_repositories: int = 200) -> dict[str, Any]:
    """List repositories visible to the configured GitHub credential.

    This is deliberately an API-backed discovery tool rather than a local
    allowlist. The credential's GitHub permissions decide which personal and
    organization repositories are returned.
    """
    token = _token()
    if not token:
        return {"ok": False, "error": "github_not_configured",
                "missing": ["GITHUB_TOKEN"]}
    try:
        limit = max(1, min(int(max_repositories), 500))
    except (TypeError, ValueError):
        limit = 200
    repositories: list[dict[str, Any]] = []
    page = 1
    while len(repositories) < limit and page <= 5:
        per_page = min(100, limit - len(repositories))
        query = urlencode({
            "affiliation": "owner,collaborator,organization_member",
            "per_page": per_page,
            "page": page,
            "sort": "updated",
        })
        response = _request("GET", f"/user/repos?{query}", token)
        if not response.get("ok"):
            return {"ok": False, "error": response.get("error", "github_repository_list_failed")}
        data = response.get("data") or []
        if not isinstance(data, list):
            return {"ok": False, "error": "github_repository_list_invalid"}
        for item in data:
            if not isinstance(item, dict):
                continue
            full_name = str(item.get("full_name") or "").strip()
            if not full_name:
                continue
            permissions = _permissions(item)
            repositories.append({
                "repository": full_name,
                "private": bool(item.get("private")),
                "default_branch": str(item.get("default_branch") or "main"),
                "permissions": permissions,
                "writable": bool(permissions.get("push") or
                                  permissions.get("maintain") or
                                  permissions.get("admin")),
                "html_url": str(item.get("html_url") or ""),
            })
            if len(repositories) >= limit:
                break
        if len(data) < per_page:
            break
        page += 1
    return {"ok": True, "repositories": repositories,
            "count": len(repositories), "truncated": len(repositories) >= limit}


def repository_access(owner: str, repo: str) -> dict[str, Any]:
    """Check read/write access to one explicitly named repository."""
    owner = str(owner or "").strip()
    repo = str(repo or "").strip()
    if not _OWNER_RE.fullmatch(owner) or not _REPO_RE.fullmatch(repo):
        return {"ok": False, "error": "invalid GitHub repository target"}
    token = _token()
    if not token:
        return {"ok": False, "error": "github_not_configured",
                "missing": ["GITHUB_TOKEN"]}
    response = _request("GET", f"/repos/{owner}/{repo}", token)
    if not response.get("ok"):
        return {"ok": False, "error": response.get("error", "github_repository_access_failed"),
                "repository": f"{owner}/{repo}"}
    data = response.get("data") or {}
    if not isinstance(data, dict):
        return {"ok": False, "error": "github_repository_access_invalid",
                "repository": f"{owner}/{repo}"}
    permissions = _permissions(data)
    writable = bool(permissions.get("push") or permissions.get("maintain") or
                    permissions.get("admin"))
    return {
        "ok": True,
        "repository": str(data.get("full_name") or f"{owner}/{repo}"),
        "private": bool(data.get("private")),
        "default_branch": str(data.get("default_branch") or "main"),
        "permissions": permissions,
        "readable": True,
        "writable": writable,
        "html_url": str(data.get("html_url") or ""),
        "pages_url": _pages_url(owner, repo, ""),
    }


def create_repository(owner: str, repo: str, *, public: bool = True) -> dict[str, Any]:
    """Create a requested repository with an initialized main branch.

    GitHub decides whether the private credential may create repositories.
    An existing repository is never replaced or modified by this function.
    """
    if not _OWNER_RE.fullmatch(str(owner or "")) or not _REPO_RE.fullmatch(str(repo or "")):
        return {"ok": False, "error": "invalid GitHub repository target"}
    token = _token()
    if not token:
        return {"ok": False, "error": "github_not_configured"}
    existing = repository_access(owner, repo)
    if existing.get("ok"):
        return {**existing, "created": False}
    if existing.get("error") != "github_resource_not_found":
        return existing
    account = _request("GET", "/user", token)
    if not account.get("ok"):
        return {"ok": False, "error": account.get("error", "github_account_check_failed")}
    login = str((account.get("data") or {}).get("login") or "")
    endpoint = "/user/repos" if login.casefold() == owner.casefold() else f"/orgs/{owner}/repos"
    created = _request("POST", endpoint, token, {
        "name": repo, "private": not public, "auto_init": True,
    })
    if not created.get("ok"):
        return {"ok": False, "error": created.get("error", "github_repository_create_failed"),
                "repository": f"{owner}/{repo}"}
    data = created.get("data") or {}
    return {"ok": True, "created": True,
            "repository": str(data.get("full_name") or f"{owner}/{repo}"),
            "default_branch": str(data.get("default_branch") or "main"),
            "private": bool(data.get("private", not public)),
            "html_url": str(data.get("html_url") or f"https://github.com/{owner}/{repo}")}


def configure_token(token: str) -> dict[str, Any]:
    """Validate then privately save a GitHub token from the loopback UI.

    The token is never returned, logged, placed in a command line, or written
    to the repository.  A connected GitHub account in the desktop app cannot
    be copied into the Pi; the operator must enter the token through the
    loopback-only ME3 setup page or provide it through the Pi service
    environment outside this chat.
    """
    value = str(token or "").strip()
    if not value or len(value) > 512 or "\n" in value or "\r" in value:
        return {"ok": False, "error": "invalid_github_token_format"}
    if not re.fullmatch(r"(?:ghp_|github_pat_)[A-Za-z0-9_]+", value):
        return {"ok": False, "error": "invalid_github_token_format"}
    checked = _request("GET", "/user", value)
    if not checked.get("ok"):
        return {"ok": False,
                "error": checked.get("error", "github_token_rejected")}
    user = checked.get("data") or {}
    saved = secure_config.set_values({"GITHUB_TOKEN": value})
    if not saved.get("ok"):
        return {"ok": False, "error": "github_token_storage_failed"}
    return {"ok": True, "configured": True,
            "login": str(user.get("login") or "")}


def _clean_path(value: str, *, allow_empty: bool = True) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text and allow_empty:
        return ""
    if not text or text.startswith("/") or "//" in text:
        raise ValueError("invalid GitHub Pages path")
    parts = [p for p in PurePosixPath(text).parts if p not in {"", "."}]
    if any(p == ".." or "\x00" in p for p in parts):
        raise ValueError("GitHub Pages path escapes its target")
    return "/".join(parts)


def _validate_target(owner: str, repo: str, branch: str,
                    base_path: str, pages_path: str) -> tuple[str, str, str, str, str]:
    owner = str(owner or "").strip()
    repo = str(repo or "").strip()
    branch = str(branch or "main").strip()
    if not _OWNER_RE.fullmatch(owner):
        raise ValueError("invalid GitHub owner")
    if not _REPO_RE.fullmatch(repo):
        raise ValueError("invalid GitHub repository")
    if not _BRANCH_RE.fullmatch(branch) or branch.startswith("/") or ".." in branch:
        raise ValueError("invalid GitHub branch")
    base_path = _clean_path(base_path)
    pages_path = "/" + _clean_path(pages_path).strip("/") if pages_path != "/" else "/"
    if pages_path not in _ALLOWED_PAGES_PATHS:
        raise ValueError("pages_path must be / or /docs")
    if (pages_path == "/") != (base_path == ""):
        raise ValueError("base_path must match the GitHub Pages source path")
    return owner, repo, branch, base_path, pages_path


def _site_files(site_dir: str, base_path: str) -> list[tuple[str, bytes]]:
    raw = Path(str(site_dir or "agent_workspace")).expanduser()
    candidate = (PROJECT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not candidate.is_relative_to(AGENT_ROOT):
        raise ValueError("site_dir must remain inside agent_workspace")
    if not candidate.is_dir():
        raise ValueError("site_dir is not a directory")
    files: list[tuple[str, bytes]] = []
    total = 0
    for path in sorted(candidate.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(AGENT_ROOT):
            raise ValueError("site contains a path outside agent_workspace")
        rel_parts = {p.lower() for p in path.relative_to(candidate).parts}
        if rel_parts & _PROTECTED_PARTS:
            continue
        data = path.read_bytes()
        if len(data) > _MAX_FILE_BYTES:
            raise ValueError(f"site file is too large: {path.name}")
        total += len(data)
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("site is larger than the ME3 upload limit")
        rel = "/".join(path.relative_to(candidate).parts)
        remote = f"{base_path}/{rel}" if base_path else rel
        files.append((remote, data))
        if len(files) > _MAX_FILES:
            raise ValueError("site has too many files for one ME3 deployment")
    expected_index = f"{base_path + '/' if base_path else ''}index.html"
    if not any(remote == expected_index for remote, _data in files):
        raise ValueError("site must contain index.html")
    return files


def _commit_site(owner: str, repo: str, branch: str,
                 files: list[tuple[str, bytes]], message: str,
                 token: str) -> dict[str, Any]:
    prefix = f"/repos/{owner}/{repo}"
    ref = _request("GET", f"{prefix}/git/ref/heads/{branch}", token)
    if not ref.get("ok"):
        return ref
    head = str(((ref.get("data") or {}).get("object") or {}).get("sha") or "")
    if not head:
        return {"ok": False, "error": "github_branch_head_missing"}
    commit = _request("GET", f"{prefix}/git/commits/{head}", token)
    if not commit.get("ok"):
        return commit
    base_tree = str(((commit.get("data") or {}).get("tree") or {}).get("sha") or "")
    if not base_tree:
        return {"ok": False, "error": "github_base_tree_missing"}
    entries = []
    for remote_path, data in files:
        blob = _request("POST", f"{prefix}/git/blobs", token, {
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
        })
        if not blob.get("ok"):
            return blob
        blob_sha = str((blob.get("data") or {}).get("sha") or "")
        if not blob_sha:
            return {"ok": False, "error": "github_blob_sha_missing"}
        entries.append({"path": remote_path, "mode": "100644",
                        "type": "blob", "sha": blob_sha})
    tree = _request("POST", f"{prefix}/git/trees", token, {
        "base_tree": base_tree, "tree": entries,
    })
    if not tree.get("ok"):
        return tree
    tree_sha = str((tree.get("data") or {}).get("sha") or "")
    new_commit = _request("POST", f"{prefix}/git/commits", token, {
        "message": message, "tree": tree_sha, "parents": [head],
    })
    if not new_commit.get("ok"):
        return new_commit
    new_sha = str((new_commit.get("data") or {}).get("sha") or "")
    updated = _request("PATCH", f"{prefix}/git/refs/heads/{branch}", token, {
        "sha": new_sha, "force": False,
    })
    if not updated.get("ok"):
        return updated
    return {"ok": True, "commit": new_sha, "files": len(files)}


def _pages_url(owner: str, repo: str, base_path: str) -> str:
    suffix = f"/{base_path}" if base_path else ""
    if repo.lower() == f"{owner}.github.io".lower():
        return f"https://{owner}.github.io{suffix}/"
    return f"https://{owner}.github.io/{repo}{suffix}/"


def _ensure_pages(owner: str, repo: str, branch: str, pages_path: str,
                  token: str, enable_pages: bool) -> dict[str, Any]:
    prefix = f"/repos/{owner}/{repo}"
    current = _request("GET", f"{prefix}/pages", token)
    if current.get("ok"):
        data = current.get("data") or {}
        return {"ok": True, "status": "configured",
                "url": str(data.get("html_url") or "")}
    if current.get("status") != 404:
        return current
    if not enable_pages:
        return {"ok": True, "status": "not_enabled", "url": ""}
    created = _request("POST", f"{prefix}/pages", token, {
        "source": {"branch": branch, "path": pages_path},
    })
    if created.get("ok"):
        data = created.get("data") or {}
        return {"ok": True, "status": "enabled",
                "url": str(data.get("html_url") or "")}
    if created.get("status") == 409:
        current = _request("GET", f"{prefix}/pages", token)
        if current.get("ok"):
            data = current.get("data") or {}
            return {"ok": True, "status": "configured",
                    "url": str(data.get("html_url") or "")}
    return created


def deploy(site_dir: str, owner: str, repo: str, branch: str = "main",
           base_path: str = "", pages_path: str = "/",
           message: str = "Deploy GitHub Pages site from ME3",
           enable_pages: bool = True, confirm: bool = False,
           create_if_missing: bool = False) -> dict[str, Any]:
    """Publish a validated site to the requested repository.

    ``confirm`` is retained for older callers but is not a permission gate.
    The voice agent routes only a user-requested publication here.
    """
    try:
        owner, repo, branch, base_path, pages_path = _validate_target(
            owner, repo, branch, base_path, pages_path)
        message = " ".join(str(message or "").split())[:120] or "Deploy GitHub Pages site from ME3"
        files = _site_files(site_dir, base_path)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    token = _token()
    if not token:
        return {"ok": False, "error": "github_not_configured",
                "missing": ["GITHUB_TOKEN"], "provider_called": False}
    access = repository_access(owner, repo)
    created_repository = False
    if (not access.get("ok") and access.get("error") == "github_resource_not_found"
            and create_if_missing):
        created = create_repository(owner, repo, public=True)
        if not created.get("ok"):
            return {"ok": False, "error": created.get("error", "github_repository_create_failed"),
                    "provider_called": True, "repository": f"{owner}/{repo}"}
        created_repository = bool(created.get("created"))
        access = repository_access(owner, repo)
    if not access.get("ok"):
        return {"ok": False,
                "error": access.get("error", "github_repository_access_failed"),
                "provider_called": True, "repository": f"{owner}/{repo}",
                "branch": branch, "pages_url": _pages_url(owner, repo, base_path)}
    if access.get("writable") is False:
        return {"ok": False, "error": "github_repository_not_writable",
                "provider_called": True, "repository": f"{owner}/{repo}",
                "branch": branch, "access": access,
                "pages_url": _pages_url(owner, repo, base_path)}
    content_sha256 = _files_digest(files)
    previous = _read_publish_state()
    same_release = bool(
        previous
        and previous.get("repository") == f"{owner}/{repo}"
        and previous.get("branch") == branch
        and previous.get("base_path") == base_path
        and previous.get("content_sha256") == content_sha256
        and previous.get("commit")
        and previous.get("pages_url")
    )
    if same_release:
        # Verify that Pages is still enabled, but do not create another commit
        # every time the user repeats an approval or an echoed confirmation.
        pages = _ensure_pages(owner, repo, branch, pages_path, token, enable_pages)
        if not pages.get("ok"):
            return {"ok": False,
                    "error": pages.get("error", "github_pages_setup_failed"),
                    "provider_called": True,
                    "repository": f"{owner}/{repo}", "branch": branch,
                    "pages_url": _pages_url(owner, repo, base_path)}
        return {
            "ok": True,
            "repository": f"{owner}/{repo}",
            "branch": branch,
            "commit": previous.get("commit"),
            "files": int(previous.get("files") or len(files)),
            "pages": pages.get("status") or previous.get("pages", "configured"),
            "pages_url": str(previous.get("pages_url") or
                               pages.get("url") or
                               _pages_url(owner, repo, base_path)),
            "idempotent": True,
            "note": "The verified GitHub Pages result was reused; no duplicate commit was created.",
        }
    committed = _commit_site(owner, repo, branch, files, message, token)
    if not committed.get("ok"):
        return {"ok": False, "error": committed.get("error", "github_commit_failed"),
                "provider_called": True, "repository": f"{owner}/{repo}"}
    pages = _ensure_pages(owner, repo, branch, pages_path, token, enable_pages)
    fallback_url = _pages_url(owner, repo, base_path)
    if not pages.get("ok"):
        return {"ok": False, "partial_commit": True,
                "error": pages.get("error", "github_pages_setup_failed"),
                "commit": committed.get("commit"), "files": len(files),
                "repository": f"{owner}/{repo}", "branch": branch,
                "pages_url": fallback_url}
    pages_url = str(pages.get("url") or fallback_url)
    result = {"ok": True, "repository": f"{owner}/{repo}", "branch": branch,
              "commit": committed.get("commit"), "files": len(files),
              "pages": pages.get("status"), "pages_url": pages_url,
              "created_repository": created_repository,
              "note": "GitHub Pages may take a short time to build the new commit."}
    try:
        _write_publish_state({
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "repository": f"{owner}/{repo}", "branch": branch,
            "base_path": base_path, "pages_path": pages_path,
            "commit": committed.get("commit"), "files": len(files),
            "content_sha256": content_sha256, "pages": pages.get("status"),
            "pages_url": pages_url,
        })
    except OSError:
        # The remote commit and Pages result are still authoritative. A local
        # cache failure must not turn a verified external publish into a false
        # failure response.
        pass
    return result

