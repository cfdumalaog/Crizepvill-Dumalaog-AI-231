"""Small, deterministic software-task checks for the voice tool loop.

These checks establish that generated source can be parsed and that a static
site's local assets exist. They do not replace project-specific tests or a
human review of behavior.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


_ACTION = re.compile(
    r"\b(?:build|create|make|develop|implement|code|fix|debug|refactor|"
    r"update|modify|add|write|repair|test|gumawa|gawa|bumuo|ayusin|"
    r"dagdagan|palitan|i-?fix|i-?build|i-?update)\b", re.IGNORECASE,
)
_SOFTWARE = re.compile(
    r"\b(?:web\s?app|website|web\s?page|app|software|code|program|script|"
    r"repository|repo|bug|feature|function|project|tests?|aplikasyon|"
    r"programa|sistema)\b", re.IGNORECASE,
)
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".tasks", "dist", "build"}
_SKIP_FILES = {".env", "credentials.json", "secrets.json", "id_rsa"}
_SKIP_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def is_software_request(text: str) -> bool:
    """Select actionable development requests, not general software questions."""
    value = str(text or "").strip()
    if re.match(r"^(?:what|why|how|explain|tell me|paano|ano|bakit|ipaliwanag)\b",
                value, re.IGNORECASE):
        return False
    return bool(_ACTION.search(value) and _SOFTWARE.search(value))


def verification_root(path: str, *, is_directory: bool = False) -> str:
    """Prefer an app root under agent_workspace to a nested source directory."""
    candidate = Path(str(path or "agent_workspace"))
    parts = candidate.parts
    lowered = [part.lower() for part in parts]
    if "agent_workspace" in lowered:
        index = lowered.index("agent_workspace")
        if len(parts) > index + 1:
            if len(parts) == index + 2 and Path(parts[-1]).suffix:
                return candidate.as_posix()
            return Path(*parts[:index + 2]).as_posix()
    return candidate.as_posix()


class _Assets(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []
        self.inline_scripts: list[tuple[str, str]] = []
        self._script_type: str | None = None
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and not values.get("src"):
            self._script_type = str(values.get("type") or "").lower()
            self._script_parts = []
        for attr in (("src",) if tag in {"script", "img", "source", "video", "audio"}
                     else ("href",) if tag == "link" else ()):
            value = values.get(attr)
            if value:
                self.references.append(value)

    def handle_data(self, data: str) -> None:
        if self._script_type is not None:
            self._script_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_type is not None:
            self.inline_scripts.append((self._script_type, "".join(self._script_parts)))
            self._script_type = None
            self._script_parts = []


def verify_project(root: Path, max_files: int = 300) -> dict:
    """Check source syntax and static-site assets without executing the app."""
    root = root.resolve()
    single_file = root.is_file()
    if not single_file and not root.is_dir():
        return {"ok": False, "path": str(root), "error": "path_not_found"}
    base = root.parent if single_file else root
    max_files = max(1, min(int(max_files), 1000))
    files: list[Path] = []
    errors: list[str] = []
    truncated = False
    if single_file:
        if root.name.lower() not in _SKIP_FILES and root.suffix.lower() not in _SKIP_SUFFIXES:
            files = [root]
    else:
        for folder, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in _SKIP_DIRS]
            for name in names:
                path = Path(folder) / name
                if (name.lower() in _SKIP_FILES or path.suffix.lower() in _SKIP_SUFFIXES
                        or path.is_symlink() or not path.is_file()):
                    continue
                files.append(path)
                if len(files) > max_files:
                    files.pop()
                    truncated = True
                    break
            if truncated:
                break
    checked = {"python": 0, "json": 0, "javascript": 0, "html": 0}
    skipped: list[str] = []
    node = shutil.which("node")
    html_files: list[Path] = []
    for path in files:
        relative = path.relative_to(base).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix == ".py":
                compile(path.read_bytes(), str(path), "exec")
                checked["python"] += 1
            elif suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                checked["json"] += 1
            elif suffix in {".js", ".mjs", ".cjs"}:
                if node:
                    result = subprocess.run(
                        [node, "--check", str(path)], capture_output=True,
                        text=True, timeout=10, errors="replace",
                    )
                    checked["javascript"] += 1
                    if result.returncode:
                        errors.append(f"{relative}: JavaScript syntax check failed: "
                                      f"{result.stderr.strip()[-350:]}")
                elif "node_missing" not in skipped:
                    skipped.append("node_missing")
            elif suffix in {".html", ".htm"}:
                html_files.append(path)
                checked["html"] += 1
        except (OSError, UnicodeError, SyntaxError, ValueError,
                subprocess.TimeoutExpired) as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {str(exc)[-350:]}")
        if len(errors) >= 20:
            break
    if html_files and not single_file and not (root / "index.html").is_file():
        errors.append("static site has HTML files but no index.html at its root")
    for path in html_files:
        relative = path.relative_to(base).as_posix()
        try:
            parser = _Assets()
            parser.feed(path.read_text(encoding="utf-8"))
            for script_type, source in parser.inline_scripts:
                if not source.strip() or script_type not in {
                        "", "text/javascript", "application/javascript", "module"}:
                    continue
                if not node:
                    if "node_missing" not in skipped:
                        skipped.append("node_missing")
                    continue
                argv = [node]
                if script_type == "module":
                    argv.append("--input-type=module")
                result = subprocess.run(
                    [*argv, "--check", "-"], input=source,
                    capture_output=True, text=True, timeout=10, errors="replace",
                )
                checked["javascript"] += 1
                if result.returncode:
                    errors.append(f"{relative}: inline JavaScript syntax check failed: "
                                  f"{result.stderr.strip()[-350:]}")
            for ref in parser.references:
                parsed = urlsplit(ref)
                if parsed.scheme or parsed.netloc or ref.startswith(("#", "//", "data:")):
                    continue
                asset = unquote(parsed.path)
                if not asset:
                    continue
                target = (base / asset.lstrip("/") if asset.startswith("/")
                          else path.parent / asset).resolve()
                if not target.is_relative_to(base) or not target.exists():
                    errors.append(f"{relative}: missing local asset {asset}")
                    if len(errors) >= 20:
                        break
        except (OSError, UnicodeError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{relative}: HTML asset check failed: {type(exc).__name__}")
        if len(errors) >= 20:
            break
    if not files:
        errors.append("project directory contains no files")
    elif not any(checked.values()):
        errors.append("no supported source files were checked")
    return {
        "ok": not errors and not truncated,
        "path": str(root),
        "files_scanned": len(files),
        "checks": checked,
        "skipped": skipped,
        "truncated": truncated,
        "errors": errors[:20],
        "scope": "syntax and local static assets only; run project tests separately",
    }
