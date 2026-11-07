"""Small protected file store for provider credentials on the ME3 host.

Secrets are written only by the loopback-only settings endpoint. On the Pi the
default directory is private to the service user and the file is mode 0600.
Environment variables override stored values for managed deployments.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


_CHAT_KEY = re.compile(r"^ME3_TELEGRAM_CHAT_[A-Z0-9_]{1,64}$")
_ALLOWED = {
    "TELEGRAM_BOT_TOKEN", "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_CALENDAR_REFRESH_TOKEN",
}


def _path() -> Path:
    configured = os.getenv("ME3_CREDENTIALS_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.getenv("APPDATA", str(Path.home()))) / "ME3"
    else:
        root = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "me3"
    return root / "credentials.json"


def _read() -> dict[str, str]:
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items()
                if isinstance(k, str) and isinstance(v, (str, int, float))}
    except (OSError, ValueError, TypeError):
        return {}


def get(name: str) -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    return _read().get(name, "").strip()


def configured(name: str) -> bool:
    return bool(get(name))


def set_values(values: dict) -> dict:
    if not isinstance(values, dict) or not values:
        return {"ok": False, "error": "credential_values_required"}
    accepted: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name or "").strip()
        value = str(raw_value or "").strip()
        if not name or (name not in _ALLOWED and not _CHAT_KEY.fullmatch(name)):
            return {"ok": False, "error": "unsupported_credential_name"}
        if not value:
            return {"ok": False, "error": "credential_value_required"}
        if len(value) > 4096 or "\n" in value or "\r" in value:
            return {"ok": False, "error": "invalid_credential_value"}
        if name == "TELEGRAM_BOT_TOKEN" and not re.fullmatch(r"\d{5,16}:[A-Za-z0-9_-]{20,256}", value):
            return {"ok": False, "error": "invalid_telegram_bot_token_format"}
        if name.startswith("ME3_TELEGRAM_CHAT_") and not re.fullmatch(r"-?\d{1,24}", value):
            return {"ok": False, "error": "telegram_chat_id_must_be_numeric"}
        accepted[name] = value

    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = _read()
    saved.update(accepted)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.name == "nt":
        temporary.write_text(json.dumps(saved), encoding="utf-8")
    else:
        path.parent.chmod(0o700)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(saved, stream)
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name != "nt":
        path.chmod(0o600)
    return {"ok": True, "saved": sorted(accepted), "storage": "private_pi_file"}
