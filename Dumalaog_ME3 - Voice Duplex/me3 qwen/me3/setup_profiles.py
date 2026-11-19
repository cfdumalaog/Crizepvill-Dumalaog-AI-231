"""Adaptive, private setup profiles for additional model/API providers.

The quick-start buttons in the UI cover the known ME3 integrations.  This
module handles the open-ended case: a user describes another OpenAI-compatible
or vLLM service and ME3 returns a structured form for the requested role.
Profiles are stored beside the protected credential file with mode 0600; the
status path never returns the API key.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from . import secure_config


ROLES = ("llm", "asr", "tts")
_SAFE_NAME = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def _path() -> Path:
    return secure_config._path().with_name("provider_profiles.json")


def _read() -> dict:
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value.setdefault("profiles", {})
            value.setdefault("active", {})
            return value
    except (OSError, ValueError, TypeError):
        pass
    return {"profiles": {}, "active": {}}


def _write(value: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.name == "nt":
        temporary.write_text(json.dumps(value), encoding="utf-8")
    else:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name != "nt":
        path.chmod(0o600)


def _slug(value: str) -> str:
    result = _SAFE_NAME.sub("-", value.strip().lower()).strip("-")
    return (result or "custom-provider")[:48]


def _role_hint(request: str) -> str:
    text = str(request or "").lower()
    if re.search(r"\b(?:tts|text[- ]to[- ]speech|synth|voice output|speech output)\b", text):
        return "tts"
    if re.search(r"\b(?:asr|stt|speech[- ]to[- ]text|transcri|recognition|listen)\b", text):
        return "asr"
    return "llm"


def setup_plan(request: str) -> dict:
    """Create a form schema from a natural-language provider request.

    The plan is intentionally deterministic and secret-free.  The model can
    describe the request in any language; the resulting fields are still
    validated by the private save endpoint before activation.
    """
    role = _role_hint(request)
    protocol_options = {
        "llm": ["OpenAI-compatible chat completions"],
        "asr": ["Qwen/vLLM audio chat", "OpenAI-compatible audio transcriptions"],
        "tts": ["Qwen/vLLM audio speech"],
    }[role]
    fields = [
        {"name": "profile_name", "label": "Provider profile name", "type": "text",
         "required": True, "default": "custom-vllm"},
        {"name": "role", "label": "Runtime role", "type": "select", "required": True,
         "options": [{"value": item, "label": item.upper()} for item in ROLES],
         "default": role},
        {"name": "base_url", "label": "vLLM/API base URL", "type": "url",
         "required": True, "placeholder": "https://host.example/v1"},
        {"name": "model", "label": "Model name", "type": "text", "required": True,
         "placeholder": "organization/model-name"},
        {"name": "api_key", "label": "API key (stored privately)", "type": "password",
         "required": True, "autocomplete": "new-password"},
        {"name": "protocol", "label": "API protocol", "type": "select", "required": True,
         "options": [{"value": "qwen_audio_chat", "label": "Qwen/vLLM audio chat"},
                     {"value": "audio_transcriptions", "label": "OpenAI-compatible audio transcriptions"},
                     {"value": "chat_completions", "label": "OpenAI-compatible chat completions"},
                     {"value": "audio_speech", "label": "Qwen/vLLM audio speech"}],
         "default": ("chat_completions" if role == "llm" else
                     "qwen_audio_chat" if role == "asr" else "audio_speech")},
        {"name": "voice", "label": "Voice name (TTS only, optional)", "type": "text",
         "placeholder": "aiden"},
        {"name": "fallback", "label": "Fallback if unavailable", "type": "select",
         "options": [{"value": "qwen", "label": "ME3 Qwen"},
                     {"value": "deepgram", "label": "Deepgram (ASR)"},
                     {"value": "piper", "label": "Piper (TTS)"},
                     {"value": "none", "label": "No fallback"}],
         "default": "none"},
    ]
    return {
        "ok": True,
        "title": "Set up a custom vLLM/API provider",
        "help": ("ME3 generated these fields from your request. The API key "
                 "goes only to the protected Pi setup endpoint; it never enters "
                 "the conversation or model prompt."),
        "inferred_role": role,
        "fields": fields,
    }


def _public(profile: dict) -> dict:
    return {
        "profile_id": profile.get("profile_id", ""),
        "profile_name": profile.get("profile_name", ""),
        "role": profile.get("role", ""),
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
        "protocol": profile.get("protocol", ""),
        "voice": profile.get("voice", ""),
        "fallback": profile.get("fallback", "none"),
        "configured": bool(profile.get("api_key")),
    }


def save_profile(payload: dict) -> dict:
    """Validate and persist one custom profile without returning its secret."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "provider_profile_must_be_object"}
    name = str(payload.get("profile_name") or "").strip()
    role = str(payload.get("role") or "").strip().lower()
    base_url = str(payload.get("base_url") or "").strip().rstrip("/")
    model = str(payload.get("model") or "").strip()
    api_key = str(payload.get("api_key") or "").strip()
    protocol = str(payload.get("protocol") or "").strip().lower()
    voice = str(payload.get("voice") or "").strip()
    fallback = str(payload.get("fallback") or "none").strip().lower()
    if not name or len(name) > 80:
        return {"ok": False, "error": "profile_name_required"}
    if role not in ROLES:
        return {"ok": False, "error": "unsupported_provider_role"}
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"ok": False, "error": "base_url_must_be_http_or_https"}
    if not model or len(model) > 256:
        return {"ok": False, "error": "model_required"}
    if not api_key or len(api_key) > 4096 or "\n" in api_key or "\r" in api_key:
        return {"ok": False, "error": "api_key_required"}
    allowed_protocols = {"qwen_audio_chat", "audio_transcriptions",
                        "chat_completions", "audio_speech"}
    if protocol not in allowed_protocols:
        return {"ok": False, "error": "unsupported_api_protocol"}
    if fallback not in {"qwen", "deepgram", "piper", "none"}:
        return {"ok": False, "error": "unsupported_fallback"}
    if role == "llm" and protocol != "chat_completions":
        return {"ok": False, "error": "llm_requires_chat_completions"}
    if role == "tts" and protocol != "audio_speech":
        return {"ok": False, "error": "tts_requires_audio_speech"}
    profile_id = _slug(name)
    value = _read()
    profile = {
        "profile_id": profile_id, "profile_name": name, "role": role,
        "base_url": base_url, "model": model, "api_key": api_key,
        "protocol": protocol, "voice": voice or "aiden",
        "fallback": fallback,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    value.setdefault("profiles", {})[profile_id] = profile
    _write(value)
    return {"ok": True, "profile": _public(profile), "activate_profile_id": profile_id,
            "restart_required": True}


def activate_profile(profile_id: str) -> dict:
    profile_id = str(profile_id or "").strip()
    value = _read()
    profile = (value.get("profiles") or {}).get(profile_id)
    if not isinstance(profile, dict) or not profile.get("api_key"):
        return {"ok": False, "error": "provider_profile_not_found"}
    value.setdefault("active", {})[profile["role"]] = profile_id
    _write(value)
    return {"ok": True, "profile": _public(profile), "restart_required": True}


def active_profile(role: str) -> dict | None:
    value = _read()
    profile_id = (value.get("active") or {}).get(str(role or "").lower())
    profile = (value.get("profiles") or {}).get(profile_id)
    return profile if isinstance(profile, dict) else None


def profile_config(role: str) -> dict:
    profile = active_profile(role)
    if not profile:
        return {}
    return {
        "base_url": profile.get("base_url", ""),
        "model": profile.get("model", ""),
        "api_key": profile.get("api_key", ""),
        "protocol": profile.get("protocol", ""),
        "voice": profile.get("voice", "aiden"),
        "fallback_engine": profile.get("fallback", "none"),
        "provider": "custom_vllm",
    }


def status() -> dict:
    value = _read()
    profiles = value.get("profiles") or {}
    active = value.get("active") or {}
    return {
        "ok": True,
        "profiles": [_public(item) for item in profiles.values() if isinstance(item, dict)],
        "active": {role: _public(profiles[profile_id])
                   for role, profile_id in active.items()
                   if profile_id in profiles and isinstance(profiles[profile_id], dict)},
    }
