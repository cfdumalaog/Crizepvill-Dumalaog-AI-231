"""Persist the independently selectable ME3 audio-layer backends.

The complete voice stack (Qwen or OpenAI) remains the high-level provider
choice.  This module lets an operator deliberately override ASR, TTS, or VAD
without placing credentials in the checked-in config or in chat history.
"""
from __future__ import annotations

import json
import os

from . import secure_config


LAYER_OPTIONS = {
    "asr": ("qwen_remote", "deepgram", "faster_whisper", "openai", "custom_vllm"),
    "tts": ("qwen_remote", "piper", "elevenlabs", "openai", "custom_vllm"),
    "vad": ("silero", "energy"),
}


def _path():
    return secure_config._path().with_name("layers.json")


def _default(provider: str) -> dict[str, str]:
    provider = str(provider or "qwen").strip().lower()
    if provider == "openai":
        return {"asr": "openai", "tts": "openai", "vad": "silero"}
    return {"asr": "qwen_remote", "tts": "qwen_remote", "vad": "silero"}


def selected(provider: str | None = None) -> dict[str, str]:
    """Return safe layer names, falling back to the active stack defaults."""
    if provider is None:
        provider = "qwen"
        try:
            from .provider_choice import selected as selected_provider
            provider = selected_provider()
        except Exception:  # pragma: no cover - defensive import fallback
            pass
    result = _default(provider)
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for name, options in LAYER_OPTIONS.items():
                if value.get(name) in options:
                    result[name] = value[name]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return result


def _write(value: dict[str, str]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.parent.chmod(0o700)
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name != "nt":
        path.chmod(0o600)


def choose(payload: dict) -> dict:
    """Validate and persist one or more layer choices."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "layer_selection_must_be_object"}
    current = selected()
    for name, options in LAYER_OPTIONS.items():
        if name in payload:
            value = str(payload.get(name) or "").strip().lower()
            if value not in options:
                return {"ok": False, "error": f"unsupported_{name}_provider"}
            current[name] = value
    if not any(name in payload for name in LAYER_OPTIONS):
        return {"ok": False, "error": "at_least_one_layer_required"}
    _write(current)
    return {"ok": True, "layers": current, "restart_required": True}


def set_for_stack(provider: str) -> dict[str, str]:
    """Reset only ASR/TTS to a complete-stack default; preserve VAD."""
    current = selected(provider)
    existing = selected(provider)
    current["asr"] = "openai" if provider == "openai" else "qwen_remote"
    current["tts"] = "openai" if provider == "openai" else "qwen_remote"
    current["vad"] = existing.get("vad", "silero")
    _write(current)
    return current


def status(provider: str | None = None) -> dict:
    return {
        "ok": True,
        "layers": selected(provider),
        "options": {name: list(values) for name, values in LAYER_OPTIONS.items()},
        "restart_required": False,
    }
