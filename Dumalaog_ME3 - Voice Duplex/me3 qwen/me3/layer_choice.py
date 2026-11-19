"""Persist the independently selectable ME3 runtime layers.

The complete voice stack (Qwen or OpenAI) remains the high-level provider
choice, but the UI also exposes the text/tool brain so the real dependency
chain is visible. Custom provider profiles are represented by their actual
profile names in the UI; the private runtime still stores the compatibility
backend name ``custom_vllm`` and the active profile separately.
"""
from __future__ import annotations

import json
import os

from . import secure_config


LAYER_OPTIONS = {
    "asr": ("qwen_remote", "deepgram", "faster_whisper", "openai"),
    "tts": ("qwen_remote", "piper", "elevenlabs", "openai"),
    "vad": ("silero", "energy"),
}
_STORAGE_OPTIONS = {
    "asr": (*LAYER_OPTIONS["asr"], "custom_vllm"),
    "tts": (*LAYER_OPTIONS["tts"], "custom_vllm"),
    "vad": LAYER_OPTIONS["vad"],
}
_BRAIN_OPTIONS = ("qwen", "openai", "gemini")


def _path():
    return secure_config._path().with_name("layers.json")


def _default(provider: str) -> dict[str, str]:
    provider = str(provider or "qwen").strip().lower()
    if provider == "openai":
        return {"asr": "openai", "tts": "openai", "vad": "silero"}
    return {"asr": "qwen_remote", "tts": "qwen_remote", "vad": "silero"}


def _profiles(role: str) -> list[dict]:
    """Return public, non-secret profiles for one runtime role."""
    try:
        from . import setup_profiles
        return [item for item in setup_profiles.status().get("profiles", [])
                if item.get("role") == role and item.get("profile_id")]
    except Exception:  # pragma: no cover - status must never break the loop
        return []


def _profile_value(profile: dict) -> str:
    return "profile:" + str(profile.get("profile_id") or "").strip()


def _dynamic_options(role: str) -> tuple[str, ...]:
    values = list(LAYER_OPTIONS.get(role, ()))
    values.extend(_profile_value(item) for item in _profiles(role))
    return tuple(value for value in values if value != "profile:")


def _profile_by_value(value: str, role: str) -> dict | None:
    if not value.startswith("profile:"):
        return None
    profile_id = value[len("profile:"):].strip()
    return next((item for item in _profiles(role)
                 if item.get("profile_id") == profile_id), None)


def _ui_value(role: str, stored: str) -> str:
    """Translate the private compatibility marker to a real profile choice."""
    if stored != "custom_vllm":
        return stored
    try:
        from . import setup_profiles
        active = setup_profiles.status().get("active", {}).get(role) or {}
        profile_id = active.get("profile_id")
        if profile_id:
            return "profile:" + str(profile_id)
    except Exception:  # pragma: no cover
        pass
    return stored


def _option_labels() -> dict[str, str]:
    labels = {
        "qwen_remote": "Qwen2-Audio / Qwen3-TTS DGX3",
        "deepgram": "Deepgram Nova-3",
        "faster_whisper": "faster-whisper local",
        "openai": "OpenAI",
        "piper": "Piper local",
        "elevenlabs": "ElevenLabs Filipino",
        "silero": "Silero local",
        "energy": "Energy gate fallback",
        "qwen": "Qwen text/tool brain",
        "gemini": "Gemini text/tool brain",
    }
    for role in ("llm", "asr", "tts"):
        for profile in _profiles(role):
            value = _profile_value(profile)
            labels[value] = f"{profile.get('profile_name') or value} · {profile.get('model') or 'provider'}"
    return labels


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
            for name, options in _STORAGE_OPTIONS.items():
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
    """Validate and persist one or more layers, including the tool brain."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "layer_selection_must_be_object"}
    current = selected()

    # A brain selection changes the high-level provider.  Apply it first,
    # because provider_choice.choose() resets ASR/TTS to complete-stack
    # defaults; explicit audio values below then restore the operator's picks.
    if "brain" in payload:
        requested = str(payload.get("brain") or "").strip().lower()
        target = requested
        if requested.startswith("profile:"):
            profile = _profile_by_value(requested, "llm")
            if profile is None:
                return {"ok": False, "error": "unknown_llm_profile"}
            from . import setup_profiles
            setup_profiles.activate_profile(profile["profile_id"])
            target = "custom_vllm"
        elif requested not in (*_BRAIN_OPTIONS, "custom_vllm"):
            return {"ok": False, "error": "unsupported_brain_provider"}
        from . import provider_choice
        result = provider_choice.choose(target)
        if not result.get("ok"):
            return result
        current = selected()

    for name in LAYER_OPTIONS:
        if name in payload:
            value = str(payload.get(name) or "").strip().lower()
            if value in LAYER_OPTIONS[name]:
                current[name] = value
            elif value == "custom_vllm":
                if not _profiles(name):
                    return {"ok": False, "error": f"no_{name}_profile_configured"}
                current[name] = value
            elif _profile_by_value(value, name) is not None:
                from . import setup_profiles
                setup_profiles.activate_profile(value[len("profile:"):])
                current[name] = "custom_vllm"
            else:
                return {"ok": False, "error": f"unsupported_{name}_provider"}
    if not any(name in payload for name in (*LAYER_OPTIONS, "brain")):
        return {"ok": False, "error": "at_least_one_layer_required"}
    _write(current)
    return {"ok": True, "layers": status()["layers"], "restart_required": True}


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
    current = selected(provider)
    try:
        from . import provider_choice, setup_profiles
        brain = provider_choice.selected()
        active_llm = setup_profiles.status().get("active", {}).get("llm") or {}
        if brain == "custom_vllm" and active_llm.get("profile_id"):
            brain = "profile:" + str(active_llm["profile_id"])
    except Exception:  # pragma: no cover
        brain = "qwen"
    values = {name: _ui_value(name, value) for name, value in current.items()}
    values["brain"] = brain
    options = {
        "asr": list(_dynamic_options("asr")),
        "tts": list(_dynamic_options("tts")),
        "vad": list(_dynamic_options("vad")),
        "brain": list(_BRAIN_OPTIONS) + [
            _profile_value(item) for item in _profiles("llm")
        ],
    }
    return {
        "ok": True,
        "layers": values,
        "options": options,
        "option_labels": _option_labels(),
        "restart_required": False,
    }
