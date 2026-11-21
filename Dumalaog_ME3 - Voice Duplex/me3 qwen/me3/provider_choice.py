"""Persist the selected ME3 voice stack outside deployable source files."""
from __future__ import annotations

import json
import os

from . import secure_config

PROVIDERS = ("qwen", "openai", "gemini", "custom_vllm")


def _path():
    return secure_config._path().with_name("provider.json")


def selected() -> str:
    override = os.getenv("ME3_PROVIDER", "").strip().lower()
    if override in PROVIDERS:
        return override
    try:
        value = json.loads(_path().read_text(encoding="utf-8"))
        if value.get("provider") in PROVIDERS:
            return value["provider"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return "qwen"


def choose(provider: str) -> dict:
    provider = str(provider or "").strip().lower()
    if provider not in PROVIDERS:
        return {"ok": False, "error": "unsupported_provider"}
    if provider == "openai" and not openai_key():
        return {"ok": False, "error": "openai_api_key_required"}
    if provider == "gemini" and not gemini_key():
        return {"ok": False, "error": "gemini_api_key_required"}
    if provider == "custom_vllm":
        from . import setup_profiles
        if not setup_profiles.active_profile("llm"):
            return {"ok": False, "error": "custom_llm_profile_required"}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        path.parent.chmod(0o700)
    tmp = path.with_suffix(".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"provider": provider}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    if os.name != "nt":
        path.chmod(0o600)
    # Keep a complete-stack switch deterministic.  The operator can then
    # deliberately override individual layers from the layer selectors.
    from . import layer_choice
    layer_choice.set_for_stack(provider)
    return {"ok": True, "provider": provider, "restart_required": True}


def status() -> dict:
    from . import setup_profiles
    custom = setup_profiles.status()
    return {
        "ok": True,
        "selected": selected(),
        "providers": list(PROVIDERS),
        "qwen_key_configured": bool(qwen_key()),
        "openai_key_configured": bool(openai_key()),
        "gemini_key_configured": bool(gemini_key()),
        "custom_vllm_configured": bool(custom.get("active", {}).get("llm")),
        "custom_vllm_profiles": custom.get("profiles", []),
        "models": {
            "openai": {"asr": "gpt-transcribe", "llm": "gpt-4.1-mini",
                        "tts": "gpt-4o-mini-tts"},
            "gemini": {"llm": "gemini-2.5-flash"},
        },
    }


def openai_key() -> str:
    """Prefer the key just saved in the UI over an older service env value."""
    return secure_config._read().get("OPENAI_API_KEY", "").strip() or secure_config.get("OPENAI_API_KEY")


def qwen_key() -> str:
    """Return the protected Qwen/vLLM key, preferring a newer private save."""
    return secure_config._read().get("VLLM_API_KEY", "").strip() or secure_config.get("VLLM_API_KEY")


def save_qwen_key(value: str) -> dict:
    """Save a Qwen/vLLM key without returning or logging its value."""
    key = str(value or "").strip()
    if len(key) < 20 or len(key) > 4096 or "\n" in key or "\r" in key:
        return {"ok": False, "error": "invalid_qwen_api_key_format"}
    result = secure_config.set_values({"VLLM_API_KEY": key})
    return {"ok": bool(result.get("ok")),
            "configured": bool(result.get("ok")),
            **({"error": result["error"]} if not result.get("ok") else {})}


def save_openai_key(value: str) -> dict:
    key = str(value or "").strip()
    if not key.startswith("sk-") or len(key) < 20:
        return {"ok": False, "error": "invalid_openai_api_key_format"}
    result = secure_config.set_values({"OPENAI_API_KEY": key})
    return {"ok": bool(result.get("ok")),
            "configured": bool(result.get("ok")),
            **({"error": result["error"]} if not result.get("ok") else {})}


def gemini_key() -> str:
    """Return the privately stored Gemini key without exposing it to status."""
    return (secure_config._read().get("GEMINI_API_KEY", "").strip()
            or secure_config.get("GEMINI_API_KEY"))


def save_gemini_key(value: str) -> dict:
    key = str(value or "").strip()
    if len(key) < 20 or len(key) > 4096 or "\n" in key or "\r" in key:
        return {"ok": False, "error": "invalid_gemini_api_key_format"}
    result = secure_config.set_values({"GEMINI_API_KEY": key})
    return {"ok": bool(result.get("ok")),
            "configured": bool(result.get("ok")),
            **({"error": result["error"]} if not result.get("ok") else {})}
