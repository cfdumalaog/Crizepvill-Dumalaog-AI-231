"""ME3 shared config loader.

All MCP servers and the agent read config from config/config.yaml
relative to the project root, overridable via ME3_CONFIG env var.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def project_root() -> Path:
    """Find the repository root from either variant's nested package path."""
    configured = os.environ.get("ME3_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    package_file = Path(__file__).resolve()
    for parent in package_file.parents:
        if (parent / "config" / "config.yaml").is_file():
            return parent
    raise RuntimeError("could not locate the ME3 project root")


def _load_project_env() -> None:
    """Load the local ElevenLabs CLI .env without overriding real env vars.

    The ElevenLabs onboarding CLI stores the authorized key in ``.env``.  The
    Pi uses systemd-provided ``~/.config/me3/me3.env`` instead, so this local
    convenience is deliberately additive: explicit process/service variables
    always win, and the file is never printed or copied into release artifacts.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - requirements install it
        return
    root = project_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache the YAML config."""
    _load_project_env()
    cfg_path = os.environ.get("ME3_CONFIG")
    if not cfg_path:
        root = project_root()
        cfg_path = root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The Pi installer can select a RAM-appropriate Whisper model without
    # rewriting the checked-in source config.  These are non-secret runtime
    # overrides; the default values remain the single-source-of-truth policy.
    asr_model = os.environ.get("ME3_ASR_MODEL")
    if asr_model:
        cfg.setdefault("asr", {})["model"] = asr_model
    # Voice selection is profile-owned. Do not let a service environment
    # switch voices mid-deployment: normal replies, backchannels, and
    # completion announcements must stay on the same model/voice.
    # The optional Qwen Instruct audio variant is a runtime composition: local
    # Silero VAD and the existing text ToolBrain stay in place, while ASR/TTS
    # point at protected DGX3 vLLM services. Endpoint values come from env so
    # the supplied credential file never enters the repository or config.yaml.
    variant = os.environ.get("ME3_VARIANT", "current").strip().lower()
    variants = cfg.get("variants", {}) or {}
    if variant not in {"current", "qwen_instruct"}:
        raise ValueError("ME3_VARIANT must be current or qwen_instruct")
    if variant == "qwen_instruct":
        selected = variants.get("qwen_instruct", {})
        cfg["asr"] = {**cfg.get("asr", {}), **selected.get("asr", {})}
        cfg["tts"] = {**cfg.get("tts", {}), **selected.get("tts", {})}
    cfg["runtime_variant"] = variant
    return cfg


def audio_cfg() -> dict:
    return load_config()["audio"]


def sr() -> int:
    return int(audio_cfg()["sample_rate"])


def frame_samples() -> int:
    a = audio_cfg()
    return int(a["sample_rate"] * a["frame_ms"] / 1000)
