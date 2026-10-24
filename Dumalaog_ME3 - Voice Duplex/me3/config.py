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


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache the YAML config."""
    cfg_path = os.environ.get("ME3_CONFIG")
    if not cfg_path:
        root = Path(__file__).resolve().parent.parent
        cfg_path = root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The Pi installer can select a RAM-appropriate Whisper model without
    # rewriting the checked-in source config.  These are non-secret runtime
    # overrides; the default values remain the single-source-of-truth policy.
    asr_model = os.environ.get("ME3_ASR_MODEL")
    if asr_model:
        cfg.setdefault("asr", {})["model"] = asr_model
    tts_voice = os.environ.get("ME3_TTS_VOICE")
    if tts_voice:
        cfg.setdefault("tts", {})["voice"] = tts_voice
    overlap_voice = os.environ.get("ME3_OVERLAP_VOICE")
    if overlap_voice:
        cfg.setdefault("tts", {})["overlap_voice"] = overlap_voice
    return cfg


def audio_cfg() -> dict:
    return load_config()["audio"]


def sr() -> int:
    return int(audio_cfg()["sample_rate"])


def frame_samples() -> int:
    a = audio_cfg()
    return int(a["sample_rate"] * a["frame_ms"] / 1000)
