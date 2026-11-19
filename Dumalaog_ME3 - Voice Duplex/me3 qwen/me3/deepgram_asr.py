"""Protected Deepgram Nova-3 ASR failover and key validation."""
from __future__ import annotations

import time

import requests

from . import secure_config

API = "https://api.deepgram.com/v1"
MODEL = "nova-3"


def validate_api_key(api_key: str) -> dict:
    """Check a key without returning its value or provider response body."""
    value = str(api_key or "").strip()
    if not value or len(value) > 4096 or "\n" in value or "\r" in value:
        return {"ok": False, "error": "invalid_deepgram_api_key"}
    try:
        response = requests.get(
            f"{API}/auth/token",
            headers={"Authorization": f"Token {value}"},
            timeout=(4, 8),
        )
    except requests.RequestException as exc:
        return {"ok": False, "error": "deepgram_validation_unreachable",
                "error_type": type(exc).__name__}
    if response.status_code != 200:
        return {"ok": False, "error": "deepgram_credentials_rejected",
                "status_code": int(response.status_code)}
    return {"ok": True, "validated": True, "provider": "deepgram",
            "model": MODEL}


def configure_api_key(api_key: str) -> dict:
    """Validate, then store the key in ME3's mode-0600 private settings file."""
    checked = validate_api_key(api_key)
    if not checked.get("ok"):
        return checked
    saved = secure_config.set_values({"DEEPGRAM_API_KEY": str(api_key).strip()})
    if not saved.get("ok"):
        return saved
    return {"ok": True, "configured": True, "validated": True,
            "provider": "deepgram", "model": MODEL,
            "storage": saved.get("storage")}


def health() -> dict:
    return {"ok": True, "provider": "deepgram", "model": MODEL,
            "configured": secure_config.configured("DEEPGRAM_API_KEY"),
            "role": "secondary_asr_fallback",
            "language_mode": "multilingual English/Tagalog"}


def transcribe_wav(raw_wav: bytes, *, timeout_s: float = 20) -> tuple[str, float]:
    """Transcribe a WAV using Nova-3 multilingual with English/Tagalog hints."""
    api_key = secure_config.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("Deepgram API key is not configured")
    started = time.perf_counter()
    response = requests.post(
        f"{API}/listen",
        params=[("model", MODEL), ("language", "multi"),
                ("language_hint", "en"), ("language_hint", "tl"),
                ("smart_format", "true")],
        headers={"Authorization": f"Token {api_key}",
                 "Content-Type": "audio/wav"},
        data=raw_wav,
        timeout=(4, float(timeout_s)),
    )
    response.raise_for_status()
    payload = response.json()
    channels = (payload.get("results") or {}).get("channels") or []
    alternatives = (channels[0].get("alternatives") or []) if channels else []
    transcript = str(alternatives[0].get("transcript") or "").strip() if alternatives else ""
    if not transcript:
        raise RuntimeError("Deepgram returned an empty transcript")
    return transcript, (time.perf_counter() - started) * 1000
