"""Optional ElevenLabs streaming TTS backend for the ME3 voice loop.

ME3 keeps Qwen as the reasoning/tool-calling model and uses ElevenLabs only
as the voice renderer. The backend requests raw 16 kHz PCM so the existing
full-duplex audio path can play it and stop it immediately on barge-in.

The API key and voice ID are read from environment variables only. They are
never logged, written to project files, or included in tool results.
"""
from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from urllib.parse import quote

import requests

log = logging.getLogger("me3.elevenlabs_tts")


class AudioChunk:
    """Small Piper-compatible wrapper consumed by LocalVoiceLoop."""

    __slots__ = ("audio_int16_bytes",)

    def __init__(self, audio_int16_bytes: bytes):
        self.audio_int16_bytes = audio_int16_bytes


class ElevenLabsTTS:
    """Streaming ElevenLabs TTS adapter with a Piper-compatible interface."""

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        base_url: str = "https://api.elevenlabs.io/v1",
        model_id: str = "eleven_flash_v2_5",
        language_code: str | None = None,
        output_format: str = "pcm_16000",
        timeout_s: float = 30.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.15,
        request_fn=None,
    ):
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")
        if not voice_id:
            raise RuntimeError("ELEVENLABS_VOICE_ID is not configured")
        self.api_key = api_key
        self.voice_id = voice_id
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.language_code = (str(language_code).strip().lower()
                              if language_code else None)
        self.output_format = output_format
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.request_fn = request_fn or requests.post
        self.config = SimpleNamespace(sample_rate=self._sample_rate(output_format))
        self._load_ms = 0.0
        self.last_event = {
            "backend": "elevenlabs",
            "ok": None,
            "sample_rate": self.config.sample_rate,
        }

    @staticmethod
    def _sample_rate(output_format: str) -> int:
        try:
            return int(output_format.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return 16000

    @classmethod
    def from_config(cls, cfg: dict) -> "ElevenLabsTTS":
        options = cfg.get("elevenlabs", {})
        key_env = options.get("api_key_env", "ELEVENLABS_API_KEY")
        voice_env = options.get("voice_id_env", "ELEVENLABS_VOICE_ID")
        return cls(
            api_key=os.environ.get(key_env, ""),
            voice_id=os.environ.get(voice_env, options.get("voice_id", "")),
            base_url=options.get("base_url", "https://api.elevenlabs.io/v1"),
            model_id=options.get("model_id", "eleven_flash_v2_5"),
            language_code=os.environ.get(
                "ME3_ELEVENLABS_LANGUAGE_CODE",
                options.get("language_code", "fil")),
            output_format=options.get("output_format", "pcm_16000"),
            timeout_s=options.get("timeout_s", 30),
            max_retries=options.get("max_retries", 1),
            retry_backoff_s=options.get("retry_backoff_s", 0.15),
        )

    @staticmethod
    def _error_class(exc: Exception, status_code: int | None = None) -> str:
        if isinstance(exc, requests.Timeout):
            return "timeout"
        if isinstance(exc, (requests.ConnectionError,
                            requests.ChunkedEncodingError)):
            return "network"
        if status_code in (401, 403):
            return "auth"
        if status_code == 429:
            return "rate_limit"
        if status_code is not None and status_code >= 500:
            return "provider"
        if status_code is not None and status_code >= 400:
            return "http"
        return "unknown"

    def _record_event(self, *, ok: bool, attempts: int,
                      started: float, n_bytes: int = 0,
                      error_class: str | None = None,
                      status_code: int | None = None) -> None:
        event = {
            "backend": "elevenlabs",
            "ok": bool(ok),
            "attempts": int(attempts),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "bytes": int(n_bytes),
            "sample_rate": self.config.sample_rate,
        }
        if error_class:
            event["error_class"] = error_class
        if status_code is not None:
            event["status_code"] = int(status_code)
        self.last_event = event

    def synthesize(self, text: str):
        """Yield raw PCM chunks as objects compatible with PiperVoice."""
        text = str(text or "").strip()
        if not text:
            return
        url = f"{self.base_url}/text-to-speech/{quote(self.voice_id, safe='')}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/pcm",
            "content-type": "application/json",
        }
        params = {"output_format": self.output_format, "enable_logging": "false"}
        body = {"text": text, "model_id": self.model_id}
        # ElevenLabs rejects/ignores this for Multilingual v2; Flash v2.5 and
        # newer multilingual-capable models accept `fil` to enforce Filipino.
        if self.language_code and self.model_id != "eleven_multilingual_v2":
            body["language_code"] = self.language_code
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            response = None
            n_bytes = 0
            status_code = None
            try:
                response = self.request_fn(
                    url,
                    headers=headers,
                    params=params,
                    json=body,
                    stream=True,
                    timeout=self.timeout_s,
                )
                status_code = getattr(response, "status_code", None)
                retryable = status_code == 429 or (
                    status_code is not None and status_code >= 500)
                if retryable and attempt < self.max_retries:
                    log.warning(
                        "ElevenLabs retry backend=elevenlabs attempt=%d "
                        "status=%s", attempt + 1, status_code)
                    continue
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    raw = bytes(chunk)
                    if len(raw) % 2:
                        raise RuntimeError(
                            "ElevenLabs returned an odd number of PCM bytes")
                    n_bytes += len(raw)
                    yield AudioChunk(raw)
                self._record_event(ok=True, attempts=attempt + 1,
                                   started=started, n_bytes=n_bytes,
                                   status_code=status_code)
                log.info(
                    "ElevenLabs synthesis backend=elevenlabs attempts=%d "
                    "bytes=%d latency_ms=%.1f sample_rate=%d",
                    attempt + 1, n_bytes,
                    self.last_event["latency_ms"], self.config.sample_rate)
                return
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt < self.max_retries:
                    log.warning(
                        "ElevenLabs retry backend=elevenlabs attempt=%d "
                        "error_class=%s", attempt + 1,
                        self._error_class(exc, status_code))
                    if self.retry_backoff_s:
                        time.sleep(self.retry_backoff_s)
                    continue
                error_class = self._error_class(exc, status_code)
                self._record_event(ok=False, attempts=attempt + 1,
                                   started=started, n_bytes=n_bytes,
                                   error_class=error_class,
                                   status_code=status_code)
                raise
            except Exception as exc:  # noqa: BLE001
                retryable = status_code == 429 or (
                    status_code is not None and status_code >= 500)
                if retryable and attempt < self.max_retries:
                    log.warning(
                        "ElevenLabs retry backend=elevenlabs attempt=%d "
                        "error_class=%s status=%s", attempt + 1,
                        self._error_class(exc, status_code), status_code)
                    if self.retry_backoff_s:
                        time.sleep(self.retry_backoff_s)
                    continue
                error_class = self._error_class(exc, status_code)
                self._record_event(ok=False, attempts=attempt + 1,
                                   started=started, n_bytes=n_bytes,
                                   error_class=error_class,
                                   status_code=status_code)
                raise
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
