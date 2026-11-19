"""Gemini tool-brain adapter through Google's OpenAI-compatible endpoint.

Google documents the Gemini OpenAI compatibility endpoint as supporting chat
completions, streaming, and function calling. ME3 therefore reuses the same
secret-safe :class:`ToolBrain` execution loop used by Qwen/OpenAI while this
module keeps Gemini's endpoint, model, and fast-mode defaults explicit.
"""
from __future__ import annotations

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GEMINI_API_KEY"


def defaults() -> dict[str, object]:
    """Return the OpenAI-compatible Gemini brain configuration."""
    return {
        "provider": "gemini",
        "base_url": BASE_URL,
        "model": DEFAULT_MODEL,
        "api_key_env": API_KEY_ENV,
        "temperature": 0.2,
        "max_tokens": 4096,
        "timeout_s": 90,
        "max_tool_steps": 24,
        "max_development_steps": 48,
        "history_messages": 16,
        "stream": True,
        "reasoning_effort": "none",
        "fast_mode": True,
        "thinking": False,
        "name": "gemini-tools",
    }
