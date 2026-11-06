"""ME3 local-dev environment check.

Mirrors ME2's local-dev-before-deploy workflow: verify the shared workspace
.venv has every dependency the full-duplex agent and its four MCP servers need,
BEFORE any Pi deployment is attempted. Run this first on the Windows dev box.

Usage:
    & '..\\..\\.venv\\Scripts\\python.exe' scripts/dev_env_check.py
    (or from the ME3 root:  python scripts/dev_env_check.py)

Exit code 0 = all required imports present and importable.
Exit code 1 = one or more missing (names printed). Nothing is installed here;
install.sh (Pi) or pip on the dev box is responsible for that.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

# (import_name, pip_name, why). import_name is what we probe; pip_name is what
# the user installs if it is missing.
REQUIRED = [
    ("numpy",          "numpy",          "core arrays / audio buffers"),
    ("yaml",           "pyyaml",         "config loading (config/config.yaml)"),
    ("mcp",            "mcp",            "MCP SDK - all four servers speak MCP"),
    ("fastapi",        "fastapi",        "FastMCP server framework"),
    ("uvicorn",        "uvicorn",        "ASGI transport for streamable-http mode"),
    ("pipecat",        "pipecat-ai",     "full-duplex orchestration pipeline"),
    ("faster_whisper", "faster-whisper", "SOTA ASR (CTranslate2)"),
    ("piper",          "piper-tts",      "SOTA low-latency TTS"),
    ("onnxruntime",    "onnxruntime",    "CPU Silero VAD runtime"),
    ("sounddevice",    "sounddevice",    "full-duplex mic/speaker I/O (PortAudio)"),
    ("requests",       "requests",       "LLM endpoint client (remote vLLM option)"),
]


def probe(module: str) -> tuple[bool, str]:
    """Return (ok, version-or-error) for a module without raising."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", None)
    if version is None:
        # some packages nest version
        sub = getattr(mod, "version", None)
        version = getattr(sub, "__version__", None) or getattr(sub, "VERSION", None)
    return True, str(version) if version else "present"


def main() -> int:
    print("ME3 local-dev environment check")
    print(f"  python : {sys.executable}")
    print(f"  project: {PROJECT}\n")
    missing = []
    for import_name, pip_name, why in REQUIRED:
        ok, detail = probe(import_name)
        flag = "OK " if ok else "MISS"
        print(f"  [{flag}] {import_name:<14} {detail:<28}  {why}")
        if not ok:
            missing.append((import_name, pip_name))
    print()
    if missing:
        print(f"MISSING ({len(missing)}):")
        for import_name, pip_name in missing:
            print(f"    pip install {pip_name}")
        print("\nRESULT: FAIL - install the above, then re-run.")
        return 1
    print("RESULT: PASS - all ME3 dependencies importable in this environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
