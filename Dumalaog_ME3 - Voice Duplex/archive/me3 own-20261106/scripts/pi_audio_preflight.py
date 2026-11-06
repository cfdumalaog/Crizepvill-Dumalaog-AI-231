"""Read-only audio preflight for the ME3 target device.

This command separates three failure layers that otherwise look identical to
the agent: ALSA hardware enumeration, PortAudio visibility, and permissions.
It never opens a stream, changes mixer state, or writes a file.

Usage on the Pi:
    python scripts/pi_audio_preflight.py
    python scripts/pi_audio_preflight.py --json
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys


def _alsa_list(command: str) -> dict:
    executable = shutil.which(command)
    if executable is None:
        return {"available": False, "command": command, "stdout": ""}
    try:
        p = subprocess.run(
            [executable, "-l"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": True,
            "command": command,
            "returncode": p.returncode,
            "stdout": (p.stdout or "").strip(),
            "stderr": (p.stderr or "").strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "command": command,
                "error": type(exc).__name__, "stdout": ""}


def _device_row(index: int, device: dict) -> dict:
    return {
        "index": index,
        "name": str(device.get("name", "")),
        "max_input_channels": int(device.get("max_input_channels", 0)),
        "max_output_channels": int(device.get("max_output_channels", 0)),
        "default_samplerate": device.get("default_samplerate"),
        "hostapi": device.get("hostapi"),
    }


def _alsa_has_hardware(record: dict) -> bool:
    """Return whether ``arecord/aplay -l`` listed a real ALSA card."""
    return any(line.lstrip().startswith("card ")
               for line in str(record.get("stdout", "")).splitlines())


def inspect_audio() -> dict:
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "portaudio": {"ok": False, "devices": [], "default": None},
        "alsa": {"capture": _alsa_list("arecord"),
                 "playback": _alsa_list("aplay")},
    }
    try:
        import sounddevice as sd
        devices = list(sd.query_devices())
        result["portaudio"] = {
            "ok": True,
            "devices": [_device_row(i, d) for i, d in enumerate(devices)],
            "default": list(sd.default.device),
        }
    except Exception as exc:  # noqa: BLE001
        result["portaudio"] = {
            "ok": False,
            "error": type(exc).__name__,
            "devices": [],
            "default": None,
        }

    devices = result["portaudio"]["devices"]
    result["input_devices"] = [d for d in devices
                                 if d["max_input_channels"] > 0]
    result["output_devices"] = [d for d in devices
                                  if d["max_output_channels"] > 0]
    if not result["portaudio"]["ok"]:
        result["status"] = "blocked_portaudio_unavailable"
        result["exit_code"] = 4
    elif not result["input_devices"] and not result["output_devices"]:
        # PortAudio can expose zero devices even when ALSA can still tell us
        # which direction is missing.  Preserve that useful diagnosis for
        # headless Pi images instead of collapsing it into a generic failure.
        alsa_capture = _alsa_has_hardware(result["alsa"]["capture"])
        alsa_playback = _alsa_has_hardware(result["alsa"]["playback"])
        if not alsa_capture and alsa_playback:
            result["status"] = "blocked_no_capture_device"
            result["exit_code"] = 2
        elif alsa_capture and not alsa_playback:
            result["status"] = "blocked_no_playback_device"
            result["exit_code"] = 3
        else:
            result["status"] = "blocked_no_input_or_output"
            result["exit_code"] = 4
    elif not result["input_devices"]:
        result["status"] = "blocked_no_capture_device"
        result["exit_code"] = 2
    elif not result["output_devices"]:
        result["status"] = "blocked_no_playback_device"
        result["exit_code"] = 3
    else:
        result["status"] = "pass_devices_visible"
        result["exit_code"] = 0
    return result


def _print_human(result: dict) -> None:
    print("ME3 audio preflight (read-only)")
    print(f"  status: {result['status']}")
    print(f"  PortAudio devices: {len(result['portaudio']['devices'])}")
    print(f"  input devices: {len(result['input_devices'])}")
    print(f"  output devices: {len(result['output_devices'])}")
    for d in result["input_devices"]:
        print(f"    in  [{d['index']}] {d['name']}")
    for d in result["output_devices"]:
        print(f"    out [{d['index']}] {d['name']}")
    for direction in ("capture", "playback"):
        alsa = result["alsa"][direction]
        if not alsa["available"]:
            print(f"  ALSA {direction}: command unavailable")
        else:
            text = alsa.get("stdout", "")
            lines = len(text.splitlines()) if text else 0
            print(f"  ALSA {direction}: command returned {lines} lines")
    if result["status"] == "blocked_no_capture_device":
        print("  ACTION: connect a USB/I2S microphone or provide an ALSA "
              "capture/loopback device, then rerun this command.")
    elif result["status"] == "blocked_no_playback_device":
        print("  ACTION: connect a speaker/HDMI/audio output, then rerun "
              "this command.")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit the redacted machine-readable report")
    args = parser.parse_args()
    result = inspect_audio()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_human(result)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
