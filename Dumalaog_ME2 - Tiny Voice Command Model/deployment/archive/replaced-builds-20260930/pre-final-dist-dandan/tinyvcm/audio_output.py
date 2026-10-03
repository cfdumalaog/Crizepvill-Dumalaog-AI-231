"""Best-effort host mixer control for Linux deployments.

Browser playback volume is controlled by the UI on every platform. On Linux
Pi deployments, this adapter also adjusts a usable system sink so local Pi
audio follows the same volume commands.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys


class SystemVolumeOutput:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled and sys.platform.startswith("linux"))
        self.mode = "browser"
        self.status = "Browser playback volume is controlled by the Assistant page."
        self._wpctl: str | None = None
        self._amixer: str | None = None
        self._alsa_control: str | None = None
        if self.enabled:
            self._detect_output()

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=3)

    def _detect_output(self) -> None:
        wpctl = shutil.which("wpctl")
        if wpctl:
            try:
                result = self._run([wpctl, "status"])
                text = result.stdout if result.returncode == 0 else ""
                sinks = text.split("Sinks:", 1)[1].split("Sources:", 1)[0] if "Sinks:" in text and "Sources:" in text else ""
                active = next((line.strip() for line in sinks.splitlines() if re.search(r"^\s*\*\s*\d+\.", line)), "")
                if active and "dummy output" not in active.casefold():
                    self._wpctl = wpctl
                    self.mode = "pipewire"
                    self.status = f"Pi system output: {active.lstrip('* ').strip()}"
                    return
            except (OSError, subprocess.SubprocessError):
                pass

        amixer = shutil.which("amixer")
        if amixer:
            try:
                result = self._run([amixer, "-c", "0", "scontrols"])
                controls = re.findall(r"Simple mixer control '([^']+)'", result.stdout or "")
                for preferred in ("PCM", "Master", "Headphone"):
                    if preferred in controls:
                        self._amixer = amixer
                        self._alsa_control = preferred
                        self.mode = "alsa-card-0"
                        self.status = f"Pi ALSA card 0 {preferred} mixer"
                        return
            except (OSError, subprocess.SubprocessError):
                pass
        self.status = "No active Pi mixer found; browser playback volume remains available."

    def set_percent(self, percent: int) -> bool:
        value = max(0, min(100, int(percent)))
        if not self.enabled:
            return False
        if self._wpctl:
            command = [self._wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{value / 100:.2f}"]
        elif self._amixer and self._alsa_control:
            command = [self._amixer, "-q", "-c", "0", "sset", self._alsa_control, f"{value}%"]
        else:
            return False
        try:
            result = self._run(command)
            if result.returncode == 0:
                return True
            self.status = f"Pi mixer rejected volume {value}%; browser playback volume remains available."
        except (OSError, subprocess.SubprocessError) as exc:
            self.status = f"Could not set Pi mixer volume ({type(exc).__name__}); browser playback volume remains available."
        return False
