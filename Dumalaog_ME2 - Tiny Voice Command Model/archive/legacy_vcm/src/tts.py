"""
Offline Voiced Speech Output Engine (Text-to-Speech & Pre-Rendered Voice Bank).
Supports instantaneous (0 ms synthesis latency) playback of high-fidelity pre-rendered
responses, with platform-native offline TTS fallback for dynamic strings.
100% on-device: 0 cloud APIs, 0 external network dependencies.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from .config import ASSETS_DIR


class VoiceOutputEngine:
    """
    Manages voiced spoken output across Windows, Linux, and Raspberry Pi 5.
    Combines high-fidelity pre-rendered 16kHz speech clips with offline dynamic TTS.
    Uses non-blocking asynchronous kernel audio dispatch to avoid thread contention.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, assets_dir: Optional[Path] = None):
        self.assets_dir = Path(assets_dir) if assets_dir else ASSETS_DIR
        self.voice_dir = self.assets_dir / "voice_responses"
        self.earcon_dir = self.assets_dir / "earcons"
        self.last_spoken_text = ""
        self.last_spoken_wav: Optional[Path] = None
        self.listeners: list[Callable[[str, Optional[Path]], None]] = []
        self.muted = os.environ.get("VCM_MUTE_VOICE", "0") == "1"

        # Windows SAPI voice synthesizer instance (lazy loaded)
        self._sapi_speaker = None
        if sys.platform == "win32":
            try:
                import win32com.client
                self._sapi_speaker = win32com.client.Dispatch("SAPI.SpVoice")
                self._sapi_speaker.Rate = 0
            except Exception:
                self._sapi_speaker = None

    @classmethod
    def get_instance(cls, assets_dir: Optional[Path] = None) -> "VoiceOutputEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(assets_dir)
            return cls._instance

    def register_listener(self, callback: Callable[[str, Optional[Path]], None]):
        """Register a callback invoked when speech is triggered (e.g. for subtitle rendering)."""
        self.listeners.append(callback)

    def play_earcon(self, kind: str = "wake", blocking: bool = False):
        """
        Instantly plays an authentic synthesized musical earcon chime.
        kind: 'wake' (rising dual-tone), 'success' (bell chime), 'cancel' (falling tone).
        """
        if self.muted:
            return
        earcon_wav = self.earcon_dir / f"{kind}_earcon.wav"
        if earcon_wav.exists():
            self._play_wav(earcon_wav, blocking=blocking)

    def speak(self, text: str, response_key: Optional[str] = None, blocking: bool = False):
        """
        Speaks the given text or plays its pre-rendered audio response.
        If response_key is given and matching WAV exists, plays the pre-rendered clip.
        Otherwise falls back to offline TTS.
        """
        wav_path: Optional[Path] = None
        if response_key:
            candidate = self.voice_dir / f"{response_key}.wav"
            if candidate.exists():
                wav_path = candidate

        self.last_spoken_text = text
        self.last_spoken_wav = wav_path

        # Notify any UI listeners (for live screen subtitles)
        for listener in self.listeners:
            try:
                listener(text, wav_path)
            except Exception:
                pass

        if self.muted:
            return

        if wav_path and wav_path.exists():
            self._play_wav(wav_path, blocking=blocking)
        else:
            self._speak_dynamic(text, blocking=blocking)

    def _play_wav(self, wav_path: Path, blocking: bool = False):
        """Asynchronously plays a WAV file via kernel audio drivers."""
        try:
            if sys.platform == "win32":
                import winsound
                flags = winsound.SND_FILENAME
                if not blocking:
                    flags |= winsound.SND_ASYNC
                winsound.PlaySound(str(wav_path), flags)
            elif sys.platform.startswith("linux"):
                # aplay handles ALSA natively without holding process locks
                cmd = f"aplay -q '{wav_path}'"
                if not blocking:
                    cmd += " &"
                os.system(cmd)
        except Exception as e:
            pass

    def _speak_dynamic(self, text: str, blocking: bool = False):
        """Offline dynamic speech synthesis fallback."""
        if not text:
            return

        def _worker():
            # 1. Windows SAPI
            if sys.platform == "win32" and self._sapi_speaker is not None:
                try:
                    # SAPI SVSFlagsAsync = 1
                    flags = 0 if blocking else 1
                    self._sapi_speaker.Speak(text, flags)
                    return
                except Exception:
                    pass

            # 2. Linux / Raspberry Pi: espeak-ng, flite, or spd-say
            if sys.platform.startswith("linux"):
                clean_text = text.replace("'", "").replace('"', "")
                bg = "" if blocking else " &"
                for cmd in [
                    f"espeak-ng -s 150 '{clean_text}' 2>/dev/null{bg}",
                    f"espeak -s 150 '{clean_text}' 2>/dev/null{bg}",
                    f"flite -t '{clean_text}' 2>/dev/null{bg}",
                    f"spd-say '{clean_text}' 2>/dev/null{bg}"
                ]:
                    if os.system(cmd) == 0:
                        return

        if blocking:
            _worker()
        else:
            threading.Thread(target=_worker, daemon=True).start()


# Global convenient singleton
VOICE_ENGINE = VoiceOutputEngine.get_instance()
