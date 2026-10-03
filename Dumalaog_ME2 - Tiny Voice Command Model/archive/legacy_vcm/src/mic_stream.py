"""
Robust Multi-Platform Hardware Microphone Streaming Engine.
Provides seamless, fault-tolerant audio capture across Windows (MME/DirectSound/WASAPI/WDM-KS)
and Linux/Raspberry Pi (ALSA/PulseAudio/PipeWire).

Automatically:
1. Probes and tests candidate input devices to find the true working microphone.
2. Captures at hardware native sample rates (44.1kHz, 48kHz, etc.) and channel counts (mono, stereo, 4-ch array).
3. Downsamples and converts to pristine 16,000 Hz float32 mono.
4. Applies Adaptive Digital Gain Control (AGC) so quiet laptop microphones are boosted cleanly.
"""

import sys
import time
from typing import Optional, Callable, Tuple, List, Dict, Any
import numpy as np
import scipy.signal
import sounddevice as sd

from .config import SAMPLE_RATE, STREAM_HOP_SEC


class RobustMicrophoneStreamer:
    """
    Intelligent hardware microphone streamer that eliminates PortAudio
    host errors (e.g. Windows MME error 1, DirectSound errors, or channel mismatches).
    """
    def __init__(
        self,
        target_sample_rate: int = SAMPLE_RATE,
        hop_seconds: float = STREAM_HOP_SEC,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        apply_agc: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.hop_seconds = hop_seconds
        self.callback = callback
        self.apply_agc = apply_agc

        self.target_hop_samples = int(target_sample_rate * hop_seconds)
        self.is_running = False
        self.stream: Optional[sd.InputStream] = None

        # Device details
        self.device_id: Optional[int] = None
        self.device_name: str = "Unknown"
        self.native_sample_rate: int = 16000
        self.channels: int = 1
        self.host_api_name: str = ""

        # Internal sample accumulator for rate conversion
        self._accumulator = np.zeros(0, dtype=np.float32)

        # Telemetry
        self.last_rms: float = 0.0
        self.last_max: float = 0.0

        self._probe_and_select_device()

    def _probe_and_select_device(self):
        """Intelligently detects and validates the best working hardware microphone."""
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        candidates: List[Tuple[int, int, Dict[str, Any]]] = []

        for idx, d in enumerate(devices):
            if d['max_input_channels'] <= 0:
                continue

            name_lower = d['name'].lower()
            score = 0

            # Prioritize built-in or USB microphone arrays
            if 'array' in name_lower:
                score += 20
            if 'microphone' in name_lower or 'mic' in name_lower:
                score += 10
            if 'usb' in name_lower:
                score += 15

            # Host API preferences on Windows
            if sys.platform == "win32":
                # WDM-KS and WASAPI bypass MME channel mapping bugs
                if d['hostapi'] == 3:  # Windows WDM-KS
                    score += 8
                elif d['hostapi'] == 2:  # Windows WASAPI
                    score += 5
                elif d['hostapi'] == 1:  # DirectSound
                    score += 3
            else:
                # Linux / ALSA
                if 'hw:' in name_lower or 'plughw' in name_lower:
                    score += 10

            # Deprioritize output loopbacks or virtual lines unless nothing else exists
            if 'speaker' in name_lower or 'stereo mix' in name_lower or 'virtual' in name_lower:
                score -= 30

            candidates.append((score, idx, d))

        # Sort descending by score
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Test opening a brief stream to ensure hardware accepts the parameters
        for _, dev_id, d in candidates:
            native_sr = int(d['default_samplerate'])
            max_ch = d['max_input_channels']

            # Test configurations from native rate down to 16kHz
            test_configs = [
                (native_sr, 1),
                (native_sr, min(2, max_ch)),
                (16000, 1),
                (48000, 1),
                (44100, 1)
            ]

            for sr, ch in test_configs:
                try:
                    captured = []
                    def _test_cb(indata, frames, time_info, status):
                        captured.append(True)

                    with sd.InputStream(
                        device=dev_id,
                        samplerate=sr,
                        channels=ch,
                        dtype="float32",
                        blocksize=int(sr * 0.05),
                        callback=_test_cb
                    ):
                        sd.sleep(50)

                    if len(captured) > 0:
                        self.device_id = dev_id
                        self.device_name = d['name']
                        self.native_sample_rate = sr
                        self.channels = ch
                        self.host_api_name = hostapis[d['hostapi']]['name']
                        return
                except Exception:
                    continue

        # Fallback to system default if probe didn't select
        default_in = sd.default.device[0]
        self.device_id = default_in if default_in >= 0 else None
        self.device_name = "Default System Microphone"
        self.native_sample_rate = 16000
        self.channels = 1

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any):
        """Processes incoming hardware audio frames, resamples to 16kHz, and invokes callback."""
        if not self.is_running:
            return

        # 1. Downmix multi-channel to mono
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0]

        self._accumulator = np.concatenate([self._accumulator, mono])

        # 2. Resample to 16,000 Hz when enough samples have accumulated
        native_hop_samples = int(self.native_sample_rate * self.hop_seconds)

        while len(self._accumulator) >= native_hop_samples:
            raw_chunk = self._accumulator[:native_hop_samples]
            self._accumulator = self._accumulator[native_hop_samples:]

            if self.native_sample_rate != self.target_sample_rate:
                chunk_16k = scipy.signal.resample(raw_chunk, self.target_hop_samples).astype(np.float32)
            else:
                chunk_16k = raw_chunk.astype(np.float32)

            # 3. Adaptive Gain Control (AGC): Cleanly boost quiet laptop microphones
            if self.apply_agc:
                peak = np.max(np.abs(chunk_16k))
                if 0.001 < peak < 0.30:
                    boost = min(6.0, 0.40 / max(peak, 0.01))
                    chunk_16k = np.clip(chunk_16k * boost, -1.0, 1.0)

            self.last_rms = float(np.sqrt(np.mean(chunk_16k ** 2) + 1e-12))
            self.last_max = float(np.max(np.abs(chunk_16k)))

            if self.callback:
                self.callback(chunk_16k)

    def start(self):
        """Starts background hardware audio capture."""
        if self.is_running:
            return

        self._accumulator = np.zeros(0, dtype=np.float32)
        self.is_running = True

        blocksize = int(self.native_sample_rate * 0.05)  # Low 50ms buffer to prevent latency
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.native_sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stops hardware audio capture."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_vu_meter(self, num_bars: int = 12) -> Tuple[str, float]:
        """Returns visual ASCII VU meter string and current RMS amplitude."""
        bars = int(min(num_bars, max(0, self.last_rms * 250)))
        vu_str = "=" * bars + "-" * (num_bars - bars)
        return f"[{vu_str}]", self.last_rms
