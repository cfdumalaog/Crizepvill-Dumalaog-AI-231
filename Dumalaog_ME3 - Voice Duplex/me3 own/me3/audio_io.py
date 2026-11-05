"""ME3 Audio I/O: full-duplex capture + playback (sounddevice).

Provides a streaming mic->frame generator and a non-blocking speaker sink
that supports instant stop (for barge-in). This is the transport layer the
local voice loop / Pipecat pipeline / VoiceDuplexAgent consume.

Device handling (why this is more than one line):
  - Windows exposes the same hardware through several host APIs (MME,
    DirectSound, WASAPI, WDM-KS) and some hosts are blocked or broken on a
    given machine while others work. `start_capture()` therefore probes
    candidate input devices (config device first, then each host's default
    input, then the rest) and uses the first one that actually opens.
  - Capture happens at the device's native rate and is resampled to the
    pipeline rate (16 kHz) inside the callback, so 48 kHz laptop mics work.
  - Playback is resampled to the output device's native rate.

Run standalone (loopback test):  python scripts/run_me3_module.py me3.audio_io
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time

import numpy as np

from .config import load_config

log = logging.getLogger("me3.audio_io")


def resample_to(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample int16 mono PCM between sample rates (soxr, high quality)."""
    if from_rate == to_rate:
        return pcm
    import soxr
    out = soxr.resample(pcm.astype(np.float32), from_rate, to_rate)
    return np.clip(out, -32768, 32767).astype(np.int16)


class AudioDeviceError(RuntimeError):
    """No usable audio device found after probing all candidates."""


class FullDuplexAudio:
    """Simultaneous mic capture + speaker playback with barge-in stop."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        a = self.cfg["audio"]
        self.sr = int(a["sample_rate"])
        self.frame_ms = int(a.get("frame_ms", 30))
        self.frame_n = int(self.sr * self.frame_ms / 1000)
        self.in_dev_cfg = a.get("input_device")
        self.out_dev_cfg = a.get("output_device")
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._sd = None
        self._stream = None
        self._play_stream = None
        self._playing = False
        self._play_buf = None  # flat int16 buffer at out rate
        self._play_pos = 0     # audio-thread cursor into _play_buf
        self._play_lock = threading.RLock()
        echo = dict(a.get("echo_guard", {}))
        self.echo_enabled = bool(echo.get("enabled", True))
        self.echo_delay_ms = float(echo.get("delay_ms", 80.0))
        self.echo_search_ms = float(echo.get("search_ms", 80.0))
        self.echo_search_step_ms = float(echo.get("search_step_ms", 10.0))
        self.echo_corr_threshold = float(echo.get("correlation_threshold", 0.45))
        self.echo_hard_gate_corr = float(echo.get("hard_gate_correlation", 0.75))
        self.echo_min_reference_rms = float(echo.get("min_reference_rms", 180.0))
        self.echo_tail_ms = float(echo.get("tail_ms", 250.0))
        self.echo_digital_loopback_hard_mute = bool(
            echo.get("digital_loopback_hard_mute", True))
        self.echo_max_gain = 10.0 ** (float(echo.get("max_gain_db", 18.0)) / 20.0)
        self._play_started_at: float | None = None
        self._play_ref: np.ndarray | None = None  # pipeline-rate playback reference
        self._echo_release_until = 0.0
        self._echo_cancelled_frames = 0
        self._echo_last_correlation = 0.0
        self._echo_peak_correlation = 0.0
        self._digital_loopback = False
        self.in_idx: int | None = None
        self.out_idx: int | None = None
        self.io_in_rate: int | None = None
        self.io_out_rate: int | None = None

    def _ensure_sd(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
        return self._sd

    # ---- Device resolution ---------------------------------------------------
    @staticmethod
    def _resolve_explicit(sd, wanted) -> int | None:
        if wanted is None:
            return None
        devs = sd.query_devices()
        if isinstance(wanted, int) and 0 <= wanted < len(devs):
            return int(wanted)
        s = str(wanted).lower()
        for i, d in enumerate(devs):
            if s == str(i):
                return i
            if s in str(d.get("name", "")).lower():
                return i
        return None

    def _input_candidates(self, sd) -> list[int]:
        """Ordered input-device candidates: explicit config, then each
        host API's default input, then every other input device."""
        devs = sd.query_devices()
        cands: list[int] = []
        explicit = self._resolve_explicit(sd, self.in_dev_cfg)
        if explicit is not None and devs[explicit].get("max_input_channels", 0) > 0:
            cands.append(explicit)
        for h in sd.query_hostapis():
            d = h.get("default_input_device")
            if d not in (None, -1) and d >= 0:
                cands.append(int(d))
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                cands.append(i)
        # de-duplicate, keep order
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _output_candidates(self, sd) -> list[int]:
        devs = sd.query_devices()
        cands: list[int] = []
        explicit = self._resolve_explicit(sd, self.out_dev_cfg)
        if explicit is not None and devs[explicit].get("max_output_channels", 0) > 0:
            cands.append(explicit)
        for h in sd.query_hostapis():
            d = h.get("default_output_device")
            if d not in (None, -1) and d >= 0:
                cands.append(int(d))
        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) > 0:
                cands.append(i)
        seen, out = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _probe_input(self, sd, idx: int) -> int | None:
        """Try to open+hear from device idx; return the working rate or None."""
        d = sd.query_devices(idx)
        rate = int(d.get("default_samplerate") or 48000)
        got = []

        def cb(indata, frames, t, status):
            got.append(1)

        for r in (rate, 48000, 16000):
            try:
                with sd.InputStream(samplerate=r, channels=1, dtype="int16",
                                    blocksize=max(480, int(r * 0.03)),
                                    device=idx, callback=cb):
                    time.sleep(0.15)
                return r
            except Exception:  # noqa: BLE001
                continue
        return None

    def _probe_output(self, sd, idx: int) -> int | None:
        d = sd.query_devices(idx)
        rate = int(d.get("default_samplerate") or 48000)
        for r in (rate, 48000, 22050):
            try:
                with sd.OutputStream(samplerate=r, channels=1, dtype="int16",
                                     device=idx):
                    pass
                return r
            except Exception:  # noqa: BLE001
                continue
        return None

    def resolve_devices(self, output_required: bool = True) -> None:
        """Find a working input and, optionally, output device.

        Browser-speaker mode still captures from the Pi's PortAudio input but
        deliberately has no Pi output device.  Keeping output resolution lazy
        also lets that mode run on a headless Pi without opening a dummy
        speaker.
        """
        sd = self._ensure_sd()
        if self.in_idx is None:
            for idx in self._input_candidates(sd):
                rate = self._probe_input(sd, idx)
                if rate is not None:
                    self.in_idx, self.io_in_rate = idx, rate
                    break
            if self.in_idx is None:
                raise AudioDeviceError(
                    "no working input device found; checked config device, "
                    "each host's default, and all inputs. Check Windows sound "
                    "settings / mic drivers.")
            name = str(sd.query_devices(self.in_idx).get("name", "")).lower()
            self._digital_loopback = any(token in name for token in (
                "stereo mix", "loopback", "what u hear"))
            if self._digital_loopback:
                log.info("digital loopback input detected: %s", name[:80])
        if output_required:
            self.resolve_output_device()

    def resolve_output_device(self) -> None:
        """Find a working output device without requiring a local microphone.

        Split-device test transports use this path: input can arrive over a
        network stream while TTS is played through this workstation's speaker.
        """
        sd = self._ensure_sd()
        if self.out_idx is None:
            for idx in self._output_candidates(sd):
                rate = self._probe_output(sd, idx)
                if rate is not None:
                    self.out_idx, self.io_out_rate = idx, rate
                    break
            if self.out_idx is None:
                raise AudioDeviceError("no working output device found")
        devs = sd.query_devices()
        log.info("speaker device: out=[%s] %s @ %d Hz -> %d Hz",
                 self.out_idx, str(devs[self.out_idx].get("name"))[:48],
                 self.io_out_rate, self.sr)

    # ---- Capture -----------------------------------------------------------
    @staticmethod
    def _rms(pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        x = pcm.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(x * x)))

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Return centered normalized correlation without allocating a matrix."""
        if a.size == 0 or b.size != a.size:
            return 0.0
        aa = a.astype(np.float32, copy=False)
        bb = b.astype(np.float32, copy=False)
        aa = aa - float(np.mean(aa))
        bb = bb - float(np.mean(bb))
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom <= 1e-6:
            return 0.0
        return float(np.dot(aa, bb) / denom)

    def _reference_slice(self, start: int, n: int) -> np.ndarray:
        """Return a zero-padded pipeline-rate playback reference slice."""
        out = np.zeros(n, dtype=np.float32)
        ref = self._play_ref
        if ref is None or n <= 0:
            return out
        lo = max(0, int(start))
        hi = min(len(ref), int(start) + n)
        if hi > lo:
            out[lo - int(start):hi - int(start)] = ref[lo:hi]
        return out

    def register_external_playback(self, pcm: np.ndarray) -> None:
        """Register audio played by a transport outside PortAudio.

        Browser-speaker mode publishes PCM over WebSocket instead of opening a
        local output stream. The capture-side echo guard still needs the exact
        playback reference so it can remove the agent's own voice before VAD
        and ASR see it.
        """
        ref = np.ascontiguousarray(pcm, dtype=np.int16).ravel()
        started = time.perf_counter()
        with self._play_lock:
            self._play_ref = ref
            self._play_started_at = started
            self._echo_release_until = started + (
                len(ref) / max(1, self.sr)) + self.echo_tail_ms / 1000.0
            self._playing = True

    def clear_external_playback(self) -> None:
        """End externally routed playback while retaining the echo tail."""
        with self._play_lock:
            self._playing = False
            self._echo_release_until = time.perf_counter() + \
                self.echo_tail_ms / 1000.0

    def _remove_speaker_echo(self, pcm: np.ndarray) -> np.ndarray:
        """Cancel the current speaker reference before VAD/ASR sees a frame.

        The microphone and speaker are independent PortAudio streams, so their
        callback clocks are not sample-aligned. We use a short delay search to
        find the best local match, estimate the acoustic coupling gain, and
        subtract only the correlated speaker component. A genuine barge-in
        voice is not identical to the playback reference and remains in the
        residual. The short tail covers room reflections after playback stops.
        """
        if not self.echo_enabled or pcm.size == 0:
            return pcm
        now = time.perf_counter()
        with self._play_lock:
            started = self._play_started_at
            active = self._playing or now < self._echo_release_until
            ref = self._play_ref
        if not active or started is None or ref is None:
            return pcm

        if self._digital_loopback and self.echo_digital_loopback_hard_mute:
            # Stereo Mix contains the system output rather than a user's
            # acoustic microphone signal. Muting it during ME3 playback is
            # deterministic and avoids VAD seeing device-transition frames.
            self._echo_cancelled_frames += 1
            return np.zeros_like(pcm)

        elapsed = now - started
        base = int((elapsed - self.echo_delay_ms / 1000.0) * self.sr)
        search = int(self.echo_search_ms / 1000.0 * self.sr)
        step = max(1, int(self.echo_search_step_ms / 1000.0 * self.sr))
        x = pcm.astype(np.float32, copy=False)
        best_corr = 0.0
        best_ref = None
        best_start = None
        for offset in range(-search, search + 1, step):
            candidate_start = base + offset
            candidate = self._reference_slice(candidate_start, len(x))
            if self._rms(candidate) < self.echo_min_reference_rms:
                continue
            corr = self._correlation(x, candidate)
            if abs(corr) > abs(best_corr):
                best_corr, best_ref, best_start = corr, candidate, candidate_start
        # The coarse search keeps the callback inexpensive over a wide device
        # jitter window. Refine the winning location at sample resolution; a
        # few milliseconds of misalignment otherwise leaves enough residual
        # speech to trigger VAD even when the waveform is highly correlated.
        if best_start is not None and step > 1:
            radius = step
            fine = self._reference_slice(
                best_start - radius, len(x) + 2 * radius)
            centered_x = x - float(np.mean(x))
            scores = np.correlate(fine, centered_x, mode="valid")
            idx = int(np.argmax(np.abs(scores)))
            refined = fine[idx:idx + len(x)]
            refined_corr = self._correlation(x, refined)
            if abs(refined_corr) >= abs(best_corr):
                best_corr, best_ref = refined_corr, refined
        self._echo_last_correlation = float(best_corr)
        self._echo_peak_correlation = max(
            self._echo_peak_correlation, abs(float(best_corr)))
        if best_ref is None or abs(best_corr) < self.echo_corr_threshold:
            return pcm

        # A high-correlation frame is the agent's own output in the common
        # Stereo Mix/digital-loopback case. Hard-gate it before Silero VAD;
        # lower-correlation frames still use residual cancellation so a user
        # speaking over the response can interrupt the agent.
        if abs(best_corr) >= self.echo_hard_gate_corr:
            self._echo_cancelled_frames += 1
            return np.zeros_like(pcm)

        ref_energy = float(np.dot(best_ref, best_ref))
        if ref_energy <= 1e-6:
            return pcm
        gain = float(np.dot(x, best_ref) / ref_energy)
        gain = float(np.clip(gain, -self.echo_max_gain, self.echo_max_gain))
        cleaned = x - gain * best_ref
        self._echo_cancelled_frames += 1
        # Keep a real user's unmatched speech. Only the correlated playback
        # component is removed; do not gate the entire frame during speaking.
        return np.clip(cleaned, -32768, 32767).astype(np.int16)

    def _enqueue_capture(self, pcm: np.ndarray) -> None:
        pcm = np.ascontiguousarray(pcm, dtype=np.int16).ravel()
        pcm = self._remove_speaker_echo(pcm)
        # Non-blocking: a blocked put() stalls the PortAudio callback
        # thread, which starves the output stream too and corrupts the
        # loopback capture after long agent turns (slow remote brain).
        # Drop the oldest frame to stay real-time.
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            try:
                self._q.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pcm)
            except queue.Full:
                pass  # still full: drop this frame

    def _on_input(self, indata, frames, t, status):
        if status:
            log.debug("input status: %s", status)
        pcm = indata[:, 0].copy()
        if self.io_in_rate != self.sr:
            pcm = resample_to(pcm, self.io_in_rate, self.sr)
        self._enqueue_capture(pcm)

    def start_capture(self):
        sd = self._ensure_sd()
        self.resolve_devices(output_required=False)
        block = int(self.io_in_rate * self.frame_ms / 1000)
        self._stream = sd.InputStream(
            samplerate=self.io_in_rate, channels=1, dtype="int16",
            blocksize=block, device=self.in_idx,
            callback=self._on_input,
        )
        self._stream.start()
        log.info("capture started (frame=%d ms)", self.frame_ms)

    def read_frame(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def startup_signal_check(self, seconds: float = 1.5) -> bool:
        """Capture a short window; True if any non-silence was heard."""
        t0 = time.perf_counter()
        peak = 0.0
        while time.perf_counter() - t0 < seconds:
            f = self.read_frame(timeout=0.25)
            if f is not None and len(f):
                peak = max(peak, float(np.abs(f).max()))
        if peak == 0.0:
            log.warning("microphone is delivering digital silence - check the "
                        "physical mic shutter/mute key and Windows mic settings")
            return False
        log.info("microphone signal OK (startup peak %.0f)", peak)
        return True

    # ---- Playback (barge-in capable) --------------------------------------
    def play(self, pcm: np.ndarray, blocking: bool = False):
        """Play int16 mono PCM at the pipeline rate. Non-blocking.

        Playback is a callback-driven OutputStream draining a flat sample
        buffer: the audio thread pulls frames from _play_buf, so multi-second
        buffers queue correctly (OutputStream.write() cannot) and
        playback_done() reads exhaustion from _play_pos (an explicit stream
        has no .finished attribute; .active stays True after the buffer
        drains, which previously hung callers forever).
        stop_playback() aborts instantly (barge-in).
        """
        sd = self._ensure_sd()
        if self.out_idx is None:
            self.resolve_output_device()
        self.stop_playback(silent=True)
        out = pcm if self.io_out_rate == self.sr else \
            resample_to(pcm, self.sr, self.io_out_rate)
        out = np.ascontiguousarray(out, dtype=np.int16).ravel()
        # Keep a pipeline-rate reference for acoustic echo cancellation. The
        # reference is retained briefly after playback for room reflections.
        ref = out if self.io_out_rate == self.sr else \
            resample_to(out, self.io_out_rate, self.sr)
        with self._play_lock:
            self._play_ref = np.ascontiguousarray(ref, dtype=np.int16).ravel()
            self._play_started_at = time.perf_counter()
            self._echo_release_until = self._play_started_at + \
                (len(out) / max(1, self.io_out_rate)) + self.echo_tail_ms / 1000.0
            self._play_buf = out
            self._play_pos = 0

        def _cb(outdata, frames, time_info, status):
            if status:
                log.debug("playback status: %s", status)
            with self._play_lock:
                buf, pos = self._play_buf, self._play_pos
            if buf is None:
                outdata[:, 0] = 0
                return
            end = min(pos + frames, len(buf))
            n = end - pos
            if n > 0:
                outdata[:n, 0] = buf[pos:end]
                if n < frames:
                    outdata[n:, 0] = 0
            else:
                outdata[:, 0] = 0
            with self._play_lock:
                self._play_pos = end

        stream = sd.OutputStream(samplerate=self.io_out_rate, channels=1,
                                 dtype="int16", device=self.out_idx,
                                 callback=_cb)
        stream.start()
        self._play_stream = stream
        self._playing = True
        log.info("playback started: %d samples (%.2f s)",
                 len(out), len(out) / self.io_out_rate)

    def playback_done(self) -> bool:
        """True when the queued audio has been played (or nothing queued)."""
        with self._play_lock:
            buf, pos = self._play_buf, self._play_pos
            done = buf is None or pos >= len(buf)
            if done and self._playing:
                self._playing = False
            return done

    def stop_playback(self, silent: bool = False):
        """Instantly stop the speaker (barge-in)."""
        with self._play_lock:
            stream, self._play_stream = self._play_stream, None
            self._playing = False
            self._echo_release_until = time.perf_counter() + \
                self.echo_tail_ms / 1000.0
            if self._play_buf is not None:
                self._play_pos = len(self._play_buf)  # mark fully consumed
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:  # noqa: BLE001
            pass
        if not silent:
            log.info("playback stopped (barge-in)")

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.stop_playback()
        with self._play_lock:
            self._play_ref = None
            self._play_started_at = None


class InjectedDuplexAudio:
    """Real-time PCM injection with a virtual, non-audible output sink.

    This transport is for an explicit evaluation lane where the speaker is
    unavailable or intentionally disabled. Input WAV/PCM is delivered as
    16-kHz frames at wall-clock cadence, so the normal VAD/ASR/brain/TTS
    control path sees the same frame timing as the live loop. TTS output is
    never sent to PortAudio; only safe timing metadata is retained. A virtual
    playback interval allows injected interrupt audio to exercise the same
    barge-in state transition without acoustic speaker feedback.
    """

    injection_only = True

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30):
        self.sr = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_n = int(self.sr * self.frame_ms / 1000)
        self.in_idx = "injected_pcm"
        self.out_idx = "virtual_discard_sink"
        self.io_in_rate = self.sr
        self.io_out_rate = self.sr
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._inject_threads: list[threading.Thread] = []
        self._closed = False
        self._play_lock = threading.Lock()
        self._playing = False
        self._play_until = 0.0
        self._play_token = 0
        self.inject_events: list[dict] = []
        self.output_events: list[dict] = []

    def start_capture(self):
        """Match the live transport API without opening a capture device."""
        return None

    def inject(self, pcm: np.ndarray, label: str = "input",
               start_at: float | None = None, tail_ms: int = 0) -> float:
        """Inject mono int16 PCM in real time and return its start timestamp.

        ``tail_ms`` appends digital silence so the normal VAD endpoint can
        close an injected utterance just as it would after a user stops
        speaking into a live microphone.
        """
        if self._closed:
            raise RuntimeError("injected audio transport is closed")
        x = np.asarray(pcm, dtype=np.int16).ravel().copy()
        if x.size == 0:
            raise ValueError("cannot inject empty PCM")
        start = float(start_at if start_at is not None
                      else time.perf_counter())
        tail_n = max(0, int(self.sr * int(tail_ms) / 1000))
        stream = (np.concatenate([x, np.zeros(tail_n, dtype=np.int16)])
                  if tail_n else x)
        event = {
            "label": str(label),
            "start_abs": start,
            "samples": int(x.size),
            "duration_s": round(float(x.size / self.sr), 3),
            "tail_s": round(float(tail_n / self.sr), 3),
        }
        self.inject_events.append(event)

        def _feed() -> None:
            first = start
            for offset in range(0, len(stream), self.frame_n):
                target = first + (offset / self.sr)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if self._closed:
                    return
                frame = stream[offset:offset + self.frame_n]
                if len(frame) < self.frame_n:
                    frame = np.pad(frame, (0, self.frame_n - len(frame)))
                self._q.put(np.ascontiguousarray(frame, dtype=np.int16))

        thread = threading.Thread(target=_feed, daemon=True,
                                  name=f"me3-inject-{label}")
        self._inject_threads.append(thread)
        thread.start()
        return start

    def read_frame(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def play(self, pcm: np.ndarray, blocking: bool = False):
        """Record virtual TTS timing and discard samples instead of playing."""
        x = np.asarray(pcm, dtype=np.int16).ravel()
        started = time.perf_counter()
        duration = float(len(x) / self.sr) if self.sr else 0.0
        with self._play_lock:
            self._play_token += 1
            token = self._play_token
            self._playing = True
            self._play_until = started + duration
            self.output_events.append({
                "token": token,
                "started_abs": started,
                "samples": int(len(x)),
                "duration_s": round(duration, 3),
                "sink": "virtual_discard",
                "stopped_abs": None,
            })
        if blocking:
            while not self.playback_done():
                time.sleep(min(0.02, max(0.001, duration)))

    def playback_done(self) -> bool:
        with self._play_lock:
            if self._playing and time.perf_counter() >= self._play_until:
                self._playing = False
            return not self._playing

    def stop_playback(self, silent: bool = False):
        with self._play_lock:
            if not self._playing:
                return
            stopped = time.perf_counter()
            self._playing = False
            self._play_until = stopped
            if self.output_events:
                self.output_events[-1]["stopped_abs"] = stopped

    def injections_done(self) -> bool:
        return (all(not t.is_alive() for t in self._inject_threads)
                and self._q.empty())

    def close(self):
        self._closed = True
        self.stop_playback(silent=True)


class RemoteMicAudio(FullDuplexAudio):
    """Use a Raspberry Pi microphone as input and this host's speaker output.

    The Pi runs ``arecord`` on the selected capture device and sends raw
    signed 16-bit mono PCM through an SSH stdout pipe.  TTS remains local and
    uses the normal PortAudio output path, so this is a deliberate local
    split-device acceptance lane, not the Pi production transport.
    """

    def __init__(self, config: dict | None = None,
                 ssh_host: str = "dalmacio@cfdfnjrpi5.local",
                 remote_device: str = "hw:2,0", remote_channels: int = 2):
        super().__init__(config)
        self.ssh_host = str(ssh_host)
        self.remote_device = str(remote_device)
        self.remote_channels = max(1, int(remote_channels))
        self.in_idx = f"ssh:{self.ssh_host}:{self.remote_device}"
        self.io_in_rate = self.sr
        self._remote_proc: subprocess.Popen | None = None
        self._remote_thread: threading.Thread | None = None
        self._remote_closed = False

    def remote_command(self) -> list[str]:
        """Return the non-secret SSH/ALSA capture command for inspection."""
        return [
            "ssh", "-T", self.ssh_host,
            "arecord", "-q", "-D", self.remote_device,
            "-f", "S16_LE", "-r", str(self.sr),
            "-c", str(self.remote_channels),
            "-t", "raw", "-",
        ]

    def start_capture(self):
        if self._remote_proc is not None:
            return
        self.resolve_output_device()
        self._remote_closed = False
        self._remote_proc = subprocess.Popen(
            self.remote_command(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        assert self._remote_proc.stdout is not None

        def _read_remote() -> None:
            bytes_per_frame = (self.frame_n * self.remote_channels
                               * np.dtype(np.int16).itemsize)
            stdout = self._remote_proc.stdout
            while not self._remote_closed:
                data = stdout.read(bytes_per_frame)
                if not data:
                    break
                frame = np.frombuffer(data, dtype=np.int16).copy()
                if self.remote_channels > 1:
                    usable = (frame.size // self.remote_channels) \
                        * self.remote_channels
                    frame = frame[:usable].reshape(
                        -1, self.remote_channels).mean(axis=1).astype(np.int16)
                if frame.size < self.frame_n:
                    frame = np.pad(frame, (0, self.frame_n - frame.size))
                self._enqueue_capture(frame)
            log.info("remote microphone stream ended")

        self._remote_thread = threading.Thread(
            target=_read_remote, daemon=True, name="me3-remote-mic")
        self._remote_thread.start()
        log.info("remote microphone started: %s (%s)",
                 self.ssh_host, self.remote_device)

    def close(self):
        self._remote_closed = True
        proc, self._remote_proc = self._remote_proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        super().close()


def main():
    logging.basicConfig(level=logging.INFO)
    audio = FullDuplexAudio()
    audio.start_capture()
    audio.startup_signal_check()
    log.info("Loopback test: speaking mic to speaker. Ctrl+C to stop.")
    try:
        while True:
            frame = audio.read_frame()
            if frame is not None:
                audio.play(frame, blocking=False)
    except KeyboardInterrupt:
        audio.close()


if __name__ == "__main__":
    main()
