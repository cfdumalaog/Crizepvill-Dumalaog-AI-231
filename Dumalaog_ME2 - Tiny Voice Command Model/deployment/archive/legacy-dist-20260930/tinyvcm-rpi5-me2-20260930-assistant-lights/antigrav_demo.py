"""ME2 local voice assistant with binary wake gating and coded actions."""
import argparse
from collections import deque
from datetime import datetime
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import sys
import threading
import time
import urllib.parse

import numpy as np
import onnxruntime as ort

# Add parent directory to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinyvcm_antigrav.config import LABELS, WAKE_LABELS, SAMPLES, SR, MELS, TIME_STEPS
from tinyvcm_antigrav.frontend import Frontend, fit_audio
from tinyvcm_antigrav.media_engine import MediaEngine
from tinyvcm.devices import Devices
from tinyvcm.microphone import Microphone
from tinyvcm.restart import queue_restart


LIGHT_DEMO_COMMANDS = frozenset({"LIGHT_OFF", "LIGHT_ON", "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE"})


def apply_light_demo_command(devices, command):
    """Apply only a fixed light-demo action from the local Assistant UI."""
    if not isinstance(command, str) or command not in LIGHT_DEMO_COMMANDS:
        raise ValueError("Unsupported light demo command")
    return devices.execute(command)


def arguments_with_selected_device(arguments, selection):
    """Replace the CLI microphone selector without treating UI text as shell code."""
    if selection not in ('auto',) and not (isinstance(selection, str) and selection.isdecimal()):
        raise ValueError('Choose a listed microphone or Automatic selection')
    cleaned = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument == '--device':
            skip_next = True
            continue
        if not argument.startswith('--device='):
            cleaned.append(argument)
    if selection != 'auto':
        cleaned.extend(['--device', selection])
    return cleaned

class AntigravPredictor:
    """Zero-torch ONNX predictor; optionally use a separate binary standby gate."""
    def __init__(self, model_path=None, binary_wake_model_path=None, wake_threshold=None):
        if not model_path:
            model_path = ROOT / 'deployment' / 'current_vcm' / 'models' / 'intent_int8.onnx'
            if binary_wake_model_path is None:
                binary_wake_model_path = ROOT / 'deployment' / 'current_vcm' / 'models' / 'binary_wake_int8.onnx'
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        out_shape = self.session.get_outputs()[0].shape
        num_classes = out_shape[-1] if (isinstance(out_shape[-1], int) and out_shape[-1] > 0) else 32
        if num_classes not in (len(LABELS), len(WAKE_LABELS)):
            raise ValueError(f"Unsupported model output size {num_classes}; expected {len(LABELS)} or {len(WAKE_LABELS)} classes")
        self.frontend = Frontend()
        self.labels = WAKE_LABELS if num_classes == 32 else LABELS
        self.binary_wake_enabled = binary_wake_model_path is not None
        self.binary_wake_model_path = Path(binary_wake_model_path) if binary_wake_model_path is not None else None
        self.wake_threshold = 0.60
        self.wake_session = None
        self.wake_input_name = None
        if self.binary_wake_enabled:
            if len(self.labels) != len(LABELS):
                raise ValueError("Separate binary wake mode requires the 31-class command model")
            if not self.binary_wake_model_path.is_file():
                raise FileNotFoundError(f"Binary wake model not found: {self.binary_wake_model_path}")
            self.wake_session = ort.InferenceSession(str(self.binary_wake_model_path), opts, providers=['CPUExecutionProvider'])
            wake_shape = self.wake_session.get_outputs()[0].shape
            wake_classes = wake_shape[-1] if (isinstance(wake_shape[-1], int) and wake_shape[-1] > 0) else None
            if wake_classes != 2:
                raise ValueError(f"Binary wake model must have 2 outputs; got {wake_classes}")
            self.wake_input_name = self.wake_session.get_inputs()[0].name
            if wake_threshold is None:
                summary_path = self.binary_wake_model_path.parent.parent / 'export_summary.json'
                if not summary_path.is_file():
                    raise ValueError("Pass --wake-threshold or keep export_summary.json beside the experiment model")
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                # Keep the measured validation threshold intact, but honor a
                # separately recorded local operating override when present.
                wake_threshold = summary.get('local_operating_threshold')
                if wake_threshold is None:
                    wake_threshold = summary.get('validation_selected_threshold')
            if wake_threshold is None or not 0.0 <= float(wake_threshold) <= 1.0:
                raise ValueError(f"Invalid binary wake threshold: {wake_threshold}")
            self.wake_threshold = float(wake_threshold)
        self.model_size_kb = round((self.model_path.stat().st_size + (self.binary_wake_model_path.stat().st_size if self.binary_wake_enabled else 0)) / 1024, 1)
        self.model_name = (
            'TinyDSCNN-48 Binary Wake + TinyDSCNN-48 Intent (31 Classes)'
            if self.binary_wake_enabled else f'TinyDSCNN-48 ({len(self.labels)} Classes)'
        )

    def predict(self, audio, mode='command'):
        t0 = time.perf_counter()
        fitted = fit_audio(audio, target_samples=SAMPLES)
        features = self.frontend(fitted)[None]  # Shape: (1, 1, 40, 251)
        if mode == 'wake':
            if not self.binary_wake_enabled:
                raise ValueError("Wake-only inference requires a separate binary wake model")
            logits = self.wake_session.run(None, {self.wake_input_name: features})[0][0]
            labels = ['NON_WAKE', 'WAKE_WORD']
        elif mode == 'command':
            logits = self.session.run(None, {self.input_name: features})[0][0]
            labels = self.labels
        else:
            raise ValueError(f"Unsupported prediction mode: {mode}")
        latency_ms = (time.perf_counter() - t0) * 1000
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        order = np.argsort(probs)[::-1]
        top_idx = order[0]
        second_idx = order[1]
        if mode == 'wake':
            wake_probability = float(probs[1])
            label = 'WAKE_WORD' if wake_probability >= self.wake_threshold else 'NON_WAKE'
            confidence = wake_probability if label == 'WAKE_WORD' else float(probs[0])
            top3 = {labels[i]: float(probs[i]) for i in order}
        else:
            label = labels[top_idx]
            confidence = float(probs[top_idx])
            top3 = {labels[i]: float(probs[i]) for i in order[:3]}

        return {
            'label': label,
            'confidence': confidence,
            'margin': float(probs[top_idx] - probs[second_idx]),
            'wake_probability': float(probs[1]) if mode == 'wake' else (float(probs[self.labels.index('WAKE_WORD')]) if 'WAKE_WORD' in self.labels else None),
            'latency_ms': latency_ms,
            'top3': top3
        }

class AntigravAssistant:
    def __init__(self, gpio=False, device=None, timeout_sec=10.0, model_path=None, binary_wake_model_path=None, wake_threshold=None, vad_threshold=0.012, inference_interval_sec=0.35, live_weather=False, wake_confirmations=1):
        if not np.isfinite(vad_threshold) or not 0.0 <= float(vad_threshold) <= 1.0:
            raise ValueError(f"Invalid VAD RMS threshold: {vad_threshold}")
        if not np.isfinite(inference_interval_sec) or float(inference_interval_sec) <= 0.0:
            raise ValueError(f"Invalid inference interval: {inference_interval_sec}")
        if int(wake_confirmations) != wake_confirmations or int(wake_confirmations) < 1:
            raise ValueError(f"Invalid wake confirmations: {wake_confirmations}")
        self.predictor = AntigravPredictor(model_path=model_path, binary_wake_model_path=binary_wake_model_path, wake_threshold=wake_threshold)
        self.devices = Devices(gpio=gpio, live_weather=live_weather)
        self.weather_lock = threading.Lock()
        self.mic = Microphone(device=device)
        self.microphone_lock = threading.RLock()
        self.mic_retry_interval_sec = 2.0
        self.next_mic_retry_at = 0.0
        self.microphone_retry_enabled = False
        self.events = deque(maxlen=30)
        self.stop_event = threading.Event()
        self.error = None
        self.muted = False
        self.worker = None
        
        # State Machine: 'STANDBY' -> 'LISTENING' -> 'TIMEOUT'
        self.state = 'STANDBY'
        self.timeout_sec = timeout_sec
        self.inactivity_deadline = 0.0
        self.cooldown_until = 0.0
        
        self.latest_prediction = {'label': None, 'confidence': 0.0, 'latency_ms': 0.0}
        self.rms = 0.0
        self.vad_threshold = float(vad_threshold)
        self.inference_interval_sec = float(inference_interval_sec)
        self.inference_count = 0
        self.audio_chunks_processed = 0
        self.last_audio_time = 0.0
        self.wake_confirmations = int(wake_confirmations)
        self.wake_streak = 0
        self.last_wake_hit = 0.0
        self.utterance_end_silence_sec = 0.35
        self.speech_active = False
        self.silence_samples = 0
        self.ignore_current_utterance = False
        self.command_armed = False
        self.pending_command = False
        
        # Ring buffer for 2.5 seconds of audio (40,000 samples)
        self.buffer = np.zeros(SAMPLES, dtype=np.float32)
        self.last_inference = 0.0

    def trigger_wake(self, source="manual"):
        now = time.monotonic()
        self.state = 'LISTENING'
        self.wake_streak = 0
        self.ignore_current_utterance = self.speech_active
        self.command_armed = not self.speech_active
        self.pending_command = False
        self.devices.ducked = True
        self.inactivity_deadline = now + self.timeout_sec
        line = f"{time.strftime('%H:%M:%S')} [WAKE] Activated via {source}! Listening for commands ({self.timeout_sec:.0f}s timeout)..."
        self.events.appendleft(line)
        self.devices.message = "Listening! Say a command now (e.g. 'Play music', 'Weather', 'Turn on lights')."
        print(f"[State] {line}", flush=True)

    def trigger_timeout(self):
        self.state = 'STANDBY'
        self.wake_streak = 0
        self.ignore_current_utterance = False
        self.command_armed = False
        self.pending_command = False
        self.devices.ducked = False
        line = f"{time.strftime('%H:%M:%S')} [TIMEOUT] Inactivity timeout ({self.timeout_sec:.0f}s elapsed). Returned to Standby."
        self.events.appendleft(line)
        self.devices.message = "Standby. Say 'Hi Dandan' to wake me up."
        print(f"[State] {line}", flush=True)

    def start_weather_lookup(self, label, confidence, latency_ms):
        """Keep a network weather request off the microphone/inference thread."""
        if not self.weather_lock.acquire(blocking=False):
            return 'Live weather lookup already in progress.'

        def lookup():
            try:
                try:
                    msg = self.devices.execute(label)
                except Exception:
                    msg = 'Live weather is unavailable right now. Check the internet connection or weather service.'
                line = f"{time.strftime('%H:%M:%S')} [{label}] ({confidence*100:.1f}%, {latency_ms:.1f}ms) -> {msg}"
                self.events.appendleft(line)
                self.devices.message = msg
                print(f"[Command] {line}", flush=True)
            finally:
                self.weather_lock.release()

        threading.Thread(target=lookup, name='vcm-weather', daemon=True).start()
        return 'Checking live weather...'

    def _try_start_microphone(self):
        """Open the selected/automatic input when available; keep the UI usable without one."""
        with self.microphone_lock:
            return self._try_start_microphone_locked()

    def _try_start_microphone_locked(self):
        stream = self.mic.stream
        if stream is not None:
            try:
                if stream.active:
                    return True
            except Exception:
                pass
            try:
                self.mic.close()
            except Exception:
                self.mic.stream = None
        try:
            self.mic.start()
        except Exception as exc:
            self.mic.name = 'No microphone connected'
            self.mic.status = str(exc)
            self.next_mic_retry_at = time.monotonic() + self.mic_retry_interval_sec
            print(f"[Audio] {self.mic.status}; retrying in {self.mic_retry_interval_sec:.0f}s", flush=True)
            return False
        self.next_mic_retry_at = 0.0
        print(f"[Antigrav] Microphone: {self.mic.name}", flush=True)
        return True

    def select_microphone(self, selection):
        """Switch capture devices in the listener without replacing the model/session."""
        if selection != 'auto' and not (isinstance(selection, str) and selection.isdecimal()):
            raise ValueError('Choose Automatic or a listed microphone index')
        with self.microphone_lock:
            self.mic.close()
            self.mic.requested_device = None if selection == 'auto' else int(selection)
            self.mic.device = None
            self.buffer.fill(0)
            self.last_audio_time = 0.0
            self.last_inference = 0.0
            self.speech_active = False
            self.pending_command = False
            if self.state == 'LISTENING':
                self.trigger_timeout()
            self.next_mic_retry_at = 0.0
            ready = self._try_start_microphone_locked()
            return {'selected': selection, 'capture_ready': ready, 'microphone': self.mic.name}

    def start(self):
        # Keep the local UI/model available without a connected microphone.
        # The listener retries and automatically starts when an input appears.
        self.microphone_retry_enabled = True
        microphone_ready = self._try_start_microphone()
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.devices.message = (
            "Standby. Say 'Hi Dandan' to wake me up."
            if microphone_ready else "VCM is ready. Connect or select a microphone to begin listening."
        )
        print(f"[Antigrav] Model: {self.predictor.model_name} ({self.predictor.model_size_kb} KB)", flush=True)
        if self.predictor.binary_wake_enabled:
            print(f"[Antigrav] Binary wake threshold: {self.predictor.wake_threshold:.6f}", flush=True)
        print(f"[Antigrav] State: STANDBY. Waiting for 'Hi Dandan'...", flush=True)

    def loop(self):
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()

                with self.microphone_lock:
                    stream = self.mic.stream
                    try:
                        active = bool(stream and stream.active)
                    except Exception:
                        active = False
                    stale = bool(active and self.mic.last_callback_time and now - self.mic.last_callback_time > 3.0)
                    if stale:
                        self.mic.close()
                        active = False
                        self.mic.status = 'Microphone stopped sending audio; reconnecting.'
                    if self.microphone_retry_enabled and not active and now >= self.next_mic_retry_at:
                        self._try_start_microphone_locked()
                        stream = self.mic.stream
                        try: active = bool(stream and stream.active)
                        except Exception: active = False
                    if not active:
                        chunk = None
                    else:
                        chunk = self.mic.read()
                if not active:
                    self.devices.tick()
                    self.stop_event.wait(0.2)
                    continue
                
                # Check inactivity timeout when in LISTENING state
                if self.state == 'LISTENING' and now >= self.inactivity_deadline and not self.speech_active and not self.pending_command:
                    self.trigger_timeout()

                if chunk is not None and len(chunk) > 0:
                    self.audio_chunks_processed += 1
                    self.last_audio_time = time.monotonic()
                    self.rms = float(np.sqrt(np.mean(chunk**2)))

                    # Wake detection is streaming; command classification waits for an
                    # utterance endpoint so partial phrases cannot fire multiple actions.
                    if self.rms > self.vad_threshold:
                        self.speech_active = True
                        self.silence_samples = 0
                    elif self.speech_active:
                        self.silence_samples += len(chunk)
                        if self.silence_samples >= int(SR * self.utterance_end_silence_sec):
                            self.speech_active = False
                            self.silence_samples = 0
                            if self.state == 'LISTENING':
                                if self.ignore_current_utterance:
                                    self.ignore_current_utterance = False
                                    self.command_armed = True
                                elif self.command_armed:
                                    self.pending_command = True
                    
                    # Update ring buffer
                    n = len(chunk)
                    if n >= SAMPLES:
                        self.buffer[:] = chunk[-SAMPLES:]
                    else:
                        self.buffer[:-n] = self.buffer[n:]
                        self.buffer[-n:] = chunk
                    
                    # VAD check
                    wake_frame_ready = self.state == 'STANDBY' and self.rms > self.vad_threshold and (now - self.last_inference) > self.inference_interval_sec
                    command_ready = self.state == 'LISTENING' and self.pending_command
                    if not self.muted and (wake_frame_ready or command_ready) and now > self.cooldown_until:
                        if self.predictor.binary_wake_enabled:
                            mode = 'wake' if self.state == 'STANDBY' else 'command'
                            pred = self.predictor.predict(self.buffer, mode=mode)
                        else:
                            mode = 'command'
                            pred = self.predictor.predict(self.buffer)
                        pred['mode'] = mode
                        self.latest_prediction = pred
                        self.inference_count += 1
                        self.last_inference = now
                        if mode == 'command':
                            self.pending_command = False
                        label = pred['label']
                        conf = pred['confidence']
                        margin = pred['margin']

                        # --- STATE MACHINE TRANSITIONS ---
                        if self.state == 'STANDBY':
                            # In STANDBY: ONLY accept WAKE_WORD
                            wake_threshold = self.predictor.wake_threshold if self.predictor.binary_wake_enabled else 0.60
                            if label == 'WAKE_WORD' and conf >= wake_threshold:
                                self.wake_streak = self.wake_streak + 1 if now - self.last_wake_hit <= 0.45 else 1
                                self.last_wake_hit = now
                                if self.wake_streak >= self.wake_confirmations:
                                    self.trigger_wake(source="Voice ('Hi Dandan')")
                                    self.cooldown_until = now + 1.2
                            else:
                                self.wake_streak = 0

                        elif self.state == 'LISTENING':
                            # In LISTENING: accept any command class (or re-wake)
                            if label == 'WAKE_WORD':
                                self.inactivity_deadline = now + self.timeout_sec
                                self.cooldown_until = now + 0.8
                            elif conf >= 0.68 and margin >= 0.15:
                                # Command accepted!
                                msg = (self.start_weather_lookup(label, conf, pred['latency_ms'])
                                       if self.devices.live_weather and label in ('WEATHER', 'question_weather')
                                       else self.devices.execute(label))
                                line = f"{time.strftime('%H:%M:%S')} [{label}] ({conf*100:.1f}%, {pred['latency_ms']:.1f}ms) -> {msg}"
                                self.events.appendleft(line)
                                print(f"[Command] {line}", flush=True)
                                self.inactivity_deadline = now + self.timeout_sec
                                self.cooldown_until = now + 1.2
                
                # Update device timers / ticks
                self.devices.tick()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[Error] Audio loop crashed: {self.error}", flush=True)

    def snapshot(self):
        now = time.monotonic()
        remaining_sec = max(0.0, self.inactivity_deadline - now) if self.state == 'LISTENING' else 0.0
        return {
            'state': 'ERROR' if self.error else 'MUTED' if self.muted else self.state,
            'remaining_sec': round(remaining_sec, 1),
            'timeout_sec': self.timeout_sec,
            'rms': self.rms,
            'vad_threshold': self.vad_threshold,
            'inference_interval_sec': self.inference_interval_sec,
            'inference_count': self.inference_count,
            'wake_confirmations': self.wake_confirmations,
            'wake_streak': self.wake_streak,
            'speech_active': self.speech_active,
            'command_armed': self.command_armed,
            'utterance_end_silence_sec': self.utterance_end_silence_sec,
            'latest': self.latest_prediction,
            'message': self.devices.message,
            'microphone': self.mic.name,
            'microphone_device_index': self.mic.device,
            'microphone_requested_device': self.mic.requested_device,
            'microphone_stream_active': bool(self.mic.stream and self.mic.stream.active),
            'microphone_status': self.mic.status,
            'audio_blocks_received': self.mic.blocks_received,
            'audio_chunks_processed': self.audio_chunks_processed,
            'audio_age_sec': round(now - self.last_audio_time, 2) if self.last_audio_time else None,
            'dropped': self.mic.dropped,
            'error': self.error,
            'events': list(self.events),
            'devices': self.devices.snapshot(),
            'model_kb': self.predictor.model_size_kb,
            'model_name': self.predictor.model_name,
            'classes_count': len(self.predictor.labels),
            'binary_wake_enabled': self.predictor.binary_wake_enabled,
            'wake_threshold': self.predictor.wake_threshold if self.predictor.binary_wake_enabled else 0.60,
            'live_weather': self.devices.live_weather,
        }

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=2)
        with self.microphone_lock:
            self.mic.close()
        self.devices.close()

ASSISTANT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <title>ME2 - VCM on Raspberry Pi 5</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #06090e;
      --card-bg: rgba(16, 24, 38, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --accent-gold: #f59e0b;
      --accent-red: #ef4444;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
    body {
      background: radial-gradient(circle at 50% 35%, #0f1929 0%, #080d14 60%, var(--bg) 100%);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
      height: 100vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 20px 32px;
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #64748b;
      box-shadow: 0 0 10px rgba(100, 116, 139, 0.5);
      transition: all 0.3s;
    }
    .status-dot.active {
      background: var(--accent-green);
      box-shadow: 0 0 16px var(--accent-green);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .pill-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 16px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      backdrop-filter: blur(12px);
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }
    .pill-btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }
    .pill-btn.active {
      background: rgba(16, 185, 129, 0.15);
      border-color: var(--accent-green);
      color: #34d399;
    }
    .device-select {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      max-width: 190px;
      font: 12px 'Plus Jakarta Sans', system-ui, sans-serif;
    }
    .device-select option { background: #101826; color: var(--text); }
    .audio-settings { position: relative; }
    .audio-settings > summary { list-style: none; }
    .audio-settings > summary::-webkit-details-marker { display: none; }
    .audio-popover {
      position: absolute;
      z-index: 60;
      right: 0;
      top: calc(100% + 10px);
      display: grid;
      gap: 10px;
      width: min(330px, calc(100vw - 32px));
      padding: 14px;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      background: #101826;
      box-shadow: 0 18px 45px rgba(0,0,0,.55);
    }
    .audio-popover label { display: grid; gap: 5px; color: var(--text-muted); font-size: 11px; }
    .audio-popover small { color: var(--text-muted); font-size: 10px; line-height: 1.4; }
    .clock-pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 999px;
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    main {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      position: relative;
      gap: 8px;
      padding: 6px 20px 14px;
      text-align: center;
      z-index: 10;
      overflow-y: auto;
    }
    .orb-stage {
      position: relative;
      width: 240px;
      height: 224px;
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }
    canvas#orbCanvas {
      width: 224px;
      height: 224px;
      border-radius: 50%;
    }
    .orb-hint {
      position: absolute;
      bottom: -10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: var(--text-muted);
      opacity: 0.7;
      transition: opacity 0.3s;
    }
    .orb-stage:hover .orb-hint {
      opacity: 1;
      color: #fff;
    }
    .subtitles-container {
      margin-top: 0;
      max-width: 720px;
      min-height: 70px;
      flex: 0 0 auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .assistant-light-card {
      --assistant-lamp-color: #f8f8ff;
      --assistant-lamp-glow: rgba(248,248,255,.06);
      --assistant-lamp-power: 0;
      width: min(640px, 94vw);
      flex: 0 0 auto;
      padding: 13px 16px 12px;
      border: 1px solid var(--card-border);
      border-radius: 18px;
      background: linear-gradient(145deg, rgba(18,29,44,.94), rgba(10,16,25,.94));
      box-shadow: 0 14px 38px rgba(0,0,0,.23);
      text-align: left;
    }
    .assistant-light-heading, .assistant-light-meta, .assistant-light-controls {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .assistant-light-heading strong { font-size: 13px; letter-spacing: .08em; }
    .assistant-light-heading small, .assistant-light-meta { color: var(--text-muted); font-size: 11px; }
    .assistant-light-state { color: var(--text-muted); font: 600 11px 'JetBrains Mono', monospace; }
    .assistant-light-preview {
      --assistant-lamp-color: #f8f8ff;
      --assistant-lamp-glow: rgba(248,248,255,.06);
      --assistant-lamp-power: 0;
      height: 66px;
      margin: 9px 0 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border: 1px solid #263544;
      border-radius: 13px;
      background: radial-gradient(ellipse at 50% 58%, var(--assistant-lamp-glow), transparent 58%), linear-gradient(145deg,#0c141c,#101b25 55%,#0a1118);
      transition: border-color .25s, background .25s;
    }
    .assistant-light-bulb {
      width: 112px;
      height: 39px;
      border: 1px solid color-mix(in srgb, var(--assistant-lamp-color), white 32%);
      border-radius: 999px;
      background: linear-gradient(145deg, color-mix(in srgb,var(--assistant-lamp-color),white 54%), var(--assistant-lamp-color) 52%, color-mix(in srgb,var(--assistant-lamp-color),black 32%));
      opacity: calc(.2 + var(--assistant-lamp-power) * .8);
      box-shadow: 0 0 calc(10px + var(--assistant-lamp-power) * 46px) var(--assistant-lamp-glow), inset 0 2px 10px rgba(255,255,255,.42);
      transition: all .25s;
    }
    .assistant-light-meta { justify-content: flex-start; gap: 20px; margin-bottom: 9px; }
    .assistant-light-controls { justify-content: flex-start; flex-wrap: wrap; }
    .assistant-light-controls button {
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 9px;
      padding: 6px 11px;
      background: rgba(255,255,255,.045);
      color: var(--text);
      font: 600 10px 'Plus Jakarta Sans', system-ui, sans-serif;
      cursor: pointer;
    }
    .assistant-light-controls button:hover, .assistant-light-controls button.active { border-color: var(--assistant-lamp-color); background: rgba(255,255,255,.12); }
    .assistant-light-footnote { margin-left: auto; color: var(--text-muted); font-size: 10px; }
    @media (max-height: 760px) {
      .orb-stage { width: 190px; height: 174px; }
      canvas#orbCanvas { width: 174px; height: 174px; }
      .subtitles-container { min-height: 56px; }
      .assistant-response { font-size: 20px; }
      .assistant-light-card { padding: 9px 12px; }
      .assistant-light-preview { height: 48px; margin: 6px 0; }
      .assistant-light-bulb { height: 31px; }
      .assistant-light-meta { margin-bottom: 6px; }
    }
    @media (max-width: 560px) {
      header { padding: 12px 14px; }
      .brand { font-size: 14px; }
      .header-actions { gap: 6px; }
      .assistant-light-heading small { max-width: 145px; text-align: right; }
      .assistant-light-footnote { width: 100%; margin-left: 0; }
    }
    .event-tag {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 4px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
      margin-bottom: 10px;
      display: inline-block;
      transition: all 0.3s;
    }
    .event-tag.listening {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .assistant-response {
      font-size: 26px;
      font-weight: 600;
      letter-spacing: -0.5px;
      line-height: 1.4;
      color: #ffffff;
      text-shadow: 0 0 24px rgba(255, 255, 255, 0.15);
      transition: all 0.3s ease;
    }
    .countdown-bar {
      width: 240px;
      height: 4px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 2px;
      margin-top: 18px;
      overflow: hidden;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .countdown-bar.active {
      opacity: 1;
    }
    .countdown-fill {
      height: 100%;
      background: linear-gradient(90deg, #10b981, #06d6a0);
      width: 100%;
      transition: width 0.1s linear;
    }
    .player-dock {
      background: var(--card-bg);
      border-top: 1px solid var(--card-border);
      backdrop-filter: blur(24px);
      padding: 16px 36px;
      display: grid;
      grid-template-columns: 1fr 1.5fr 1fr;
      align-items: center;
      gap: 20px;
      z-index: 20;
    }
    .track-info {
      display: flex;
      align-items: center;
      gap: 14px;
      overflow: hidden;
    }
    .album-art {
      width: 52px;
      height: 52px;
      border-radius: 10px;
      background: #1e293b;
      object-fit: cover;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .track-meta {
      overflow: hidden;
    }
    .track-title {
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .track-artist {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 2px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge-source {
      font-size: 9px;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.1);
      color: #cbd5e1;
    }
    .player-controls {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .btn-row {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .ctrl-btn {
      background: none;
      border: none;
      color: var(--text);
      font-size: 18px;
      cursor: pointer;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }
    .ctrl-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      transform: scale(1.08);
    }
    .ctrl-btn.play-btn {
      background: #ffffff;
      color: #000000;
      width: 44px;
      height: 44px;
      font-size: 20px;
    }
    .ctrl-btn.play-btn:hover {
      background: #f1f5f9;
      transform: scale(1.05);
      box-shadow: 0 0 16px rgba(255, 255, 255, 0.3);
    }
    .bars-eq {
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 16px;
    }
    .bar {
      width: 3px;
      height: 4px;
      background: var(--accent-green);
      border-radius: 2px;
      transition: height 0.15s ease;
    }
    .bar.playing {
      animation: bounce 0.6s infinite alternate;
    }
    .bar:nth-child(1) { animation-delay: 0.1s; }
    .bar:nth-child(2) { animation-delay: 0.3s; }
    .bar:nth-child(3) { animation-delay: 0.5s; }
    .bar:nth-child(4) { animation-delay: 0.2s; }
    .bar:nth-child(5) { animation-delay: 0.4s; }
    @keyframes bounce { to { height: 16px; } }
    .player-actions {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 16px;
    }
    .search-input-box {
      display: flex;
      align-items: center;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 4px 12px;
      width: 220px;
      transition: width 0.3s;
    }
    .search-input-box:focus-within {
      width: 260px;
      border-color: var(--accent-blue);
    }
    .search-input-box input {
      background: none;
      border: none;
      color: #fff;
      font-size: 12px;
      width: 100%;
      outline: none;
      font-family: inherit;
    }
    .search-input-box button {
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 14px;
    }
    .search-input-box button:hover { color: #fff; }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(3, 7, 18, 0.85);
      backdrop-filter: blur(20px);
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s;
    }
    .modal-backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }
    .alarm-modal {
      background: #0f172a;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 24px;
      padding: 36px 44px;
      width: 480px;
      text-align: center;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.7);
    }
    .alarm-title {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 20px;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    .flip-clock {
      font-family: 'JetBrains Mono', monospace;
      font-size: 54px;
      font-weight: 700;
      color: #f8fafc;
      letter-spacing: -2px;
      margin: 16px 0 24px;
      text-shadow: 0 0 32px rgba(59, 130, 246, 0.4);
    }
    .alarm-banner-ringing {
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid var(--accent-red);
      color: #fca5a5;
      padding: 12px;
      border-radius: 12px;
      font-weight: 600;
      margin-bottom: 20px;
      animation: pulseAlert 1s infinite alternate;
      display: none;
    }
    @keyframes pulseAlert { to { transform: scale(1.02); box-shadow: 0 0 24px rgba(239, 68, 68, 0.5); } }
    .alarm-list {
      background: rgba(0, 0, 0, 0.25);
      border-radius: 14px;
      padding: 12px 18px;
      margin-bottom: 24px;
      text-align: left;
      font-size: 13px;
    }
    .alarm-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .alarm-item:last-child { border-bottom: none; }
    .btn-group-modal {
      display: flex;
      gap: 12px;
      justify-content: center;
    }
    .btn-primary {
      background: var(--accent-blue);
      border: none;
      color: #fff;
      padding: 10px 24px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      font-size: 14px;
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: #e2e8f0;
      padding: 10px 24px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      font-size: 14px;
    }
    .btn-danger {
      background: var(--accent-red);
      color: #fff;
      border: none;
      padding: 10px 24px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    audio#htmlAudio { display: none; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div id="statusDot" class="status-dot"></div>
      <span>ME2 - VCM on Raspberry Pi 5</span>
    </div>
    <div class="header-actions">
      <details class="audio-settings">
        <summary class="pill-btn">🎧 Audio devices</summary>
        <div class="audio-popover">
          <label>VCM microphone · switches live<select id="vcmMicSelect" class="device-select" aria-label="VCM input microphone">
            <option value="auto">Loading microphones…</option>
          </select></label>
          <label>Browser output · music and page audio<select id="vcmSpeakerSelect" class="device-select" aria-label="Browser audio output">
            <option value="">Loading speakers…</option>
          </select></label>
          <button class="pill-btn" type="button" onclick="refreshAudioDevices()">↻ Refresh audio devices</button>
          <small id="speakerSelectStatus">This routes browser music and page audio. Spoken feedback uses the browser's default output.</small>
        </div>
      </details>
      <div id="clockDisplay" class="clock-pill">12:00:00 PM</div>
      <button id="btnVoiceToggle" class="pill-btn active" onclick="toggleVoiceOutput()">
        <span id="voiceIcon">🔊</span> Voice Feedback
      </button>
      <button class="pill-btn" onclick="openAlarmModal()">🔔 Alarm Clock</button>
      <button id="btnRestartVCM" class="pill-btn" onclick="restartVCM('btnRestartVCM')" title="Restart the local VCM service">↻ Restart VCM</button>
      <a href="/studio" class="pill-btn">🎛️ Studio</a>
    </div>
  </header>

  <main>
    <div class="orb-stage" onclick="simulateWake()" title="Click to Wake ('Hi Dandan')">
      <canvas id="orbCanvas" width="280" height="280"></canvas>
      <div id="orbHint" class="orb-hint">Say "Hi Dandan"</div>
    </div>

    <div class="subtitles-container">
      <div id="eventTag" class="event-tag">STANDBY</div>
      <div id="assistantResponse" class="assistant-response">Standby. Say "Hi Dandan" to wake me up.</div>
      <div id="countdownBar" class="countdown-bar">
        <div id="countdownFill" class="countdown-fill"></div>
      </div>
    </div>

    <section class="assistant-light-card" aria-labelledby="assistantLightTitle">
      <div class="assistant-light-heading">
        <strong id="assistantLightTitle">RGB LIGHT</strong>
        <small>Voice actions update this lamp</small>
        <span id="assistantLightState" class="assistant-light-state">OFF</span>
      </div>
      <div id="assistantRgbPreview" class="assistant-light-preview" role="img" aria-label="RGB light is off">
        <div class="assistant-light-bulb"></div>
      </div>
      <div class="assistant-light-meta">
        <span>Color: <strong id="assistantLightColor">White</strong></span>
        <span>Brightness: <strong id="assistantLightBrightness">0%</strong></span>
      </div>
      <div class="assistant-light-controls" aria-label="Direct LED demonstration controls">
        <button type="button" data-light-command="LIGHT_OFF" onclick="setAssistantLight('LIGHT_OFF')">OFF</button>
        <button type="button" data-light-command="LIGHT_ON" onclick="setAssistantLight('LIGHT_ON')">ON</button>
        <button type="button" data-light-command="COLOR_RED" onclick="setAssistantLight('COLOR_RED')">RED</button>
        <button type="button" data-light-command="COLOR_GREEN" onclick="setAssistantLight('COLOR_GREEN')">GREEN</button>
        <button type="button" data-light-command="COLOR_BLUE" onclick="setAssistantLight('COLOR_BLUE')">BLUE</button>
        <small id="assistantLightHardware" class="assistant-light-footnote">UI simulation · GPIO disabled</small>
      </div>
    </section>
  </main>

  <footer class="player-dock">
    <div class="track-info">
      <img id="albumArt" class="album-art" src="https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=300&h=300&fit=crop" alt="Album Cover">
      <div class="track-meta">
        <div id="trackTitle" class="track-title">Lofi Chill Beats</div>
        <div class="track-artist">
          <span id="trackArtist">Lofi Girl / ChilledCow</span>
          <span id="trackSource" class="badge-source">CURATED</span>
        </div>
      </div>
    </div>

    <div class="player-controls">
      <div class="btn-row">
        <button class="ctrl-btn" onclick="mediaControl('prev')" title="Previous Track">⏮</button>
        <button id="btnPlayPause" class="ctrl-btn play-btn" onclick="mediaControl('toggle')" title="Play / Pause">▶</button>
        <button class="ctrl-btn" onclick="mediaControl('next')" title="Next Track">⏭</button>
      </div>
      <div id="barsEq" class="bars-eq">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
      </div>
    </div>

    <div class="player-actions">
      <div class="search-input-box">
        <input type="text" id="songQuery" placeholder="Play song (YouTube)..." onkeydown="if(event.key==='Enter') searchAndPlay()">
        <button onclick="searchAndPlay()" title="Search & Play">🔍</button>
      </div>
    </div>
  </footer>

  <div id="alarmModal" class="modal-backdrop">
    <div class="alarm-modal">
      <div class="alarm-title">🔔 Alarm Clock</div>
      <div id="modalFlipClock" class="flip-clock">08:00 AM</div>
      <div id="alarmRingingBanner" class="alarm-banner-ringing">🚨 ALARM RINGING! WAKE UP!</div>
      
      <div class="alarm-list">
        <div style="font-weight: 600; margin-bottom: 8px; color: #94a3b8;">SCHEDULED ALARMS</div>
        <div id="alarmListContainer">
          <div class="alarm-item"><span>06:00 AM</span> <span style="color:#10b981">Active (Voice)</span></div>
          <div class="alarm-item"><span>08:00 AM</span> <span style="color:#10b981">Active (Voice)</span></div>
          <div class="alarm-item"><span>09:00 PM</span> <span style="color:#10b981">Active (Voice)</span></div>
        </div>
      </div>

      <div class="btn-group-modal">
        <button id="btnDismissAlarm" class="btn-danger" style="display:none;" onclick="dismissAlarm()">DISMISS ALARM</button>
        <button id="btnSnoozeAlarm" class="btn-secondary" style="display:none;" onclick="snoozeAlarm()">Snooze 5m</button>
        <button class="btn-secondary" onclick="closeAlarmModal()">Close</button>
      </div>
    </div>
  </div>

  <audio id="htmlAudio" crossorigin="anonymous"></audio>

  <script>
    let currentState = 'STANDBY';
    let voiceOutputEnabled = true;
    let lastSpokenMessage = '';
    let currentRms = 0.0;
    let isMediaPlaying = false;
    let currentTrack = null;
    let currentVolume = 50;

    const canvas = document.getElementById('orbCanvas');
    const ctx = canvas.getContext('2d');
    const statusDot = document.getElementById('statusDot');
    const eventTag = document.getElementById('eventTag');
    const assistantResponse = document.getElementById('assistantResponse');
    const countdownBar = document.getElementById('countdownBar');
    const countdownFill = document.getElementById('countdownFill');
    const clockDisplay = document.getElementById('clockDisplay');
    const htmlAudio = document.getElementById('htmlAudio');
    const btnPlayPause = document.getElementById('btnPlayPause');

    async function populateVCMInputs() {
      const select = document.getElementById('vcmMicSelect');
      if (!select) return;
      try {
        const [deviceResponse, stateResponse] = await Promise.all([fetch('/api/audio-inputs'), fetch('/api/state')]);
        const data = await deviceResponse.json();
        const state = await stateResponse.json();
        select.innerHTML = '<option value="auto">Automatic (any available input)</option>';
        if (!(data.inputs || []).length) {
          const missing = new Option('No input detected · connect one, then refresh', 'none');
          missing.disabled = true;
          select.add(missing);
        }
        for (const item of data.inputs || []) {
          const option = document.createElement('option');
          option.value = String(item.index);
          option.textContent = `#${item.index} ${item.name} · ${item.hostapi}`;
          select.appendChild(option);
        }
        const current = state.microphone_requested_device == null ? 'auto' : String(state.microphone_requested_device);
        select.value = [...select.options].some(option => option.value === current) ? current : 'auto';
      } catch (_) {
        select.innerHTML = '<option value="auto">Audio devices unavailable</option>';
      }
    }

    async function populateSpeakerOutputs() {
      const select = document.getElementById('vcmSpeakerSelect');
      if (!select) return;
      const media = navigator.mediaDevices;
      if (!media || !media.enumerateDevices) {
        select.innerHTML = '<option value="">Browser output selection unavailable</option>';
        return;
      }
      try {
        const outputs = (await media.enumerateDevices()).filter(item => item.kind === 'audiooutput');
        const saved = localStorage.getItem('vcm.audioOutput') || '';
        select.innerHTML = '';
        if (!outputs.length) select.add(new Option('Default system output', ''));
        outputs.forEach((item, index) => select.add(new Option(item.label || `Audio output ${index + 1}`, item.deviceId)));
        if ([...select.options].some(option => option.value === saved)) select.value = saved;
        applySpeakerOutput().catch(error => console.warn('Could not select audio output:', error));
      } catch (_) {
        select.innerHTML = '<option value="">Could not list audio outputs</option>';
      }
    }

    async function applySpeakerOutput() {
      const select = document.getElementById('vcmSpeakerSelect');
      if (!select) return;
      const deviceId = select.value;
      localStorage.setItem('vcm.audioOutput', deviceId);
      if (typeof htmlAudio.setSinkId === 'function') await htmlAudio.setSinkId(deviceId);
      const status = document.getElementById('speakerSelectStatus');
      if (status) status.textContent = typeof htmlAudio.setSinkId === 'function'
        ? 'Selected for browser music and page audio; spoken feedback uses the browser default output.'
        : 'This browser cannot route page audio; choose its system output instead.';
    }

    function refreshAudioDevices() {
      populateVCMInputs();
      populateSpeakerOutputs();
    }

    async function selectVCMInput() {
      const select = document.getElementById('vcmMicSelect');
      if (!select) return;
      const response = await fetch('/api/audio-input/select', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({device: select.value})
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.message || 'Microphone selection failed');
      await populateVCMInputs();
    }

    async function restartVCM(buttonId) {
      if (!window.confirm('Restart the local VCM now? Listening pauses briefly and the current assistant session resets.')) return;
      const button = document.getElementById(buttonId);
      const original = button ? button.innerText : '↻ Restart VCM';
      if (button) { button.disabled = true; button.innerText = 'Restarting…'; }
      try {
        const response = await fetch('/api/restart', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device: document.getElementById('vcmMicSelect')?.value || 'unchanged' })
        });
        if (!response.ok) throw new Error(`Restart request failed (${response.status})`);
      } catch (error) {
        if (button) { button.disabled = false; button.innerText = original; }
        window.alert(`Could not restart the VCM: ${error.message}`);
        return;
      }
      const deadline = Date.now() + 45000;
      const check = async () => {
        try {
          const response = await fetch('/api/state', { cache: 'no-store' });
          if (response.ok) { window.location.reload(); return; }
        } catch (_) { /* expected while the old process exits and the new one starts */ }
        if (Date.now() < deadline) window.setTimeout(check, 1000);
        else window.alert('The VCM did not return within 45 seconds. Check the local launcher or service process.');
      };
      window.setTimeout(check, 1200);
    }
    populateVCMInputs();
    populateSpeakerOutputs();
    document.getElementById('vcmMicSelect')?.addEventListener('change', () => selectVCMInput().catch(error => window.alert(error.message)));
    document.getElementById('vcmSpeakerSelect')?.addEventListener('change', () => applySpeakerOutput().catch(error => window.alert(`Could not select this speaker: ${error.message}`)));
    navigator.mediaDevices?.addEventListener?.('devicechange', refreshAudioDevices);

    let orbHue = 220;
    let targetHue = 220;
    let orbPhase = 0;

    function drawOrb() {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      orbHue += (targetHue - orbHue) * 0.08;
      orbPhase += 0.03 + (currentRms * 0.4);

      const cx = w / 2;
      const cy = h / 2;
      const baseRadius = 75 + (currentRms * 120);

      const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, baseRadius * 1.4);
      grad.addColorStop(0, `hsla(${orbHue}, 90%, 65%, 0.85)`);
      grad.addColorStop(0.5, `hsla(${orbHue + 20}, 80%, 45%, 0.45)`);
      grad.addColorStop(1, `hsla(${orbHue}, 100%, 10%, 0)`);

      ctx.save();
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius * 1.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.lineWidth = 2.5;
      for (let layer = 0; layer < 3; layer++) {
        ctx.beginPath();
        const layerHue = (orbHue + layer * 25) % 360;
        ctx.strokeStyle = `hsla(${layerHue}, 95%, 70%, 0.65)`;
        for (let a = 0; a <= Math.PI * 2; a += 0.05) {
          const noise = Math.sin(a * 4 + orbPhase + layer) * (6 + currentRms * 35)
                      + Math.cos(a * 2 - orbPhase) * (4 + currentRms * 20);
          const r = baseRadius + noise - (layer * 12);
          const x = cx + Math.cos(a) * r;
          const y = cy + Math.sin(a) * r;
          if (a === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
      }
      ctx.restore();

      requestAnimationFrame(drawOrb);
    }
    requestAnimationFrame(drawOrb);

    function speakText(text) {
      if (!voiceOutputEnabled || !('speechSynthesis' in window)) return;
      if (!text || text === lastSpokenMessage) return;

      let clean = text.replace(/\[.*?\]/g, '').replace(/->/g, '').trim();
      if (!clean) return;

      lastSpokenMessage = text;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;

      const voices = window.speechSynthesis.getVoices();
      const naturalVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
      if (naturalVoice) utterance.voice = naturalVoice;

      targetHue = 280;
      utterance.onend = () => {
        targetHue = (currentState === 'LISTENING') ? 150 : 220;
      };

      window.speechSynthesis.speak(utterance);
    }

    function toggleVoiceOutput() {
      voiceOutputEnabled = !voiceOutputEnabled;
      const btn = document.getElementById('btnVoiceToggle');
      const icon = document.getElementById('voiceIcon');
      btn.classList.toggle('active', voiceOutputEnabled);
      icon.innerText = voiceOutputEnabled ? '🔊' : '🔇';
    }

    let audioCtx = null;
    let alarmInterval = null;

    populateVCMInputs();
    populateSpeakerOutputs();
    document.getElementById('vcmSpeakerSelect')?.addEventListener('change', () => applySpeakerOutput().catch(error => window.alert(`Could not select this speaker: ${error.message}`)));

    function playAlarmChime() {
      if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === 'suspended') audioCtx.resume();

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, audioCtx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.4);
    }

    async function pollState() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();

        currentState = data.state;
        currentRms = data.rms || 0.0;
        const remaining = data.remaining_sec || 0.0;
        const devices = data.devices || {};
        updateAssistantLight(devices);

        if (currentState === 'LISTENING') {
          targetHue = 150;
          statusDot.className = 'status-dot active';
          eventTag.className = 'event-tag listening';
          eventTag.innerText = `LISTENING (${remaining.toFixed(1)}s)`;
          countdownBar.className = 'countdown-bar active';
          countdownFill.style.width = `${(remaining / (data.timeout_sec || 10.0)) * 100}%`;
          document.getElementById('orbHint').innerText = 'Listening for command...';
        } else {
          targetHue = 220;
          statusDot.className = 'status-dot';
          eventTag.className = 'event-tag';
          eventTag.innerText = 'STANDBY';
          countdownBar.className = 'countdown-bar';
          document.getElementById('orbHint').innerText = 'Say "Hi Dandan"';
        }

        if (data.message && data.message !== assistantResponse.innerText) {
          assistantResponse.innerText = data.message;
          if (data.message.includes('Diliman') || data.message.includes('Lights') || data.message.includes('Alarm') || data.message.includes('Playing') || data.message.includes('Volume')) {
            speakText(data.message);
          }
        }

        if (devices.clock_str) {
          clockDisplay.innerText = devices.clock_str;
          document.getElementById('modalFlipClock').innerText = devices.clock_str.replace(/:[0-9]{2} /, ' ');
        }

        if (devices.current_track) {
          const t = devices.current_track;
          if (!currentTrack || currentTrack.id !== t.id) {
            currentTrack = t;
            document.getElementById('trackTitle').innerText = t.title;
            document.getElementById('trackArtist').innerText = t.artist;
            document.getElementById('trackSource').innerText = (t.source || 'curated').toUpperCase();
            if (t.cover) document.getElementById('albumArt').src = t.cover;

            if (t.url && htmlAudio.src !== new URL(t.url, window.location.href).href) {
              htmlAudio.src = t.url;
              htmlAudio.loop = t.source === 'local';
              if (devices.media_playing) htmlAudio.play().catch(e => console.log('Autoplay:', e));
            }
          }
        }

        isMediaPlaying = devices.media_playing;
        btnPlayPause.innerText = isMediaPlaying ? '⏸' : '▶';
        document.querySelectorAll('.bar').forEach(b => b.classList.toggle('playing', isMediaPlaying));

        currentVolume = devices.volume ?? 50;
        const ducked = devices.ducked;
        const effectiveVol = ducked ? 0.15 : (currentVolume / 100);
        htmlAudio.volume = Math.max(0, Math.min(1, effectiveVol));

        if (isMediaPlaying && htmlAudio.paused) {
          htmlAudio.play().catch(() => {});
        } else if (!isMediaPlaying && !htmlAudio.paused) {
          htmlAudio.pause();
        }
        if (devices.playback_state === 'stopped' && htmlAudio.currentTime > 0) {
          try { htmlAudio.currentTime = 0; } catch (_) { /* live streams may not be seekable */ }
        }

        if (devices.alarm_ringing) {
          openAlarmModal();
          document.getElementById('alarmRingingBanner').style.display = 'block';
          document.getElementById('btnDismissAlarm').style.display = 'inline-block';
          document.getElementById('btnSnoozeAlarm').style.display = 'inline-block';
          if (!alarmInterval) {
            alarmInterval = setInterval(playAlarmChime, 800);
          }
        } else {
          document.getElementById('alarmRingingBanner').style.display = 'none';
          document.getElementById('btnDismissAlarm').style.display = 'none';
          document.getElementById('btnSnoozeAlarm').style.display = 'none';
          if (alarmInterval) {
            clearInterval(alarmInterval);
            alarmInterval = null;
          }
        }

      } catch (err) {
        console.error('Polling error:', err);
      }
    }

    setInterval(pollState, 350);

    function updateAssistantLight(devices) {
      const stage = document.getElementById('assistantRgbPreview');
      if (!stage || !devices) return;
      const palette = {
        white: ['#f8f8ff', 'rgba(248,248,255,.52)'],
        red: ['#ff334f', 'rgba(255,51,79,.6)'],
        green: ['#36f28b', 'rgba(54,242,139,.58)'],
        blue: ['#4388ff', 'rgba(67,136,255,.62)'],
      };
      const colorName = String(devices.lights_color || 'white').toLowerCase();
      const [color, glow] = palette[colorName] || palette.white;
      const intensity = Math.max(0, Math.min(100, Number(devices.lights_percent) || 0));
      const power = intensity / 100;
      document.querySelector('.assistant-light-card').style.setProperty('--assistant-lamp-color', color);
      stage.style.setProperty('--assistant-lamp-color', color);
      stage.style.setProperty('--assistant-lamp-glow', glow);
      stage.style.setProperty('--assistant-lamp-power', String(power));
      stage.style.borderColor = power ? color : '#263544';
      stage.setAttribute('aria-label', `${power ? 'On' : 'Off'} ${colorName} light at ${intensity}% brightness`);
      const state = document.getElementById('assistantLightState');
      state.innerText = power ? 'ON' : 'OFF';
      state.style.color = power ? color : '#94a3b8';
      document.getElementById('assistantLightColor').innerText = colorName[0].toUpperCase() + colorName.slice(1);
      document.getElementById('assistantLightBrightness').innerText = `${intensity}%`;
      document.getElementById('assistantLightHardware').innerText = devices.rgb_hardware_enabled
        ? 'Pi GPIO · BCM 17 / 27 / 22'
        : 'UI simulation · GPIO disabled';
      const activeCommand = !power ? 'LIGHT_OFF' : colorName === 'red' || colorName === 'green' || colorName === 'blue'
        ? `COLOR_${colorName.toUpperCase()}` : 'LIGHT_ON';
      document.querySelectorAll('[data-light-command]').forEach(button => {
        button.classList.toggle('active', button.dataset.lightCommand === activeCommand);
      });
    }

    async function setAssistantLight(command) {
      try {
        const response = await fetch('/api/light/demo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command }),
        });
        const data = await response.json();
        if (!response.ok || data.status !== 'ok') throw new Error(data.message || 'Light control failed');
        updateAssistantLight(data.devices);
        assistantResponse.innerText = data.message;
      } catch (error) {
        assistantResponse.innerText = `Light demo unavailable: ${error.message}`;
      }
    }

    async function simulateWake() {
      await fetch('/api/wake', { method: 'POST' });
      speakText("I'm listening.");
    }

    async function mediaControl(action) {
      if (action === 'toggle') {
        const endpoint = isMediaPlaying ? '/api/music/pause' : '/api/music/play';
        await fetch(endpoint, { method: 'POST' });
      } else if (action === 'next') {
        await fetch('/api/music/next', { method: 'POST' });
      } else if (action === 'prev') {
        await fetch('/api/music/prev', { method: 'POST' });
      }
    }

    async function searchAndPlay() {
      const q = document.getElementById('songQuery').value.trim();
      if (!q) return;
      await fetch('/api/music/play?q=' + encodeURIComponent(q), { method: 'POST' });
      document.getElementById('songQuery').value = '';
    }

    function openAlarmModal() {
      document.getElementById('alarmModal').classList.add('show');
    }

    function closeAlarmModal() {
      document.getElementById('alarmModal').classList.remove('show');
    }

    async function dismissAlarm() {
      await fetch('/api/alarm/dismiss', { method: 'POST' });
      closeAlarmModal();
    }

    async function snoozeAlarm() {
      await fetch('/api/alarm/snooze', { method: 'POST' });
      closeAlarmModal();
    }
  </script>
</body>
</html>
"""

STUDIO_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ME2 - VCM on Raspberry Pi 5</title>
  <style>
    *{box-sizing:border-box}
    body{margin:0;background:#0b1117;color:#e8f1f5;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
    main{max-width:1050px;margin:auto;padding:36px 24px}
    header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e2c38;padding-bottom:18px}
    h1{font-size:26px;margin:0;color:#f0f8ff}
    .badge{border-radius:20px;padding:6px 14px;font-size:12px;font-weight:600;letter-spacing:1px}
    .badge-standby{background:#1e293b;border:1px solid #475569;color:#94a3b8}
    .badge-listening{background:#064e3b;border:1px solid #10b981;color:#34d399;box-shadow:0 0 16px #10b98144}
    .hero{background:#111b24;border:1px solid #1e2c38;border-radius:20px;text-align:center;padding:32px 20px;margin:24px 0}
    .orb{margin:auto;width:96px;height:96px;border-radius:50%;transition:all .3s}
    .orb-standby{background:radial-gradient(circle at 35% 30%,#64748b,#334155 60%,#0f172a);box-shadow:0 0 24px #33415533}
    .orb-listening{background:radial-gradient(circle at 35% 30%,#34d399,#059669 60%,#064e3b);box-shadow:0 0 48px #34d39988;animation:pulse 1.2s infinite alternate}
    @keyframes pulse{to{transform:scale(1.06);box-shadow:0 0 64px #34d399aa}}
    h2{font-size:24px;margin:16px 0 6px;letter-spacing:1px}
    .level-bar{height:8px;background:#1e2c38;max-width:380px;margin:16px auto 6px;border-radius:6px;overflow:hidden}
    .level-fill{height:100%;background:#10b981;width:0%;transition:width .1s}
    .timer-container{max-width:380px;margin:12px auto;background:#1e2c38;height:6px;border-radius:4px;overflow:hidden}
    .timer-fill{height:100%;background:#f59e0b;width:0%;transition:width .2s}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}
    .card{background:#111b24;border:1px solid #1e2c38;padding:18px;border-radius:14px}
    .card small{color:#8da5b5;font-size:12px;text-transform:uppercase;letter-spacing:1px}
    .card .val{font-size:22px;font-weight:600;margin-top:8px;color:#e8f1f5}
    .row{display:grid;grid-template-columns:1.2fr 0.8fr;gap:20px;margin:24px 0}
    .terminal{background:#070b0e;border:1px solid #16222c;padding:16px;border-radius:12px;font-family:monospace;font-size:13px;color:#a3e635;min-height:170px;max-height:230px;overflow-y:auto;white-space:pre-wrap}
    .vocab{background:#111b24;border:1px solid #1e2c38;padding:18px;border-radius:14px;font-size:13px;line-height:1.7;color:#94a3b8}
    .vocab b{color:#38bdf8}
    .lightbox-panel{background:#111b24;border:1px solid #1e2c38;border-radius:18px;padding:20px;margin:24px 0}
    .lightbox-header,.lightbox-footer{display:flex;justify-content:space-between;align-items:center;gap:16px}
    .lightbox-header h3{font-size:15px;margin:0 0 4px;text-transform:uppercase;letter-spacing:.08em;color:#e8f1f5}
    .lightbox-header small,.lightbox-footer{color:#8da5b5;font-size:12px}
    .lightbox-state{border:1px solid #334155;border-radius:999px;padding:7px 12px;font-size:11px;letter-spacing:.08em;color:#94a3b8}
    .rgb-stage{--light-color:#f8f8ff;--light-glow:rgba(248,248,255,.08);--light-intensity:0;position:relative;display:grid;place-items:center;isolation:isolate;overflow:hidden;width:100%;height:clamp(250px,42vh,390px);margin:18px 0 14px;border:1px solid #263544;border-radius:16px;background:radial-gradient(ellipse at 50% 66%,var(--light-glow),transparent 55%),linear-gradient(145deg,#0c141c,#101b25 55%,#0a1118);transition:background .3s,border-color .3s}
    .rgb-stage::before{content:"";position:absolute;inset:auto 9% 10%;height:18%;border-radius:50%;background:var(--light-glow);filter:blur(28px);opacity:calc(var(--light-intensity)*.95);transition:all .35s}
    .rgb-source{position:relative;z-index:1;width:min(34vw,300px);height:min(27vw,210px);min-width:190px;min-height:145px;border:1px solid color-mix(in srgb,var(--light-color),white 35%);border-radius:34px;background:linear-gradient(145deg,color-mix(in srgb,var(--light-color),white 62%),var(--light-color) 45%,color-mix(in srgb,var(--light-color),black 34%));box-shadow:0 0 24px var(--light-glow),0 0 calc(40px + var(--light-intensity)*100px) var(--light-glow),inset 0 2px 20px rgba(255,255,255,.42);opacity:calc(.22 + var(--light-intensity)*.78);transform:perspective(700px) rotateX(10deg);transition:all .35s}
    .rgb-source::after{content:"RGB";position:absolute;inset:0;display:grid;place-items:center;color:rgba(10,16,24,.72);font-size:clamp(28px,5vw,52px);font-weight:800;letter-spacing:.28em;text-shadow:0 1px rgba(255,255,255,.35)}
    .lightbox-footer strong{font-size:14px;color:#f8fafc}
    @media(max-width:700px){.lightbox-panel{padding:14px}.rgb-stage{height:270px}.rgb-source{width:220px;height:165px}.lightbox-footer{align-items:flex-start;flex-direction:column}}
    .btn-group{display:flex;gap:10px}
    .device-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#111b24;border:1px solid #1e2c38;border-radius:12px;padding:12px;margin:18px 0}
    .device-controls label{display:flex;align-items:center;gap:8px;color:#8da5b5;font-size:12px}
    .device-controls select{max-width:310px;background:#0b1117;color:#e8f1f5;border:1px solid #334155;border-radius:7px;padding:8px}
    button, a.btn{background:#1a2834;color:#e2e8f0;border:1px solid #3b82f6;border-radius:8px;padding:8px 14px;cursor:pointer;font-weight:600;font-size:12px;text-decoration:none;display:inline-flex;align-items:center}
    button:hover, a.btn:hover{background:#2563eb}
    .btn-wake{background:#064e3b;border-color:#10b981;color:#a7f3d0}
    .btn-wake:hover{background:#059669}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>ME2 - VCM on Raspberry Pi 5</h1>
      <small style="color:#8da5b5">Wake and intent studio</small>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <button class="btn" onclick="restartVCM('btnRestartStudio')" id="btnRestartStudio" title="Restart the local VCM service">↻ Restart VCM</button>
      <a href="/" class="btn">✨ Modern UI</a>
      <div id="stateBadge" class="badge badge-standby">STANDBY</div>
    </div>
  </header>

  <section class="device-controls" aria-label="Audio device settings">
    <label>VCM microphone · switches live <select id="vcmMicSelect"><option value="auto">Loading inputs…</option></select></label>
    <label>Browser output · music/page audio <select id="vcmSpeakerSelect"><option value="">Loading outputs…</option></select></label>
    <button type="button" onclick="refreshAudioDevices()">↻ Refresh audio devices</button>
    <small id="speakerSelectStatus" style="color:#8da5b5">The output selector routes browser music/page audio. Spoken feedback uses the browser's default output.</small>
  </section>

  <section class="hero">
    <div id="orb" class="orb orb-standby"></div>
    <h2 id="msg">Standby. Say 'Hi Dandan' to wake me up.</h2>
    <div class="level-bar"><div id="levelFill" class="level-fill"></div></div>
    <div class="timer-container"><div id="timerFill" class="timer-fill"></div></div>
    <small id="sub" style="color:#8da5b5">Microphone active · Waiting for 'Hi Dandan'</small>
    <small id="micSource" style="color:#8da5b5;display:block;margin-top:6px">Checking audio input...</small>
  </section>

  <div class="cards">
    <div class="card"><small id="predictionLabel">Prediction</small><div id="cmd" class="val">-</div></div>
    <div class="card"><small id="confidenceLabel">Confidence</small><div id="conf" class="val">-</div></div>
    <div class="card"><small>Latency</small><div id="lat" class="val">1.2 ms</div></div>
    <div class="card"><small>Model Footprint</small><div id="size" class="val">--</div></div>
  </div>

  <section class="lightbox-panel" aria-label="RGB lights preview">
    <div class="lightbox-header">
      <div><h3>Dedicated RGB lights</h3><small>Live preview of the light device state</small></div>
      <span class="lightbox-state" id="lightState">OFF</span>
    </div>
    <div class="rgb-stage" id="rgbStage" role="img" aria-label="Simulated RGB light preview">
      <div class="rgb-source"></div>
    </div>
    <div class="lightbox-footer">
      <span>Color: <strong id="lightColor">White</strong></span>
      <span>Brightness: <strong id="lightBrightness">0%</strong></span>
      <span id="lightHardwareNote">Software simulation · GPIO is off</span>
    </div>
  </section>

  <audio id="htmlAudio" style="display:none"></audio>

  <div class="row">
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h3 style="font-size:14px;margin:0;text-transform:uppercase;color:#8da5b5">Live Event Stream</h3>
        <div class="btn-group">
          <button class="btn-wake" onclick="fetch('/api/wake',{method:'POST'})">🎤 Simulate Wake</button>
          <button onclick="fetch('/api/test_weather',{method:'POST'})">🌤️ Weather</button>
        </div>
      </div>
      <div id="terminal" class="terminal">Starting listener...</div>
    </div>
    <div>
      <h3 style="font-size:14px;margin:0 0 8px;text-transform:uppercase;color:#8da5b5">Active Devices & Actions</h3>
      <div class="vocab">
        <p>• <b>Lights:</b> <span id="devLights">0%</span></p>
        <p>• <b>Music:</b> <span id="devMedia">Paused</span></p>
        <label for="volumeMeter">Volume <strong id="devVol">50%</strong></label>
        <progress id="volumeMeter" max="100" value="50" style="width:100%;height:20px;accent-color:#38bdf8" aria-label="Simulated volume level"></progress>
        <p>• <b>Thermostat target:</b> <span id="devTemperature">22 °C (demo)</span></p>
        <p>• <b>Clock & Alarms:</b> <span id="devAlarm">None</span></p>
        <p>• <b>Weather:</b> <span id="weatherSource">Checking mode...</span></p>
      </div>
    </div>
  </div>
</main>
<script>
  async function populateVCMInputs() {
    const select = document.getElementById('vcmMicSelect');
    if (!select) return;
    try {
      const [deviceResponse, stateResponse] = await Promise.all([fetch('/api/audio-inputs'), fetch('/api/state')]);
      const data = await deviceResponse.json();
      const state = await stateResponse.json();
      select.innerHTML = '<option value="auto">Automatic (any available input)</option>';
      if (!(data.inputs || []).length) {
        const missing = new Option('No Pi input detected · connect one, then refresh', 'none');
        missing.disabled = true;
        select.add(missing);
      }
      for (const item of data.inputs || []) {
        select.add(new Option(`#${item.index} ${item.name} · ${item.hostapi}`, String(item.index)));
      }
      const current = state.microphone_requested_device == null ? 'auto' : String(state.microphone_requested_device);
      select.value = [...select.options].some(option => option.value === current) ? current : 'auto';
    } catch (_) { select.innerHTML = '<option value="auto">Audio devices unavailable</option>'; }
  }

  async function populateSpeakerOutputs() {
    const select = document.getElementById('vcmSpeakerSelect');
    if (!select || !navigator.mediaDevices?.enumerateDevices) return;
    try {
      const outputs = (await navigator.mediaDevices.enumerateDevices()).filter(item => item.kind === 'audiooutput');
      const saved = localStorage.getItem('vcm.audioOutput') || '';
      select.innerHTML = '';
      if (!outputs.length) select.add(new Option('Default system output', ''));
      outputs.forEach((item, index) => select.add(new Option(item.label || `Audio output ${index + 1}`, item.deviceId)));
      if ([...select.options].some(option => option.value === saved)) select.value = saved;
      applySpeakerOutput().catch(() => {});
    } catch (_) { select.innerHTML = '<option value="">Could not list audio outputs</option>'; }
  }

  async function applySpeakerOutput() {
    const select = document.getElementById('vcmSpeakerSelect');
    const audio = document.getElementById('htmlAudio');
    if (!select || !audio) return;
    const deviceId = select.value;
    localStorage.setItem('vcm.audioOutput', deviceId);
    if (typeof audio.setSinkId === 'function') await audio.setSinkId(deviceId);
    const status = document.getElementById('speakerSelectStatus');
    if (status) status.textContent = typeof audio.setSinkId === 'function'
      ? 'Selected for browser music and page audio; spoken feedback uses the browser default output.'
      : 'This browser cannot route page audio; choose its system output instead.';
  }

  function refreshAudioDevices() {
    populateVCMInputs();
    populateSpeakerOutputs();
  }

  async function selectVCMInput() {
    const select = document.getElementById('vcmMicSelect');
    if (!select) return;
    const response = await fetch('/api/audio-input/select', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device: select.value})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || 'Microphone selection failed');
    await populateVCMInputs();
  }

  async function restartVCM(buttonId) {
    if (!window.confirm('Restart the local VCM now? Listening pauses briefly and the current assistant session resets.')) return;
    const button = document.getElementById(buttonId);
    if (button) { button.disabled = true; button.innerText = 'Restarting…'; }
    try {
      const response = await fetch('/api/restart', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: document.getElementById('vcmMicSelect')?.value || 'unchanged' })
      });
      if (!response.ok) throw new Error(`Restart request failed (${response.status})`);
    } catch (error) {
      if (button) { button.disabled = false; button.innerText = '↻ Restart VCM'; }
      window.alert(`Could not restart the VCM: ${error.message}`);
      return;
    }
    const deadline = Date.now() + 45000;
    const check = async () => {
      try { const response = await fetch('/api/state', { cache: 'no-store' }); if (response.ok) { window.location.reload(); return; } } catch (_) {}
      if (Date.now() < deadline) window.setTimeout(check, 1000);
      else window.alert('The VCM did not return within 45 seconds. Check the local launcher or service process.');
    };
    window.setTimeout(check, 1200);
  }
  populateVCMInputs();
  populateSpeakerOutputs();
  document.getElementById('vcmMicSelect')?.addEventListener('change', () => selectVCMInput().catch(error => window.alert(error.message)));
  document.getElementById('vcmSpeakerSelect')?.addEventListener('change', () => applySpeakerOutput().catch(error => window.alert(`Could not select this speaker: ${error.message}`)));
  navigator.mediaDevices?.addEventListener?.('devicechange', refreshAudioDevices);

  async function update() {
    try {
      const res = await fetch('/api/state');
      const d = await res.json();
      const st = d.state;
      const modelSizeKb = Number(d.model_kb);
      document.getElementById('size').innerText =
        Number.isFinite(modelSizeKb) && modelSizeKb > 0 ? modelSizeKb.toFixed(1) + ' KB' : 'Unavailable';
      const b = document.getElementById('stateBadge');
      const orb = document.getElementById('orb');
      b.className = 'badge ' + (st === 'LISTENING' ? 'badge-listening' : 'badge-standby');
      b.innerText = st;
      orb.className = 'orb ' + (st === 'LISTENING' ? 'orb-listening' : 'orb-standby');

      document.getElementById('msg').innerText = d.message || '';
      document.getElementById('levelFill').style.width = Math.min(100, Math.round((d.rms || 0) * 1500)) + '%';
      const captureOk = d.microphone_stream_active && d.audio_age_sec !== null && d.audio_age_sec < 1;
      document.getElementById('micSource').innerText = `${captureOk ? 'Receiving audio' : 'No recent audio'} · ${d.microphone || 'No input selected'} · ${d.audio_chunks_processed || 0} chunks · ${d.microphone_status || 'no stream warning'}`;
      document.getElementById('weatherSource').innerText = d.live_weather ? 'Live Open-Meteo lookup (Diliman, QC)' : 'Offline mode';

      if (st === 'LISTENING') {
        const pct = ((d.remaining_sec || 0) / (d.timeout_sec || 10.0)) * 100;
        document.getElementById('timerFill').style.width = pct + '%';
        document.getElementById('sub').innerText = `LISTENING · ${d.remaining_sec}s remaining before sleep · mic RMS ${(d.rms || 0).toFixed(4)} · ${(d.inference_interval_sec || 0).toFixed(2)}s interval · ${d.inference_count || 0} inferences`;
      } else {
        document.getElementById('timerFill').style.width = '0%';
        if (d.binary_wake_enabled) {
          const score = d.latest?.mode === 'wake' ? d.latest.wake_probability : null;
          const scoreText = score === null || score === undefined ? 'no wake inference yet' : `wake ${(score * 100).toFixed(1)}% / threshold ${((d.wake_threshold || 0) * 100).toFixed(1)}% · confirmation ${d.wake_streak || 0}/${d.wake_confirmations || 1}`;
          document.getElementById('sub').innerText = `Mic RMS ${(d.rms || 0).toFixed(4)} / VAD gate ${(d.vad_threshold || 0).toFixed(4)} · ${(d.inference_interval_sec || 0).toFixed(2)}s interval · ${scoreText} · ${d.inference_count || 0} inferences`;
        } else {
          document.getElementById('sub').innerText = "Microphone active · Say 'Hi Dandan' to wake me up";
        }
      }

      if (d.latest && d.latest.label) {
        const inWakeMode = d.binary_wake_enabled && d.state === 'STANDBY' && d.latest.mode === 'wake';
        document.getElementById('predictionLabel').innerText = inWakeMode ? 'Wake decision' : 'Command';
        document.getElementById('confidenceLabel').innerText = inWakeMode ? 'Wake probability' : 'Confidence';
        document.getElementById('cmd').innerText = d.latest.label;
        const displayedConfidence = inWakeMode ? d.latest.wake_probability : d.latest.confidence;
        document.getElementById('conf').innerText = Math.round(displayedConfidence * 100) + '%';
        document.getElementById('lat').innerText = d.latest.latency_ms.toFixed(1) + ' ms';
      }

      if (d.events) {
        document.getElementById('terminal').innerText = d.events.slice(0, 10).join('\n');
      }

      if (d.devices) {
        document.getElementById('devLights').innerText = d.devices.lights_percent + '%';
        const lightColors = {
          white: ['#f8f8ff', 'rgba(248,248,255,.46)'],
          red: ['#ff334f', 'rgba(255,51,79,.48)'],
          green: ['#36f28b', 'rgba(54,242,139,.45)'],
          blue: ['#4388ff', 'rgba(67,136,255,.5)'],
        };
        const colorName = String(d.devices.lights_color || 'white').toLowerCase();
        const [color, glow] = lightColors[colorName] || lightColors.white;
        const intensity = Math.max(0, Math.min(100, Number(d.devices.lights_percent) || 0));
        const stage = document.getElementById('rgbStage');
        stage.style.setProperty('--light-color', color);
        stage.style.setProperty('--light-glow', glow);
        stage.style.setProperty('--light-intensity', String(intensity / 100));
        stage.style.borderColor = intensity ? color : '#263544';
        document.getElementById('lightState').innerText = intensity ? 'ON' : 'OFF';
        document.getElementById('lightState').style.color = intensity ? color : '#94a3b8';
        document.getElementById('lightColor').innerText = colorName[0].toUpperCase() + colorName.slice(1);
        document.getElementById('lightBrightness').innerText = intensity + '%';
        document.getElementById('lightHardwareNote').innerText = d.devices.rgb_hardware_enabled ? 'GPIO RGB hardware enabled' : 'Software simulation · GPIO is off';
        document.getElementById('devMedia').innerText = d.devices.media_playing
          ? 'Playing: ' + (d.devices.current_track?.title || '')
          : (d.devices.playback_state === 'paused' ? 'Paused' : 'Stopped');
        document.getElementById('devVol').innerText = d.devices.volume + '%';
        document.getElementById('volumeMeter').value = d.devices.volume;
        document.getElementById('devTemperature').innerText = d.devices.thermostat_demo_c + ' °C (demo)';
        document.getElementById('devAlarm').innerText = d.devices.clock_str || 'None';
      }
    } catch (e) {}
  }
  setInterval(update, 350);
  update();
</script>
</body>
</html>
"""

def run_server(port=7860, host='127.0.0.1', gpio=False, device=None, model_path=None, timeout_sec=10.0, binary_wake_model_path=None, wake_threshold=None, vad_threshold=0.012, inference_interval_sec=0.35, live_weather=False, wake_confirmations=1):
    assistant = AntigravAssistant(
        gpio=gpio,
        device=device,
        live_weather=live_weather,
        wake_confirmations=wake_confirmations,
        timeout_sec=timeout_sec,
        model_path=model_path,
        binary_wake_model_path=binary_wake_model_path,
        wake_threshold=wake_threshold,
        vad_threshold=vad_threshold,
        inference_interval_sec=inference_interval_sec,
    )
    assistant.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == '/api/state':
                payload = json.dumps(assistant.snapshot()).encode('utf-8')
                mime = 'application/json'
            elif path == '/api/audio-inputs':
                try:
                    import sounddevice as sd
                    hosts = sd.query_hostapis()
                    inputs = [
                        {'index': index, 'name': info['name'], 'hostapi': hosts[info['hostapi']]['name']}
                        for index, info in enumerate(sd.query_devices()) if info['max_input_channels']
                    ]
                    payload = json.dumps({'inputs': inputs}).encode('utf-8')
                except Exception as exc:
                    payload = json.dumps({'inputs': [], 'error': str(exc)}).encode('utf-8')
                mime = 'application/json'
            elif path in ('/assets/demo_melody_1.wav', '/assets/demo_melody_2.wav'):
                payload = (ROOT / path.lstrip('/')).read_bytes()
                mime = 'audio/wav'
            elif path in ('/', '/assistant'):
                payload = ASSISTANT_HTML.encode('utf-8')
                mime = 'text/html; charset=utf-8'
            elif path in ('/studio', '/dashboard'):
                payload = STUDIO_HTML.encode('utf-8')
                mime = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            if path == '/api/audio-input/select':
                if self.client_address[0] not in ('127.0.0.1', '::1'):
                    self.send_error(403, 'Audio device changes are local-only')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0') or 0)
                    if length < 1 or length > 4096:
                        raise ValueError('Invalid audio selection request size')
                    request_data = json.loads(self.rfile.read(length).decode('utf-8'))
                    if not isinstance(request_data, dict):
                        raise ValueError('Audio selection must be an object')
                    result = assistant.select_microphone(request_data.get('device'))
                    payload = json.dumps({'status': 'ok', **result}).encode('utf-8')
                    status = 200
                except Exception as exc:
                    payload = json.dumps({'status': 'error', 'message': str(exc)}).encode('utf-8')
                    status = 400
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == '/api/light/demo':
                if self.client_address[0] not in ('127.0.0.1', '::1'):
                    self.send_error(403, 'Direct light demo controls are local-only')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0') or 0)
                    if length < 1 or length > 4096:
                        raise ValueError('Invalid light demo request size')
                    request_data = json.loads(self.rfile.read(length).decode('utf-8'))
                    if not isinstance(request_data, dict):
                        raise ValueError('Light demo request must be an object')
                    message = apply_light_demo_command(assistant.devices, request_data.get('command'))
                    assistant.devices.message = message
                    assistant.events.appendleft(f"{time.strftime('%H:%M:%S')} [LIGHT DEMO] -> {message}")
                    payload = json.dumps({'status': 'ok', 'message': message, 'devices': assistant.devices.snapshot()}).encode('utf-8')
                    status = 200
                except Exception as exc:
                    payload = json.dumps({'status': 'error', 'message': str(exc)}).encode('utf-8')
                    status = 400
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
                return

            if path == '/api/restart':
                if self.client_address[0] not in ('127.0.0.1', '::1'):
                    self.send_error(403, 'Restart is available only from this computer')
                    return
                try:
                    content_length = int(self.headers.get('Content-Length', '0') or 0)
                    request_data = json.loads(self.rfile.read(content_length).decode('utf-8')) if content_length else {}
                    if not isinstance(request_data, dict):
                        raise ValueError('Restart settings must be a JSON object')
                    arguments = list(sys.argv[1:])
                    if 'device' in request_data and request_data['device'] != 'unchanged':
                        arguments = arguments_with_selected_device(arguments, request_data['device'])
                    queue_restart('vcm', port=port, arguments=arguments)
                except Exception as exc:
                    payload = json.dumps({'status': 'error', 'message': str(exc)}).encode('utf-8')
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                payload = json.dumps({'status': 'restarting'}).encode('utf-8')
                self.send_response(202)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
                threading.Timer(0.8, assistant.stop_event.set).start()
                return

            if path == '/api/wake':
                assistant.trigger_wake(source="Web UI / API")
            elif path == '/api/music/play':
                q = params.get('q', [''])[0]
                if q:
                    track = assistant.devices.play_query(q)
                    line = f"{time.strftime('%H:%M:%S')} [MUSIC SEARCH] -> Playing '{track['title']}' ({track['artist']})"
                    assistant.events.appendleft(line)
                else:
                    msg = assistant.devices.execute('PLAY_MUSIC')
                    line = f"{time.strftime('%H:%M:%S')} [PLAY_MUSIC] -> {msg}"
                    assistant.events.appendleft(line)
            elif path == '/api/music/pause':
                msg = assistant.devices.execute('PAUSE')
                line = f"{time.strftime('%H:%M:%S')} [PAUSE] -> {msg}"
                assistant.events.appendleft(line)
            elif path == '/api/music/next':
                msg = assistant.devices.execute('NEXT')
                line = f"{time.strftime('%H:%M:%S')} [NEXT] -> {msg}"
                assistant.events.appendleft(line)
            elif path == '/api/music/prev':
                track = assistant.devices.media_engine.previous_track()
                assistant.devices.media = True
                assistant.devices.playback_state = 'playing'
                line = f"{time.strftime('%H:%M:%S')} [PREV] -> Playing '{track['title']}'"
                assistant.events.appendleft(line)
            elif path == '/api/alarm/dismiss':
                msg = assistant.devices.dismiss_alarm()
                line = f"{time.strftime('%H:%M:%S')} [ALARM] -> {msg}"
                assistant.events.appendleft(line)
            elif path == '/api/alarm/snooze':
                msg = assistant.devices.snooze_alarm()
                line = f"{time.strftime('%H:%M:%S')} [ALARM] -> {msg}"
                assistant.events.appendleft(line)
            elif path == '/api/test_weather':
                if assistant.state != 'LISTENING':
                    assistant.trigger_wake(source="Test Weather (Auto-Wake)")
                msg = assistant.devices.execute('WEATHER')
                line = f"{time.strftime('%H:%M:%S')} [TEST WEATHER] -> {msg}"
                assistant.events.appendleft(line)
            else:
                self.send_error(404)
                return

            payload = json.dumps(assistant.snapshot()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = None
    try:
        server = ReusableServer((host, port), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"\n[Dandan AI Assistant Active] Serving on http://{host}:{port}", flush=True)
        print(f"Assistant UI (local): http://127.0.0.1:{port}/assistant", flush=True)
        print(f"Studio dashboard (local): http://127.0.0.1:{port}/studio", flush=True)

        def shutdown_handler(*args):
            print("\nReceived stop signal...", flush=True)
            assistant.stop_event.set()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        while not assistant.stop_event.wait(0.5):
            pass
    finally:
        if server:
            server.shutdown()
            server.server_close()
        assistant.close()
        print("[Antigrav] Assistant service stopped cleanly.", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--host', default='127.0.0.1', help='Bind address; use 0.0.0.0 only on a trusted LAN.')
    parser.add_argument('--model', type=Path, help='Optional explicit ONNX model path (31-class scratch or 32-class wake extension).')
    parser.add_argument('--binary-wake-model', type=Path, help='Optional separate two-class wake ONNX; uses --model as the 31-class command model.')
    parser.add_argument('--wake-threshold', type=float, help='Override the validation-selected binary wake threshold; defaults to export_summary.json.')
    parser.add_argument('--vad-threshold', type=float, default=0.012, help='RMS voice-activity gate (default 0.012); lower values hear quieter speech but run more inferences.')
    parser.add_argument('--inference-interval', type=float, default=0.35, help='Minimum seconds between model evaluations (default 0.35s).')
    parser.add_argument('--timeout', type=float, default=10.0, help='Inactivity timeout in seconds (default 10.0s).')
    parser.add_argument('--gpio', action='store_true')
    parser.add_argument('--device', help='Microphone input index or case-insensitive name fragment. Defaults to automatic selection.')
    parser.add_argument('--list-input-devices', action='store_true', help='List available PortAudio input device indices and exit.')
    parser.add_argument('--live-weather', action='store_true', help='Allow weather actions to query Open-Meteo over the network.')
    parser.add_argument('--wake-confirmations', type=int, default=1, help='Consecutive positive wake windows required (default 1).')
    args = parser.parse_args()
    if args.list_input_devices:
        import sounddevice as sd
        host_apis = sd.query_hostapis()
        default_input = sd.default.device[0]
        for index, info in enumerate(sd.query_devices()):
            if info['max_input_channels']:
                marker = ' (Windows default)' if index == default_input else ''
                print(f"[{index}] {info['name']} / {host_apis[info['hostapi']]['name']} / {int(info['default_samplerate'])} Hz / {info['max_input_channels']} input ch{marker}")
        sys.exit(0)
    run_server(
        port=args.port,
        host=args.host,
        gpio=args.gpio,
        device=args.device,
        live_weather=args.live_weather,
        wake_confirmations=args.wake_confirmations,
        model_path=args.model,
        timeout_sec=args.timeout,
        binary_wake_model_path=args.binary_wake_model,
        wake_threshold=args.wake_threshold,
        vad_threshold=args.vad_threshold,
        inference_interval_sec=args.inference_interval,
    )
