"""Always-listening edge voice assistant with Wake Word gating, 5s inactivity timeout,
Modern AI Assistant UI with Canvas Voice Orb, Spoken Voice Responses (TTS),
Actual Music Streaming (YouTube & Curated), and Interactive Alarm Clock on Raspberry Pi 5.
"""
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

from tinyvcm_antigrav.config import LABELS, SAMPLES, SR, MELS, TIME_STEPS
from tinyvcm_antigrav.frontend import Frontend, fit_audio
from tinyvcm_antigrav.media_engine import MediaEngine
from tinyvcm.devices import Devices
from tinyvcm.microphone import Microphone

class AntigravPredictor:
    """Zero-torch, standalone INT8 ONNX predictor with 32 classes (31 commands + WAKE_WORD)."""
    def __init__(self, model_path=None):
        if not model_path:
            model_path = ROOT / 'models' / 'antigrav_optionb_int8.onnx'
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.frontend = Frontend()
        self.labels = LABELS
        self.model_size_kb = round(self.model_path.stat().st_size / 1024, 1)

    def predict(self, audio):
        t0 = time.perf_counter()
        fitted = fit_audio(audio, target_samples=SAMPLES)
        features = self.frontend(fitted)[None]  # Shape: (1, 1, 40, 251)
        logits = self.session.run(None, {self.input_name: features})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000
        
        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        order = np.argsort(probs)[::-1]
        top_idx = order[0]
        second_idx = order[1]
        
        return {
            'label': self.labels[top_idx],
            'confidence': float(probs[top_idx]),
            'margin': float(probs[top_idx] - probs[second_idx]),
            'latency_ms': latency_ms,
            'top3': {self.labels[i]: float(probs[i]) for i in order[:3]}
        }

class AntigravAssistant:
    def __init__(self, gpio=False, device=None, timeout_sec=5.0):
        self.predictor = AntigravPredictor()
        self.devices = Devices(gpio=gpio)
        self.mic = Microphone(device=device)
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
        
        # Ring buffer for 2.5 seconds of audio (40,000 samples)
        self.buffer = np.zeros(SAMPLES, dtype=np.float32)
        self.last_inference = 0.0
        self.vad_threshold = 0.012

    def trigger_wake(self, source="manual"):
        now = time.monotonic()
        self.state = 'LISTENING'
        self.devices.ducked = True
        self.inactivity_deadline = now + self.timeout_sec
        line = f"{time.strftime('%H:%M:%S')} [WAKE] Activated via {source}! Listening for commands (5s timeout)..."
        self.events.appendleft(line)
        self.devices.message = "Listening! Say a command now (e.g. 'Play music', 'Weather', 'Turn on lights')."
        print(f"[State] {line}", flush=True)

    def trigger_timeout(self):
        self.state = 'STANDBY'
        self.devices.ducked = False
        line = f"{time.strftime('%H:%M:%S')} [TIMEOUT] Inactivity timeout (5s elapsed). Returned to Standby."
        self.events.appendleft(line)
        self.devices.message = "Standby. Say 'Hi Dandan' to wake me up."
        print(f"[State] {line}", flush=True)

    def start(self):
        self.mic.start()
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.devices.message = "Standby. Say 'Hi Dandan' to wake me up."
        print(f"[Antigrav] Microphone: {self.mic.name}", flush=True)
        print(f"[Antigrav] Model: {self.predictor.model_size_kb} KB ({len(LABELS)} classes with Wake Word)", flush=True)
        print(f"[Antigrav] State: STANDBY. Waiting for 'Hi Dandan'...", flush=True)

    def loop(self):
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                
                # Check inactivity timeout when in LISTENING state
                if self.state == 'LISTENING' and now >= self.inactivity_deadline:
                    self.trigger_timeout()

                chunk = self.mic.read()
                if chunk is not None and len(chunk) > 0:
                    self.rms = float(np.sqrt(np.mean(chunk**2)))
                    
                    # Update ring buffer
                    n = len(chunk)
                    if n >= SAMPLES:
                        self.buffer[:] = chunk[-SAMPLES:]
                    else:
                        self.buffer[:-n] = self.buffer[n:]
                        self.buffer[-n:] = chunk
                    
                    # VAD check
                    if not self.muted and self.rms > self.vad_threshold and (now - self.last_inference) > 0.35 and now > self.cooldown_until:
                        pred = self.predictor.predict(self.buffer)
                        self.latest_prediction = pred
                        self.last_inference = now
                        label = pred['label']
                        conf = pred['confidence']
                        margin = pred['margin']

                        # --- STATE MACHINE TRANSITIONS ---
                        if self.state == 'STANDBY':
                            # In STANDBY: ONLY accept WAKE_WORD
                            if label == 'WAKE_WORD' and conf >= 0.60:
                                self.trigger_wake(source="Voice ('Hi Dandan')")
                                self.cooldown_until = now + 1.2

                        elif self.state == 'LISTENING':
                            # In LISTENING: accept any command class (or re-wake)
                            if label == 'WAKE_WORD':
                                self.inactivity_deadline = now + self.timeout_sec
                                self.cooldown_until = now + 0.8
                            elif conf >= 0.68 and margin >= 0.15:
                                # Command accepted!
                                msg = self.devices.execute(label)
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
            'latest': self.latest_prediction,
            'message': self.devices.message,
            'microphone': self.mic.name,
            'dropped': self.mic.dropped,
            'error': self.error,
            'events': list(self.events),
            'devices': self.devices.snapshot(),
            'model_kb': self.predictor.model_size_kb,
            'model_name': 'TinyDSCNN-48 Antigrav (32 Classes)',
            'classes_count': len(LABELS)
        }

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=2)
        self.mic.close()
        self.devices.close()

ASSISTANT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <title>Dandan AI · Modern Assistant</title>
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
      justify-content: center;
      position: relative;
      padding: 0 20px;
      text-align: center;
      z-index: 10;
    }
    .orb-stage {
      position: relative;
      width: 280px;
      height: 280px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }
    canvas#orbCanvas {
      width: 280px;
      height: 280px;
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
      margin-top: 28px;
      max-width: 720px;
      min-height: 120px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
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
      <span>Dandan AI</span>
    </div>
    <div class="header-actions">
      <div id="clockDisplay" class="clock-pill">12:00:00 PM</div>
      <button id="btnVoiceToggle" class="pill-btn active" onclick="toggleVoiceOutput()">
        <span id="voiceIcon">🔊</span> Voice Feedback
      </button>
      <button class="pill-btn" onclick="openAlarmModal()">🔔 Alarm Clock</button>
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

        if (currentState === 'LISTENING') {
          targetHue = 150;
          statusDot.className = 'status-dot active';
          eventTag.className = 'event-tag listening';
          eventTag.innerText = `LISTENING (${remaining.toFixed(1)}s)`;
          countdownBar.className = 'countdown-bar active';
          countdownFill.style.width = `${(remaining / 5.0) * 100}%`;
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

            if (t.url && htmlAudio.src !== t.url) {
              htmlAudio.src = t.url;
              if (devices.media_playing) htmlAudio.play().catch(e => console.log('Autoplay:', e));
            }
          }
        }

        isMediaPlaying = devices.media_playing;
        btnPlayPause.innerText = isMediaPlaying ? '⏸' : '▶';
        document.querySelectorAll('.bar').forEach(b => b.classList.toggle('playing', isMediaPlaying));

        currentVolume = devices.volume || 50;
        const ducked = devices.ducked;
        const effectiveVol = ducked ? 0.15 : (currentVolume / 100);
        htmlAudio.volume = Math.max(0, Math.min(1, effectiveVol));

        if (isMediaPlaying && htmlAudio.paused) {
          htmlAudio.play().catch(() => {});
        } else if (!isMediaPlaying && !htmlAudio.paused) {
          htmlAudio.pause();
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
  <title>Raspberry Pi 5 · TinyDSCNN-48 Studio</title>
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
    .btn-group{display:flex;gap:10px}
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
      <h1>TinyDSCNN-48 Studio</h1>
      <small style="color:#8da5b5">Raspberry Pi 5 · Option B (32 Classes)</small>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <a href="/" class="btn">✨ Modern UI</a>
      <div id="stateBadge" class="badge badge-standby">STANDBY</div>
    </div>
  </header>

  <section class="hero">
    <div id="orb" class="orb orb-standby"></div>
    <h2 id="msg">Standby. Say 'Hi Dandan' to wake me up.</h2>
    <div class="level-bar"><div id="levelFill" class="level-fill"></div></div>
    <div class="timer-container"><div id="timerFill" class="timer-fill"></div></div>
    <small id="sub" style="color:#8da5b5">Microphone active · Waiting for 'Hi Dandan'</small>
  </section>

  <div class="cards">
    <div class="card"><small>Command</small><div id="cmd" class="val">-</div></div>
    <div class="card"><small>Confidence</small><div id="conf" class="val">-</div></div>
    <div class="card"><small>Latency</small><div id="lat" class="val">1.2 ms</div></div>
    <div class="card"><small>Model Footprint</small><div id="size" class="val">38.5 KB</div></div>
  </div>

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
        <p>• <b>Music:</b> <span id="devMedia">Paused</span> (Vol: <span id="devVol">50%</span>)</p>
        <p>• <b>Clock & Alarms:</b> <span id="devAlarm">None</span></p>
        <p>• <b>Real-time Weather:</b> Open-Meteo REST API (Diliman, QC)</p>
      </div>
    </div>
  </div>
</main>
<script>
  async function update() {
    try {
      const res = await fetch('/api/state');
      const d = await res.json();
      const st = d.state;
      const b = document.getElementById('stateBadge');
      const orb = document.getElementById('orb');
      b.className = 'badge ' + (st === 'LISTENING' ? 'badge-listening' : 'badge-standby');
      b.innerText = st;
      orb.className = 'orb ' + (st === 'LISTENING' ? 'orb-listening' : 'orb-standby');

      document.getElementById('msg').innerText = d.message || '';
      document.getElementById('levelFill').style.width = Math.min(100, Math.round((d.rms || 0) * 1500)) + '%';

      if (st === 'LISTENING') {
        const pct = ((d.remaining_sec || 0) / (d.timeout_sec || 5.0)) * 100;
        document.getElementById('timerFill').style.width = pct + '%';
        document.getElementById('sub').innerText = `LISTENING · ${d.remaining_sec}s remaining before sleep`;
      } else {
        document.getElementById('timerFill').style.width = '0%';
        document.getElementById('sub').innerText = "Microphone active · Say 'Hi Dandan' to wake me up";
      }

      if (d.latest && d.latest.label) {
        document.getElementById('cmd').innerText = d.latest.label;
        document.getElementById('conf').innerText = Math.round(d.latest.confidence * 100) + '%';
        document.getElementById('lat').innerText = d.latest.latency_ms.toFixed(1) + ' ms';
      }

      if (d.events) {
        document.getElementById('terminal').innerText = d.events.slice(0, 10).join('\n');
      }

      if (d.devices) {
        document.getElementById('devLights').innerText = d.devices.lights_percent + '%';
        document.getElementById('devMedia').innerText = (d.devices.media_playing ? 'Playing: ' + (d.devices.current_track?.title || '') : 'Paused');
        document.getElementById('devVol').innerText = d.devices.volume + '%';
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

def run_server(port=7860, gpio=False):
    assistant = AntigravAssistant(gpio=gpio, timeout_sec=5.0)
    assistant.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path == '/api/state':
                payload = json.dumps(assistant.snapshot()).encode('utf-8')
                mime = 'application/json'
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
        server = ReusableServer(('0.0.0.0', port), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"\n[Dandan AI Assistant Active] Serving on http://0.0.0.0:{port}", flush=True)
        print(f"Modern Assistant UI: http://192.168.254.106:{port}/assistant", flush=True)
        print(f"Studio Dashboard:    http://192.168.254.106:{port}/studio", flush=True)

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
    parser.add_argument('--gpio', action='store_true')
    args = parser.parse_args()
    run_server(port=args.port, gpio=args.gpio)
