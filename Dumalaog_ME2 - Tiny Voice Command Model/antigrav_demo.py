"""Always-listening edge voice assistant with Wake Word gating & 5s inactivity timeout on Raspberry Pi 5."""
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

import numpy as np
import onnxruntime as ort

# Add parent directory to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinyvcm_antigrav.config import LABELS, SAMPLES, SR, MELS, TIME_STEPS
from tinyvcm_antigrav.frontend import Frontend, fit_audio
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
        
        # State Machine: 'STANDBY' -> 'LISTENING' -> 'TIMEOUT (back to STANDBY)'
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
        self.inactivity_deadline = now + self.timeout_sec
        line = f"{time.strftime('%H:%M:%S')} [WAKE] Activated via {source}! Listening for commands (5s timeout)..."
        self.events.appendleft(line)
        self.devices.message = "Listening! Say a command now (e.g. 'Turn on lights', 'Weather')."
        print(f"[State] {line}", flush=True)

    def trigger_timeout(self):
        self.state = 'STANDBY'
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
                            else:
                                # Ambient speech / non-wake word is ignored in Standby
                                pass

                        elif self.state == 'LISTENING':
                            # In LISTENING: accept any command class (or re-wake)
                            if label == 'WAKE_WORD':
                                # Re-wake resets the 5-second countdown
                                self.inactivity_deadline = now + self.timeout_sec
                                self.cooldown_until = now + 0.8
                            elif conf >= 0.68 and margin >= 0.15:
                                # Command accepted!
                                msg = self.devices.execute(label)
                                line = f"{time.strftime('%H:%M:%S')} [{label}] ({conf*100:.1f}%, {pred['latency_ms']:.1f}ms) -> {msg}"
                                self.events.appendleft(line)
                                print(f"[Command] {line}", flush=True)
                                # Reset inactivity timer after successful command
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

DASHBOARD_HTML = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Raspberry Pi 5 · TinyDSCNN-48 Antigrav</title>
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
    button{background:#1a2834;color:#e2e8f0;border:1px solid #3b82f6;border-radius:8px;padding:8px 14px;cursor:pointer;font-weight:600;font-size:12px}
    button:hover{background:#2563eb}
    .btn-wake{background:#064e3b;border-color:#10b981;color:#a7f3d0}
    .btn-wake:hover{background:#059669}
    
    footer{font-size:12px;color:#64748b;margin-top:28px;text-align:center;line-height:1.6}
    @media(max-width:768px){.cards{grid-template-columns:repeat(2,1fr)}.row{grid-template-columns:1fr}}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>TinyDSCNN-48 Antigrav</h1>
      <small style="color:#94a3b8">Raspberry Pi 5 Edge Assistant · Wake Word Gating & 5s Inactivity Timeout</small>
    </div>
    <div class="badge badge-standby" id="badge">STANDBY</div>
  </header>

  <section class="hero">
    <div class="orb orb-standby" id="orb"></div>
    <h2 id="state">STANDBY</h2>
    <p id="msg" style="color:#94a3b8;margin:6px 0">Say "Hi Dandan" to wake up...</p>
    
    <!-- Audio VU Meter -->
    <div class="level-bar"><div class="level-fill" id="vu"></div></div>
    
    <!-- Inactivity Timer Countdown (Visible in LISTENING state) -->
    <div id="countdown_box" style="display:none;margin-top:10px">
      <small style="color:#f59e0b">Inactivity Timeout: <span id="countdown_text">5.0</span>s remaining</small>
      <div class="timer-container"><div class="timer-fill" id="timer_bar"></div></div>
    </div>
    
    <div style="margin-top:10px;font-size:14px;color:#38bdf8" id="pred">Waiting for speech...</div>
  </section>

  <div class="cards">
    <div class="card"><small>LIGHTS / LED</small><div class="val" id="lights">0%</div></div>
    <div class="card"><small>TIMER</small><div class="val" id="timer">Not set</div></div>
    <div class="card"><small>THERMOSTAT</small><div class="val" id="temp">22°C</div></div>
    <div class="card"><small>MEDIA</small><div class="val" id="media">Idle</div></div>
  </div>

  <div class="row">
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <b>Live Event Stream</b>
        <div class="btn-group">
          <button class="btn-wake" onclick="triggerWake()">Simulate Wake ("Hi Dandan")</button>
          <button onclick="testWeather()">Test Weather</button>
        </div>
      </div>
      <div class="terminal" id="events">Listening on Fifine USB Microphone (Standby)...</div>
    </div>
    <div class="vocab">
      <b>Wake Word & Command Flow:</b><br>
      1. <b>Wake Up:</b> Say <b>"Hi Dandan"</b> (turns active green for 5s)<br>
      2. <b>Commands (within 5s):</b><br>
      • <b>Weather:</b> "What is the weather?" (Open-Meteo REST API)<br>
      • <b>Lights:</b> "Turn on the lights", "Lights off", "Brightness 60%"<br>
      • <b>Colors:</b> "Change color to red / green / blue"<br>
      • <b>Timers:</b> "Timer 10s", "Timer 30s", "Timer 1m"<br>
      • <b>Alarms:</b> "Alarm 6:00 AM", "Alarm 8:00 AM", "Alarm 9:00 PM"<br>
      • <b>Clock:</b> "What time is it?"<br>
      3. <b>5s Timeout:</b> If silent for 5 seconds, goes back to Standby!
    </div>
  </div>

  <footer id="footer_info">Raspberry Pi 5 ARM Cortex-A76 · 32-Class INT8 ONNX (39.3 KB) · 1.3 ms Latency</footer>
</main>

<script>
  const el = id => document.getElementById(id);
  async function triggerWake() {
    await fetch('/api/wake', {method:'POST'});
    update();
  }
  async function testWeather() {
    await fetch('/api/test_weather', {method:'POST'});
    update();
  }
  async function update() {
    try {
      const res = await fetch('/api/state', {cache:'no-store'});
      const s = await res.json();
      
      const isListening = s.state === 'LISTENING';
      el('state').textContent = s.state;
      el('msg').textContent = s.message;
      
      el('orb').className = 'orb ' + (isListening ? 'orb-listening' : 'orb-standby');
      el('badge').className = 'badge ' + (isListening ? 'badge-listening' : 'badge-standby');
      el('badge').textContent = isListening ? 'LISTENING (ACTIVE)' : 'STANDBY (SLEEPING)';
      
      el('vu').style.width = Math.min(100, s.rms * 900) + '%';
      
      // Inactivity Countdown
      if (isListening && s.remaining_sec > 0) {
        el('countdown_box').style.display = 'block';
        el('countdown_text').textContent = s.remaining_sec.toFixed(1);
        const pct = (s.remaining_sec / s.timeout_sec) * 100;
        el('timer_bar').style.width = pct + '%';
      } else {
        el('countdown_box').style.display = 'none';
      }
      
      if (s.latest && s.latest.label) {
        el('pred').textContent = `Detected: ${s.latest.label} (${Math.round(s.latest.confidence*100)}% conf | ${s.latest.latency_ms.toFixed(1)} ms)`;
      }
      
      el('lights').textContent = s.devices.lights_percent + '%';
      el('timer').textContent = s.devices.timer_seconds === null ? 'Not set' : s.devices.timer_seconds + 's';
      el('temp').textContent = s.devices.temperature_f + '°C';
      el('media').textContent = s.devices.media_playing ? 'Playing' : 'Idle';
      
      if (s.events && s.events.length > 0) {
        el('events').textContent = s.events.join('\n');
      }
      
      el('footer_info').textContent = `Microphone: ${s.microphone} | Model: ${s.model_name} (${s.model_kb} KB) | Inactivity Timeout: ${s.timeout_sec}s`;
    } catch (e) {
      el('state').textContent = 'DISCONNECTED';
    }
  }
  setInterval(update, 250);
  update();
</script>
</body>
</html>
'''

def run_server(port=7860, gpio=False):
    assistant = AntigravAssistant(gpio=gpio, timeout_sec=5.0)
    assistant.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == '/api/state':
                payload = json.dumps(assistant.snapshot()).encode('utf-8')
                mime = 'application/json'
            elif self.path == '/':
                payload = DASHBOARD_HTML.encode('utf-8')
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
            if self.path == '/api/wake':
                assistant.trigger_wake(source="Web UI / API")
                payload = json.dumps(assistant.snapshot()).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == '/api/test_weather':
                if assistant.state != 'LISTENING':
                    assistant.trigger_wake(source="Test Weather (Auto-Wake)")
                msg = assistant.devices.execute('WEATHER')
                line = f"{time.strftime('%H:%M:%S')} [TEST WEATHER] -> {msg}"
                assistant.events.appendleft(line)
                payload = json.dumps(assistant.snapshot()).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = None
    try:
        server = ReusableServer(('0.0.0.0', port), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"\n[Antigrav Dashboard Active] Serving on http://0.0.0.0:{port}", flush=True)
        print(f"Open in your browser: http://192.168.254.106:{port}", flush=True)

        def shutdown_handler(*args):
            print("\nReceived stop signal...", flush=True)
            assistant.stop_event.set()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        while not assistant.stop_event.wait(0.5):
            if assistant.error:
                raise RuntimeError(assistant.error)
    finally:
        print("Shutting down assistant and server...", flush=True)
        assistant.close()
        if server:
            server.shutdown()
            server.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--gpio', action='store_true')
    args = parser.parse_args()
    run_server(port=args.port, gpio=args.gpio)
