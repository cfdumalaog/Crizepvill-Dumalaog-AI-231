"""Password-gated, inference-only remote microphone evaluator.

It binds to loopback, accepts short in-memory WAV uploads, and never saves audio
or exposes GPIO, weather, music, or appliance-action routes.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

from vcm_app import VCMPredictor
from tinyvcm_model.config import SR, SAMPLES
from tinyvcm_model.frontend import fit_audio

ROOT = Path(__file__).resolve().parent
AUTH_ITERATIONS = 200_000
MAX_AUDIO_BYTES = 192_000

REMOTE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ME2 - VCM on Raspberry Pi 5</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#07111a;color:#eaf3f8;font:16px system-ui,sans-serif}
main{width:min(900px,94vw);margin:32px auto;padding:28px;background:#101c27;border:1px solid #263746;border-radius:22px}
h1{font-size:clamp(22px,4vw,34px);margin:14px 0 8px}.muted{color:#9bb0bf;line-height:1.5}.pill{display:inline-flex;border-radius:999px;padding:7px 12px;background:#172a38;color:#a7f3d0;font-size:12px}
.panel{margin-top:22px;padding:20px;border:1px solid #263746;border-radius:16px;background:#0b151e}.row{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}button{border:0;border-radius:10px;padding:12px 18px;color:#062116;background:#48e39e;font-weight:750;cursor:pointer}button.secondary{color:#e2e8f0;background:#253746}button:disabled{opacity:.4;cursor:not-allowed}
.score{font-size:clamp(24px,5vw,40px);font-weight:750;color:#fff}.status{min-height:25px;color:#91a7b7}.result{font:13px ui-monospace,Consolas,monospace;color:#b8d2df;white-space:pre-wrap;overflow-wrap:anywhere}.privacy{border-left:3px solid #48e39e;padding-left:12px;color:#b6c8d2;font-size:13px}
</style></head><body><main>
<span class="pill">Local ONNX · password protected · no cloud speech model</span>
<h1>ME2 - VCM on Raspberry Pi 5</h1>
<p class="muted">Test the two models in order. Capture a two-second wake phrase; the evaluator checks five temporal placements to match the offline wake test. After wake, capture a 2.5-second intent phrase. This page reports predictions only; it does not run appliance actions.</p>
<div class="panel"><strong id="mode">Standby: capture a wake phrase</strong><div class="row"><button id="wake">Capture wake phrase</button><button id="intent" class="secondary" disabled>Capture intent</button><button id="reset" class="secondary" hidden>Reset</button></div><p id="status" class="status">The browser mic is used only after a button press.</p><div id="score" class="score">—</div><pre id="result" class="result"></pre></div>
<p class="privacy">Audio is held in memory for one request and is not written to disk. Stop or leave the page to release the microphone. This is a remote evaluation page, not a continuous wake listener.</p>
</main><script>
const wakeButton=document.getElementById('wake'),intentButton=document.getElementById('intent'),resetButton=document.getElementById('reset');
const modeText=document.getElementById('mode'),statusText=document.getElementById('status'),scoreText=document.getElementById('score'),resultText=document.getElementById('result');
let commandMode=false,busy=false,timeoutId=null;
function reset(){commandMode=false;intentButton.disabled=true;wakeButton.disabled=false;resetButton.hidden=true;modeText.textContent='Standby: capture a wake phrase';statusText.textContent='The browser mic is used only after a button press.';scoreText.textContent='—';resultText.textContent='';if(timeoutId)clearTimeout(timeoutId)}
function wavBlob(samples){const rate=16000,b=new ArrayBuffer(44+samples.length*2),v=new DataView(b),put=(p,s)=>{for(let i=0;i<s.length;i++)v.setUint8(p+i,s.charCodeAt(i))};put(0,'RIFF');v.setUint32(4,36+samples.length*2,true);put(8,'WAVE');put(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);put(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,x<0?x*32768:x*32767,true)}return new Blob([b],{type:'audio/wav'})}
async function captureWindow(mode){let stream=null,ctx=null;try{stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});ctx=new AudioContext();await ctx.resume();const src=ctx.createMediaStreamSource(stream),proc=ctx.createScriptProcessor(4096,1,1),gain=ctx.createGain();gain.gain.value=0;let parts=[],count=0;return await new Promise((resolve,reject)=>{const seconds=mode==='wake'?2:2.5,target=Math.ceil(ctx.sampleRate*seconds),timer=setTimeout(()=>reject(new Error('Timed out waiting for microphone audio')),9000);proc.onaudioprocess=e=>{const a=new Float32Array(e.inputBuffer.getChannelData(0));parts.push(a);count+=a.length;if(count>=target){clearTimeout(timer);const joined=new Float32Array(count);let offset=0;for(const part of parts){joined.set(part,offset);offset+=part.length}const n=Math.round(target*16000/ctx.sampleRate),out=new Float32Array(n);for(let i=0;i<n;i++){const pos=i*ctx.sampleRate/16000,l=Math.floor(pos),f=pos-l,r=Math.min(l+1,joined.length-1);out[i]=joined[l]*(1-f)+joined[r]*f}proc.onaudioprocess=null;resolve(wavBlob(out))}};src.connect(proc);proc.connect(gain);gain.connect(ctx.destination)})}finally{if(stream)stream.getTracks().forEach(t=>t.stop());if(ctx)await ctx.close()}}
async function evaluate(mode){if(busy)return;busy=true;wakeButton.disabled=true;intentButton.disabled=true;statusText.textContent='Capturing a short window…';scoreText.textContent='…';resultText.textContent='';try{const wav=await captureWindow(mode);statusText.textContent='Running local ONNX inference…';const response=await fetch('/api/evaluate?mode='+mode,{method:'POST',headers:{'Content-Type':'audio/wav'},body:wav});if(!response.ok)throw new Error('Evaluation failed ('+response.status+')');const d=await response.json();scoreText.textContent=d.label+' · '+Math.round(d.confidence*100)+'%';resultText.textContent=JSON.stringify({mode:d.mode,wake_probability:d.wake_probability,wake_threshold:d.wake_threshold,window_count:d.window_count,top3:d.top3,latency_ms:d.latency_ms},null,2);if(mode==='wake'){commandMode=d.label==='WAKE_WORD';intentButton.disabled=!commandMode;wakeButton.disabled=commandMode;resetButton.hidden=!commandMode;modeText.textContent=commandMode?'Wake accepted: capture an intent phrase':'Wake not accepted: try again';statusText.textContent=commandMode?'Intent window is open for ten seconds.':'No appliance action was performed.';if(commandMode)timeoutId=setTimeout(reset,10000)}else{statusText.textContent='Intent prediction complete. No appliance action was performed.';reset()}}catch(e){statusText.textContent=e.message||String(e);wakeButton.disabled=false;intentButton.disabled=!commandMode}finally{busy=false}}
wakeButton.addEventListener('click',()=>evaluate('wake'));intentButton.addEventListener('click',()=>evaluate('command'));resetButton.addEventListener('click',reset);
</script></body></html>"""


def _predict_wake_placements(predictor, audio: np.ndarray, count: int = 5) -> dict:
    """Max-pool wake probability across temporal placements used by evaluation."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) > SAMPLES:
        audio = fit_audio(audio)
    max_start = max(0, SAMPLES - len(audio))
    results = []
    for start in np.linspace(0, max_start, count, dtype=int):
        window = np.zeros(SAMPLES, dtype=np.float32)
        window[start:start + len(audio)] = audio
        results.append(predictor.predict(window, mode="wake"))
    best_index = int(np.argmax([r["wake_probability"] for r in results]))
    best = dict(results[best_index])
    wake_probability = max(float(r["wake_probability"]) for r in results)
    is_wake = wake_probability >= float(predictor.wake_threshold)
    best.update({
        "label": "WAKE_WORD" if is_wake else "NON_WAKE",
        "confidence": wake_probability if is_wake else 1.0 - wake_probability,
        "wake_probability": wake_probability,
        "window_count": len(results),
    })
    return best


def _load_auth() -> tuple[str, bytes, bytes]:
    try:
        username = os.environ["VCM_REMOTE_USER"]
        salt = bytes.fromhex(os.environ["VCM_REMOTE_SALT"])
        digest = bytes.fromhex(os.environ["VCM_REMOTE_PASSWORD_HASH"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Remote auth is missing; start through scripts/start_remote_vcm.py") from exc
    if len(salt) < 16 or len(digest) != 32:
        raise RuntimeError("Remote auth hash parameters are invalid")
    return username, salt, digest


def make_handler(predictor, auth_user: str, auth_salt: bytes, auth_hash: bytes):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ME2-VCM-Remote/1.0"

        def log_message(self, _format, *_args):
            return

        def _reply(self, status: int, payload: bytes, content_type: str, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, payload: dict, headers=None):
            self._reply(status, json.dumps(payload, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", headers)

        def _authorized(self) -> bool:
            try:
                scheme, token = self.headers.get("Authorization", "").split(" ", 1)
                if scheme.lower() != "basic":
                    return False
                user_pass = base64.b64decode(token, validate=True).decode("utf-8")
                username, password = user_pass.split(":", 1)
            except (ValueError, UnicodeError):
                return False
            if not hmac.compare_digest(username, auth_user):
                return False
            candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), auth_salt, AUTH_ITERATIONS, dklen=32)
            return hmac.compare_digest(candidate, auth_hash)

        def _require_auth(self) -> bool:
            if self._authorized():
                return True
            self._reply(401, b"Authentication required", "text/plain; charset=utf-8", {"WWW-Authenticate": 'Basic realm="ME2 - VCM on Raspberry Pi 5", charset="UTF-8"'})
            return False

        def do_GET(self):
            if not self._require_auth():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._reply(200, REMOTE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/health":
                self._json(200, {"ok": True, "model": predictor.model_name, "classes": len(predictor.labels)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._require_auth():
                return
            parsed = urlparse(self.path)
            if parsed.path != "/api/evaluate":
                self._json(404, {"error": "not found"})
                return
            if self.headers.get_content_type() != "audio/wav":
                self._json(415, {"error": "send one WAV clip as audio/wav"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "invalid content length"})
                return
            if size <= 44 or size > MAX_AUDIO_BYTES:
                self._json(413, {"error": "WAV upload size is out of range"})
                return
            mode = parse_qs(parsed.query).get("mode", [""])[0]
            if mode not in ("wake", "command"):
                self._json(400, {"error": "mode must be wake or command"})
                return
            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("incomplete WAV request")
                audio, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if rate != SR:
                    from scipy.signal import resample_poly
                    gcd = np.gcd(int(rate), SR)
                    audio = resample_poly(audio, SR // gcd, int(rate) // gcd).astype(np.float32)
                if audio.ndim != 1 or not len(audio) or len(audio) > SAMPLES or not np.isfinite(audio).all():
                    raise ValueError("invalid clip")
                result = _predict_wake_placements(predictor, audio) if mode == "wake" else predictor.predict(audio, mode=mode)
                result["mode"] = mode
                result["wake_threshold"] = float(predictor.wake_threshold) if mode == "wake" else None
                self._json(200, result)
            except Exception:
                self._json(400, {"error": "WAV could not be decoded or evaluated"})

    return Handler


def create_server(predictor=None, host: str = "127.0.0.1", port: int = 7870):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The remote evaluator must bind to loopback; use an authenticated tunnel for public access")
    username, salt, digest = _load_auth()
    if predictor is None:
        intent = Path(os.environ.get("VCM_REMOTE_INTENT_MODEL", ""))
        wake = Path(os.environ.get("VCM_REMOTE_WAKE_MODEL", ""))
        threshold = os.environ.get("VCM_REMOTE_WAKE_THRESHOLD")
        if not intent.is_file() or not wake.is_file() or threshold is None:
            raise RuntimeError("Model paths and validation threshold are required")
        predictor = VCMPredictor(intent, wake, float(threshold))
    server = ThreadingHTTPServer((host, port), make_handler(predictor, username, salt, digest))
    server.daemon_threads = True
    return server


def main():
    server = create_server()
    print("ME2 - VCM on Raspberry Pi 5 remote evaluator: authenticated, loopback-only; audio is not saved", flush=True)
    print(f"Local evaluator: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
