#!/usr/bin/env python3
"""Always-listening native microphone with a local dashboard. Works without a browser."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time
import numpy as np

from tinyvcm.config import ROOT
from tinyvcm.devices import Devices
from tinyvcm.microphone import Microphone
from tinyvcm.runtime import Predictor, StreamEngine


class Sound:
    """Local synthesized demonstration tune; output volume actually changes."""
    def __init__(self,devices):
        self.devices=devices; self.stream=None; self.position=0; self.chime_left=0; self.chime_hz=1000
        self.error=None; self.rate=48000
        try:
            import sounddevice as sd
            device=sd.query_devices(kind='output'); self.rate=int(device['default_samplerate'])
            self.stream=sd.OutputStream(samplerate=self.rate,channels=1,dtype='float32',callback=self.callback)
            self.stream.start()
        except Exception as exc:
            self.error=str(exc)

    def callback(self,out,frames,info,status):
        t=(np.arange(frames)+self.position)/self.rate
        wave=np.zeros(frames,dtype=np.float32)
        if self.devices.media:
            notes=np.array([261.63,329.63,392.,329.63,293.66,349.23,440.,349.23])
            freq=notes[(t*2).astype(int)%len(notes)]
            envelope=np.minimum(1,(t*2%1)*20)*np.minimum(1,(1-t*2%1)*8)
            volume=self.devices.volume/100*(.15 if self.devices.ducked else 1)
            wave+=.12*volume*np.sin(2*np.pi*freq*t)*envelope
        if self.chime_left>0:
            n=min(frames,self.chime_left)
            wave[:n]+=.08*np.sin(2*np.pi*self.chime_hz*t[:n])
            self.chime_left-=n
        out[:,0]=wave
        self.position+=frames

    def chime(self,kind):
        self.chime_hz={'WAKE':1400,'COMMAND':1000,'TIMEOUT':500,'ALERT':1800,'REJECT':650}[kind]
        self.chime_left=int(self.rate*.16)

    def close(self):
        if self.stream: self.stream.stop(); self.stream.close()


class Assistant:
    def __init__(self,gpio=False,device=None,sound=True):
        self.predictor=Predictor(); self.engine=StreamEngine(self.predictor)
        self.devices=Devices(gpio); self.mic=Microphone(device)
        self.sound=Sound(self.devices) if sound else None
        self.events=deque(maxlen=40); self.stop_event=threading.Event()
        self.error=None; self.muted=False; self.worker=None

    def start(self):
        self.mic.start()
        self.worker=threading.Thread(target=self.loop,daemon=True); self.worker.start()
        print('Microphone: '+self.mic.name,flush=True)
        print('Listening locally. Say Hi Dandan, pause for the chime, then a command.',flush=True)

    def loop(self):
        try:
            while not self.stop_event.is_set():
                chunk=self.mic.read()
                if chunk is not None and not self.muted:
                    event=self.engine.feed(chunk)
                    if event:
                        message=self.devices.event(event)
                        if self.sound: self.sound.chime(event['event'])
                        if event['event']=='COMMAND' and event['prediction']['label']=='media_next' and self.sound:
                            self.sound.position=0
                        line=f'{time.strftime("%H:%M:%S")} {event["event"]}: {message}'
                        self.events.appendleft(line); print(line,flush=True)
                if self.devices.tick():
                    if self.sound: self.sound.chime('ALERT')
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            print('Audio worker stopped: '+self.error,flush=True)

    def snapshot(self):
        meta=self.predictor.meta
        return dict(state='ERROR' if self.error else 'MUTED' if self.muted else self.engine.gate.state,
            message=self.devices.message,rms=self.engine.rms,latest=self.engine.latest,
            microphone=self.mic.name,dropped=self.mic.dropped,audio_status=self.mic.status,
            output_error=self.sound.error if self.sound else None,error=self.error,
            events=list(self.events),devices=self.devices.snapshot(),
            model_kb=round(meta['metrics']['int8_bytes']/1000,1),
            baseline=not meta['audit']['human_dataset_requirement_met'])

    def close(self):
        self.stop_event.set()
        if self.worker: self.worker.join(timeout=2)
        self.mic.close()
        if self.sound: self.sound.close()
        self.devices.close()


PAGE=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dandan · Tiny VCM</title><style>
*{box-sizing:border-box}body{margin:0;background:#101922;color:#edf3f7;font:16px system-ui,sans-serif}main{max-width:1000px;margin:auto;padding:42px 28px}header{display:flex;justify-content:space-between;align-items:center}h1{font-size:30px;margin:0}small,.muted{color:#97adbc}.pill{border:1px solid #375362;border-radius:30px;padding:8px 16px;color:#80decc;font-size:12px;letter-spacing:1.6px}.notice{margin:28px 0;padding:13px 18px;background:#393122;color:#f2d299;border-radius:12px}.hero{background:#192834;border:1px solid #304755;border-radius:24px;text-align:center;padding:40px 24px}.orb{margin:auto;width:112px;height:112px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#a5eedc,#32a68e 50%,#183c43);box-shadow:0 0 44px #48ddbd24}.listening{animation:pulse 1.2s infinite alternate}@keyframes pulse{to{box-shadow:0 0 66px #74efd68a;transform:scale(1.06)}}h2{font-size:30px;letter-spacing:2px;margin-bottom:10px}.meter{height:6px;background:#304755;max-width:360px;margin:24px auto 8px;border-radius:8px;overflow:hidden}.meter div{height:100%;background:#71d8c0;width:0;transition:width .15s}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}.card{background:#192834;border:1px solid #304755;padding:20px;border-radius:16px}.value{font-size:25px;margin-top:10px}.row{display:flex;justify-content:space-between;gap:20px;margin:24px 0}.commands{font-size:14px;line-height:1.8;color:#b7c9d5}pre{white-space:pre-wrap;font:13px ui-monospace,monospace;color:#a9c9c1;min-height:75px}.error{color:#ffad9e}button{background:#233d49;color:#e7f8f3;border:1px solid #486675;border-radius:8px;padding:10px 18px;cursor:pointer}footer{font-size:12px;line-height:1.8;margin-top:28px;color:#91a8b8}@media(max-width:650px){.cards{grid-template-columns:1fr}main{padding:22px 16px}.row{display:block}}
</style></head><body><main><header><div><h1>Dandan</h1><small>Tiny voice command assistant</small></div><div class="pill">LOCAL · OFFLINE</div></header>
<div class="notice" id="notice">Synthetic baseline · recognition of real people is not validated yet.</div>
<section class="hero"><div class="orb" id="orb"></div><h2 id="state">CONNECTING</h2><p id="message">Connecting to your microphone…</p><div class="meter"><div id="level"></div></div><small id="prediction">Waiting for audio</small></section>
<div class="cards"><div class="card"><small>LIGHTS / LED</small><div class="value" id="lights">0%</div></div><div class="card"><small>TIMER</small><div class="value" id="timer">Not set</div></div><div class="card"><small>DEMO MEDIA</small><div class="value" id="media">Paused</div></div></div>
<div class="row"><div><b>Say “Hi Dandan”</b><div class="commands">Pause for the wake chime, then try:<br>“Turn on the lights” · “Lights off” · “Play music”<br>“Volume up” · “Timer for one minute” · “What time is it”</div></div><div><button onclick="mute()" id="mute">Mute recognition</button><p><a style="color:#80decc" href="http://127.0.0.1:7862" target="_blank">Open dataset recorder ↗</a></p></div></div>
<div class="card"><small>ACTIVITY</small><pre id="events">No commands yet.</pre><div class="error" id="error"></div></div>
<footer id="details"></footer><footer>Supported vocabulary only. Calls, weather and thermostat control are demonstration intents. Timers and the clock use real time. Microphone audio stays in memory and is not saved by this assistant.</footer></main>
<script>const el=id=>document.getElementById(id);let muted=false;
async function mute(){await fetch('/api/mute',{method:'POST'});await update()}
async function update(){try{const s=await (await fetch('/api/state',{cache:'no-store'})).json();el('state').textContent=s.state;el('message').textContent=s.message;el('orb').className='orb '+(s.state==='LISTENING'?'listening':'');el('level').style.width=Math.min(100,s.rms*800)+'%';el('notice').hidden=!s.baseline;el('lights').textContent=s.devices.lights_percent+'%';el('timer').textContent=s.devices.timer_seconds===null?'Not set':s.devices.timer_seconds+' seconds';el('media').textContent=(s.devices.media_playing?'Playing':'Paused')+' · '+s.devices.volume+'%';el('events').textContent=s.events.join('\n')||'No commands yet.';el('prediction').textContent=s.latest.label?s.latest.label+' · '+Math.round(s.latest.confidence*100)+'% confidence':'Waiting for audio';el('error').textContent=s.error||s.output_error||'';el('details').textContent='Microphone: '+s.microphone+' | INT8 model: '+s.model_kb+' KB | Dropped audio blocks: '+s.dropped;el('mute').textContent=s.state==='MUTED'?'Resume recognition':'Mute recognition'}catch(e){el('state').textContent='DISCONNECTED';el('error').textContent='Start the PC demo launcher to reconnect.'}}
update();setInterval(update,400);</script></body></html>'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpio',action='store_true')
    parser.add_argument('--device',type=int)
    parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--no-sound',action='store_true')
    parser.add_argument('--check-seconds',type=float,default=0)
    args=parser.parse_args()
    assistant=Assistant(args.gpio,args.device,not args.no_sound)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            if self.path=='/api/state':
                payload=json.dumps(assistant.snapshot()).encode(); mime='application/json'
            elif self.path=='/': payload=PAGE.encode(); mime='text/html; charset=utf-8'
            else: self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',mime)
            self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(payload)
        def do_POST(self):
            if self.path!='/api/mute': self.send_error(404); return
            origin=self.headers.get('Origin')
            if origin and origin!=f'http://127.0.0.1:{args.port}': self.send_error(403); return
            assistant.muted=not assistant.muted; assistant.engine.clear()
            assistant.engine.gate.state='STANDBY'
            assistant.devices.ducked=False
            self.send_response(204); self.end_headers()
    server=None
    try:
        assistant.start()
        server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        print(f'Dashboard: http://127.0.0.1:{args.port}',flush=True)
        def stop(*_): assistant.stop_event.set()
        signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
        started=time.monotonic()
        while not assistant.stop_event.wait(.3):
            if assistant.error: raise RuntimeError(assistant.error)
            if args.check_seconds and time.monotonic()-started>=args.check_seconds:
                print(json.dumps(assistant.snapshot(),indent=2),flush=True); break
    finally:
        if server: server.shutdown(); server.server_close()
        assistant.close()


if __name__=='__main__': main()
