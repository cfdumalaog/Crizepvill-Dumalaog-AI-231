"""CPU-only runtime; no torch, ASR, pretrained weights, or network services."""
import argparse
from collections import deque
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits
from .config import ROOT, SR, SAMPLES
from .frontend import Frontend, fit_audio


def session_for(path):
    # Bound both feature-extraction BLAS and inference thread counts on 4 GB Pi.
    threadpool_limits(limits=1)
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def model_dir():
    if (ROOT/'models/metadata.json').exists():
        return ROOT/'models'
    latest=json.loads((ROOT/'runs/latest.json').read_text(encoding='utf-8'))
    return ROOT/'runs'/latest['run']


class Predictor:
    def __init__(self, directory=None):
        directory=Path(directory) if directory else model_dir()
        self.meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        path=directory/self.meta['model_file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.meta['int8_sha256']:
            raise ValueError('Model checksum does not match metadata')
        self.session=session_for(path)
        self.frontend=Frontend()
        self.labels=self.meta['labels']

    def __call__(self,audio):
        start=time.perf_counter()
        features=self.frontend(audio)[None]
        logits=self.session.run(None,{'features':features})[0][0]
        probs=np.exp(logits-logits.max()); probs/=probs.sum()
        order=np.argsort(probs)[::-1]
        return dict(label=self.labels[order[0]],confidence=float(probs[order[0]]),
                    margin=float(probs[order[0]]-probs[order[1]]),
                    latency_ms=(time.perf_counter()-start)*1000,
                    top3={self.labels[i]:float(probs[i]) for i in order[:3]})


class WakeState:
    """Two consecutive windows, fresh command audio, timeout and feedback holdoff."""
    def __init__(self,wake_threshold=.8,command_threshold=.8,margin=.2,timeout=7):
        self.wake_threshold=wake_threshold
        self.command_threshold=command_threshold
        self.margin=margin
        self.timeout=timeout
        self.state='STANDBY'
        self.deadline=0
        self.blocked_until=0
        self.candidate=None
        self.hits=0

    def tick(self,now):
        if self.state=='LISTENING' and now>=self.deadline:
            self.state='STANDBY'; self.candidate=None; self.hits=0
            return {'event':'TIMEOUT'}

    def accept(self,prediction,now,endpoint=False):
        event=self.tick(now)
        if event:
            return event
        if now<self.blocked_until:
            return None
        label=prediction['label']
        expected=label=='wake_word' if self.state=='STANDBY' else not (label=='wake_word' or label.startswith('_'))
        threshold=self.wake_threshold if self.state=='STANDBY' else self.command_threshold
        if not expected or prediction['confidence']<threshold or prediction['margin']<self.margin:
            self.candidate=None; self.hits=0
            return None
        self.hits=self.hits+1 if self.candidate==label else 1
        self.candidate=label
        if self.hits<2 and not (endpoint and self.state=='LISTENING'):
            return None
        self.candidate=None; self.hits=0
        if self.state=='STANDBY':
            self.state='LISTENING'; self.deadline=now+self.timeout
            self.blocked_until=now+.30
            return {'event':'WAKE','prediction':prediction}
        self.state='STANDBY'; self.blocked_until=now+.8
        return {'event':'COMMAND','prediction':prediction}


class StreamEngine:
    def __init__(self,predictor):
        self.predictor=predictor
        m=predictor.meta
        self.gate=WakeState(m['wake_threshold'],m['command_threshold'],m['margin'])
        self.buffer=np.zeros(SAMPLES,dtype=np.float32)
        self.filled=0
        self.latest={}
        self.rms=0
        self.latencies=deque(maxlen=200)
        self.noise_rms=deque(maxlen=40)
        self.utterance=[]
        self.quiet_samples=0
        self.speech_samples=0

    def clear(self):
        self.buffer.fill(0); self.filled=0
        self.utterance=[]; self.quiet_samples=0; self.speech_samples=0

    def feed(self,chunk,now=None):
        now=time.monotonic() if now is None else now
        event=self.gate.tick(now)
        if event:
            self.clear(); return event
        if now<self.gate.blocked_until:
            self.clear(); return None
        chunk=np.asarray(chunk,dtype=np.float32).reshape(-1)
        self.rms=float(np.sqrt(np.mean(chunk**2)+1e-12))
        n=len(chunk)
        if not n:
            return None
        if self.gate.state=='LISTENING':
            # Classify a completed, centered utterance instead of triggering on
            # a high-confidence but unfinished prefix of a longer command.
            floor=float(np.percentile(self.noise_rms,20)) if self.noise_rms else .001
            speech=self.rms>max(.003,2.5*floor)
            if speech or self.utterance:
                self.utterance.append(chunk.copy())
                if speech:
                    self.speech_samples+=n; self.quiet_samples=0
                else: self.quiet_samples+=n
            length=sum(len(x) for x in self.utterance)
            if length>SR*2.1:
                self.clear()
                return {'event':'REJECT','message':'Command too long. Try a shorter phrase.'}
            if self.utterance and self.quiet_samples>=int(.30*SR):
                wave=np.concatenate(self.utterance)
                spoken=self.speech_samples
                self.clear()
                if spoken<int(.15*SR): return None
                active=np.flatnonzero(np.abs(wave)>max(.002,float(np.max(np.abs(wave)))*.04))
                if not len(active): return None
                wave=wave[max(0,active[0]-400):min(len(wave),active[-1]+401)]
                if len(wave)>SAMPLES:
                    return {'event':'REJECT','message':'Command too long. Try a shorter phrase.'}
                self.latest=self.predictor(fit_audio(wave))
                self.latencies.append(self.latest['latency_ms'])
                event=self.gate.accept(self.latest,now,endpoint=True)
                if event: return event
                return {'event':'REJECT','message':'Not confident. Please repeat the command.'}
            return None
        self.noise_rms.append(self.rms)
        if n>=SAMPLES:
            self.buffer[:]=chunk[-SAMPLES:]
        else:
            self.buffer[:-n]=self.buffer[n:]; self.buffer[-n:]=chunk
        self.filled=min(SAMPLES,self.filled+n)
        if self.filled<SAMPLES or np.sqrt(np.mean(self.buffer**2))<0.002:
            return None
        self.latest=self.predictor(self.buffer.copy())
        self.latencies.append(self.latest['latency_ms'])
        event=self.gate.accept(self.latest,now)
        if event:
            self.clear()  # Never reinterpret the wake phrase as a device command.
        return event


def benchmark(path=None,iterations=1000):
    path=Path(path) if path else model_dir()/'model_int8.onnx'
    session=session_for(path); fe=Frontend()
    wave=np.random.default_rng(231).normal(0,.03,SAMPLES).astype(np.float32)
    features=fe(wave)[None]
    for _ in range(20):
        session.run(None,{'features':features})
    model_times=[]; total_times=[]
    for _ in range(iterations):
        start=time.perf_counter(); session.run(None,{'features':features})
        model_times.append((time.perf_counter()-start)*1000)
        start=time.perf_counter(); session.run(None,{'features':fe(wave)[None]})
        total_times.append((time.perf_counter()-start)*1000)
    def stats(values):
        return {k:float(v) for k,v in zip(('p50_ms','p95_ms','p99_ms'),np.percentile(values,[50,95,99]))}
    board=Path('/proc/device-tree/model')
    board_name=board.read_text().strip('\x00') if board.exists() else None
    import psutil
    return dict(platform=platform.platform(),machine=platform.machine(),board=board_name,
        is_raspberry_pi=bool(board_name and 'Raspberry Pi' in board_name),iterations=iterations,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        threads=1,model=stats(model_times),frontend_and_model=stats(total_times),
        process_rss_mb=psutil.Process().memory_info().rss/1024**2,
        note='Process RSS includes currently loaded libraries. The 1.5 s context, 0.15 s hop, wake confirmation and command endpoint wait are additional latency.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=benchmark()
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
