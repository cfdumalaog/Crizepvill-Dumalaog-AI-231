"""Bounded audio queue: PortAudio callback never runs inference or device actions."""
import math
import queue
import time
import numpy as np
from scipy.signal import resample_poly
import sounddevice as sd
from .config import SR


class Microphone:
    def __init__(self,device=None,hop=.15):
        self.requested_device=device; self.device=device; self.hop=hop; self.stream=None
        self.queue=queue.Queue(maxsize=24)
        self.dropped=0; self.status=''; self.name='not started'
        self.blocks_received=0; self.last_callback_time=0.0
        self.pending=np.zeros(0,dtype=np.float32)
        self.rate=SR

    def _callback(self,data,frames,info,status):
        self.blocks_received+=1
        self.last_callback_time=time.monotonic()
        if status: self.status=str(status)
        try: self.queue.put_nowait(data.copy())
        except queue.Full: self.dropped+=1

    def start(self):
        devices=sd.query_devices(); hosts=sd.query_hostapis()
        default_input=sd.default.device[0]
        candidates=[]
        for i,d in enumerate(devices):
            if not d['max_input_channels']: continue
            requested=self.requested_device
            selected=False
            if requested is not None:
                if isinstance(requested, int) or (isinstance(requested, str) and requested.isdecimal()):
                    selected=i == int(requested)
                else:
                    selected=str(requested).casefold() in d['name'].casefold()
            name=d['name'].lower(); host=hosts[d['hostapi']]['name']
            score=10000*selected+1000*(i==default_input)+20*('usb' in name or 'fifine' in name)+10*('mic' in name)+8*('WASAPI' in host)-20*('WDM-KS' in host)
            score-=50*any(s in name for s in ('stereo mix','loopback','virtual','speaker'))
            candidates.append((score,i,d,selected))
        errors=[]
        for _,i,d,selected in sorted(candidates,reverse=True):
            for ch in dict.fromkeys([1,int(d['max_input_channels']),min(2,int(d['max_input_channels']))]):
                stream=None
                try:
                    rate=int(d['default_samplerate'])
                    stream=sd.InputStream(device=i,samplerate=rate,channels=ch,dtype='float32',
                        blocksize=int(rate*.05),callback=self._callback)
                    stream.start()
                    first=self.queue.get(timeout=1)
                    self.stream=stream; self.rate=rate; self.device=i
                    self.name=f'[{i}] {d["name"]} / {hosts[d["hostapi"]]["name"]} / {rate} Hz / {ch} ch'
                    self.pending=first.mean(axis=1)
                    self.status = '' if self.requested_device is None or selected else 'Preferred input unavailable; using an available microphone.'
                    return
                except Exception as exc:
                    errors.append(f'device {i}, {ch} ch: {exc}')
                    if stream is not None:
                        try: stream.close()
                        except Exception: pass
                    while not self.queue.empty(): self.queue.get_nowait()
        if not candidates:
            raise RuntimeError(f'No input device matches {self.requested_device!r}. Run --list-input-devices to inspect available inputs.')
        raise RuntimeError('No usable microphone. '+ '; '.join(errors[-4:]))

    def read(self,timeout=.3):
        needed=round(self.rate*self.hop)
        while len(self.pending)<needed:
            try: data=self.queue.get(timeout=timeout)
            except queue.Empty: return None
            self.pending=np.concatenate([self.pending,data.mean(axis=1)])
        raw,self.pending=self.pending[:needed],self.pending[needed:]
        div=math.gcd(self.rate,SR)
        return resample_poly(raw,SR//div,self.rate//div).astype(np.float32) if self.rate!=SR else raw

    def close(self):
        if self.stream:
            stream,self.stream=self.stream,None
            try: stream.stop()
            except Exception: pass
            try: stream.close()
            except Exception: pass
        self.pending=np.zeros(0,dtype=np.float32)
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break


if __name__=='__main__':
    mic=Microphone()
    try:
        mic.start(); start=time.monotonic(); samples=0; peak=0.
        while time.monotonic()-start<3:
            chunk=mic.read()
            if chunk is not None: samples+=len(chunk); peak=max(peak,float(np.max(np.abs(chunk))))
        print({'device':mic.name,'resampled_samples':samples,'peak':peak,'dropped_blocks':mic.dropped})
    finally: mic.close()
