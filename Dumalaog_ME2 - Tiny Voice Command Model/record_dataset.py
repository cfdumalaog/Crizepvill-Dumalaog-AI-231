"""Local human recording workflow. Never labels synthesized audio as a person."""
import csv
import math
from pathlib import Path
import re
import uuid
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import gradio as gr
from tinyvcm.config import ROOT, LABELS, PHRASES, SR, SAMPLES
from tinyvcm.frontend import fit_audio


def save_recording(audio,speaker,condition,label,consent):
    if not consent: raise gr.Error('Confirm the speaker agreed to local recording for this class project.')
    if not re.fullmatch(r'[A-Za-z0-9-]{2,30}',speaker or ''): raise gr.Error('Use an anonymous speaker ID, e.g. person-01.')
    if condition not in ('quiet-near','fan-near','quiet-far'): raise gr.Error('Choose a recording condition.')
    if label not in PHRASES: raise gr.Error('Choose a label.')
    if audio is None: raise gr.Error('Record an utterance first.')
    rate,wave=audio
    if np.issubdtype(wave.dtype,np.integer): wave=wave.astype(np.float32)/np.iinfo(wave.dtype).max
    wave=np.asarray(wave,dtype=np.float32)
    if wave.ndim==2: wave=wave.mean(axis=1)
    if not len(wave) or not np.isfinite(wave).all(): raise gr.Error('Empty or invalid audio.')
    if np.mean(np.abs(wave)>.99)>.001: raise gr.Error('Audio is clipping; move back or reduce microphone gain and record again.')
    div=math.gcd(rate,SR); wave=resample_poly(wave,SR//div,rate//div)
    if not label.startswith('_'):
        threshold=max(.004,float(np.max(np.abs(wave)))*.04)
        active=np.flatnonzero(np.abs(wave)>threshold)
        if not len(active): raise gr.Error('No audible speech found.')
        wave=wave[max(0,active[0]-800):min(len(wave),active[-1]+801)]
        if len(wave)>SAMPLES: raise gr.Error('Speech is longer than 1.5 seconds. Use a shorter phrase; words will not be cut off.')
    else:
        if len(wave)<SAMPLES: raise gr.Error('Record at least 1.5 seconds for background or unrelated speech.')
        wave=wave[:SAMPLES]
    wave=fit_audio(wave)
    source=uuid.uuid4().hex
    directory=ROOT/'data/human'/speaker/condition/label
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(source+'.wav'); sf.write(path,wave,SR,subtype='PCM_16')
    manifest=ROOT/'data/human/manifest.csv'
    fields=['path','speaker','condition','label','source_id']
    exists=manifest.exists()
    with manifest.open('a',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        if not exists: writer.writeheader()
        writer.writerow(dict(path=path.relative_to(ROOT).as_posix(),speaker=speaker,condition=condition,label=label,source_id=source))
    with manifest.open(newline='',encoding='utf-8') as file: rows=list(csv.DictReader(file))
    count=sum(r['speaker']==speaker and r['condition']==condition and r['label']==label for r in rows)
    return f'Saved {count} take(s) for {speaker} / {condition} / {label}. Total dataset: {len(rows)} recordings.',(SR,wave)


def main():
    with gr.Blocks(title='Tiny VCM — human dataset recorder') as app:
        gr.Markdown('# Record your Tiny VCM dataset\nUse at least **three real people**, anonymous IDs, and multiple conditions. Aim for 10 repetitions per phrase per condition. Keep a complete speaker aside for testing. Record wake phrases, unrelated speech, near misses and room noise too. Stop or mute the assistant while recording. Files remain on this PC.')
        with gr.Row():
            speaker=gr.Textbox(label='Anonymous speaker ID',value='person-01')
            condition=gr.Dropdown(['quiet-near','fan-near','quiet-far'],value='quiet-near',label='Condition')
            label=gr.Dropdown(LABELS+['_unknown_'],value='wake_word',label='Class')
        prompt=gr.Textbox(value=PHRASES['wake_word'],label='Suggested phrase (speak naturally)',interactive=False)
        label.change(lambda x:PHRASES[x],label,prompt)
        consent=gr.Checkbox(label='This real speaker agrees to these local course-project recordings.')
        audio=gr.Audio(sources=['microphone'],type='numpy',label='Record then listen before saving')
        save=gr.Button('Save this take',variant='primary')
        result=gr.Textbox(label='Saved recording status'); playback=gr.Audio(label='Saved 16 kHz WAV')
        save.click(save_recording,[audio,speaker,condition,label,consent],[result,playback],concurrency_limit=1)
    app.launch(server_name='127.0.0.1',server_port=7862,share=False,inbrowser=False)


if __name__=='__main__': main()
