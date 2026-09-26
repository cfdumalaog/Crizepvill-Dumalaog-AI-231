"""Replay held-out WAVs through the real INT8 streaming model; no fake predictions."""
from pathlib import Path
import json
import sys
import numpy as np
import soundfile as sf
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from tinyvcm.runtime import Predictor, StreamEngine, model_dir

predictor=Predictor()
run=model_dir()
rows=json.loads((run/'split_manifest.json').read_text())
human=predictor.meta['audit']['human_dataset_requirement_met']
held=[r for r in rows if r['split']=='test' and (human or r['path'].endswith('_clean.wav'))]
wakes=[r for r in held if r['label']=='wake_word']
commands=[r for r in held if not r['label'].startswith('_') and r['label']!='wake_word']
if not wakes or not commands:
    raise ValueError('Replay needs both held-out wake and command recordings; refusing an empty evaluation.')
results=[]
for i,row in enumerate(commands):
    engine=StreamEngine(predictor)
    wake=sf.read(root/wakes[i%len(wakes)]['path'],dtype='float32')[0]
    command=sf.read(root/row['path'],dtype='float32')[0]
    audio=np.concatenate([np.zeros(16000),wake,np.zeros(12800),command,np.zeros(16000)]).astype(np.float32)
    events=[]
    for start in range(0,len(audio),2400):
        chunk=audio[start:start+2400]
        event=engine.feed(chunk,now=start/16000)
        if event: events.append(event)
    executed=[e['prediction']['label'] for e in events if e['event']=='COMMAND']
    results.append(dict(path=row['path'],expected=row['label'],wake=any(e['event']=='WAKE' for e in events),
        executed=executed,success=executed==[row['label']]))
report=dict(scope=('held-out human file replay; not a live microphone test' if human else
                   'held-out synthetic clean-utterance replay; not a microphone or human test'),
    trials=len(results),wake_detected=sum(r['wake'] for r in results),
    correct_command_sequences=sum(r['success'] for r in results),details=results)
(run/'stream_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
