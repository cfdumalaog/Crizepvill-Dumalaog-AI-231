from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tinyvcm.frontend import Frontend, fit_audio
from tinyvcm.config import SAMPLES
from tinyvcm.runtime import WakeState, StreamEngine, Predictor, model_dir
from tinyvcm.devices import Devices


def pred(label,confidence=.95,margin=.8):
    return dict(label=label,confidence=confidence,margin=margin,latency_ms=1)


def test_command_cannot_execute_before_wake():
    gate=WakeState()
    for t in np.arange(1,5,.15): assert gate.accept(pred('lights_on'),t) is None
    assert gate.state=='STANDBY'
    assert gate.accept(pred('wake_word'),5) is None
    assert gate.accept(pred('wake_word'),5.15)['event']=='WAKE'
    assert gate.accept(pred('lights_on'),5.2) is None  # feedback holdoff
    assert gate.accept(pred('lights_on'),5.6) is None
    assert gate.accept(pred('lights_on'),5.75)['event']=='COMMAND'
    assert gate.state=='STANDBY'


def test_low_confidence_unknown_and_nonconsecutive_hits_rejected():
    gate=WakeState()
    assert gate.accept(pred('wake_word',.7),1) is None
    assert gate.accept(pred('wake_word',.95,.05),2) is None
    gate.accept(pred('wake_word'),3)
    assert gate.accept(pred('_unknown_'),3.15) is None
    assert gate.accept(pred('wake_word'),3.3) is None
    assert gate.state=='STANDBY'


def test_silent_timeout_and_no_stale_audio_after_wake():
    class Fake:
        meta={'wake_threshold':.8,'command_threshold':.8,'margin':.2}
        def __call__(self,audio): return pred('wake_word')
    engine=StreamEngine(Fake())
    wave=np.ones(2400,dtype=np.float32)*.05
    for i in range(10): assert engine.feed(wave,i*.15) is None
    assert engine.feed(wave,1.5)['event']=='WAKE'
    assert engine.filled==0 and not engine.buffer.any()
    assert engine.feed(np.zeros(2400),9)['event']=='TIMEOUT'
    assert engine.gate.state=='STANDBY'


@pytest.mark.parametrize('kind',['silence','noise','tone'])
def test_frontend_matches_existing_torch_math(kind):
    import torch
    from src.audio import LogMelFrontend
    wave=np.zeros(SAMPLES,dtype=np.float32)
    if kind=='noise': wave=np.random.default_rng(1).normal(0,.05,SAMPLES).astype(np.float32)
    if kind=='tone': wave=np.sin(np.arange(SAMPLES)*2*np.pi*440/16000).astype(np.float32)*.1
    actual=Frontend()(wave)
    with torch.inference_mode(): expected=LogMelFrontend()(torch.from_numpy(wave)[None]).numpy()[0]
    assert actual.shape==(1,40,151) and np.isfinite(actual).all()
    if kind=='silence':
        assert not actual.any()  # Legacy torch normalization amplifies rounding noise on constant input.
    else:
        np.testing.assert_allclose(actual,expected,atol=.005,rtol=.005)


def test_overlong_speech_is_not_silently_truncated():
    with pytest.raises(ValueError): fit_audio(np.zeros(SAMPLES+1))


def test_completed_utterance_dispatches_once_after_silence():
    class Fake:
        meta={'wake_threshold':.8,'command_threshold':.8,'margin':.2}
        def __call__(self,audio): return pred('lights_on')
    engine=StreamEngine(Fake())
    engine.gate.state='LISTENING'; engine.gate.deadline=20
    speech=np.sin(np.arange(2400)*.08).astype(np.float32)*.1
    for i in range(5): assert engine.feed(speech,1+i*.15) is None
    assert engine.feed(np.zeros(2400,dtype=np.float32),2) is None
    event=engine.feed(np.zeros(2400,dtype=np.float32),2.15)
    assert event['event']=='COMMAND' and event['prediction']['label']=='lights_on'
    assert engine.gate.state=='STANDBY'


def test_manifest_provenance_and_disjointness():
    from tinyvcm.data import audit_dataset
    rows,labels,report=audit_dataset()
    assert not report['human_dataset_requirement_met']
    assert len(labels)==26
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not groups['train'] & (groups['val']|groups['test'])
    assert not groups['val'] & groups['test']
    train_speakers={r['speaker'] for r in rows if r['split']=='train' and r['provenance']=='Windows_SAPI_TTS'}
    assert report['test_speaker'] not in train_speakers


def test_real_timer_and_explicit_simulation(monkeypatch):
    monkeypatch.setattr('tinyvcm.devices.time.monotonic',lambda:100.)
    d=Devices()
    d.execute('timer_1min')
    assert d.timer_end==160.
    assert 'No call placed' in d.execute('call_mom')
    assert 'No live weather' in d.execute('question_weather')
    monkeypatch.setattr('tinyvcm.devices.time.monotonic',lambda:159.)
    assert d.tick() is None
    monkeypatch.setattr('tinyvcm.devices.time.monotonic',lambda:160.)
    assert d.tick()=='Timer expired.'


def test_saved_int8_executes_quantized_convolutions(tmp_path):
    import onnx
    import onnxruntime as ort
    from collections import Counter
    path=model_dir()/'model_int8.onnx'
    assert path.stat().st_size<500000
    options=ort.SessionOptions()
    options.intra_op_num_threads=1
    options.optimized_model_filepath=str(tmp_path/'optimized.onnx')
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    out=session.run(None,{'features':np.zeros((1,1,40,151),dtype=np.float32)})[0]
    labels=json.loads((model_dir()/'metadata.json').read_text())['labels']
    assert out.shape==(1,len(labels)) and np.isfinite(out).all()
    graph=onnx.load(options.optimized_model_filepath)
    counts=Counter(n.op_type for n in graph.graph.node)
    assert counts['QLinearConv']==9,counts
    assert counts['QGemm']==1,counts


def test_model_checksum_failure_is_loud(tmp_path):
    import shutil
    src=model_dir()
    shutil.copy(src/'metadata.json',tmp_path/'metadata.json')
    (tmp_path/'model_int8.onnx').write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'): Predictor(tmp_path)
