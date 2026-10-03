"""Replay frozen personal/reference features on the Pi; never invoke actions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def class_metrics(expected, predicted, names):
    rows = {}
    for i, name in enumerate(names):
        tp = int(((expected == i) & (predicted == i)).sum())
        support = int((expected == i).sum())
        predicted_n = int((predicted == i).sum())
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / support if support else 0.0
        rows[name] = dict(precision=precision, recall=recall, f1=2*precision*recall/(precision+recall) if precision+recall else 0.0, support=support)
    return dict(accuracy=float((expected == predicted).mean()), count=len(expected), per_class=rows,
                macro_f1=float(np.mean([r['f1'] for r in rows.values() if r['support']])))


def prepare(project, output):
    sys.path.insert(0, str(project))
    from tinyvcm_model.config import LABELS
    from tinyvcm_model.frontend import Frontend
    from tinyvcm_model.personalized_retrain import load_personal_records, load_optionb_test, _feature_matrix, _wake_test_windows
    meta = json.loads((project / 'deployment/current_vcm/metadata.json').read_text(encoding='utf-8'))
    run = project / 'runs' / meta['source_run']
    human, _ = load_personal_records(project, manifest_path=run / 'human_manifest_snapshot.csv')
    held = [r for r in human if r['split'] == 'test']
    frontend = Frontend()
    wx, wg = _wake_test_windows([r for r in held if r['label'] == 'wake_word'], frontend)
    nx, ng = _wake_test_windows([r for r in held if r['label'] != 'wake_word'], frontend)
    px, py = _feature_matrix([r for r in held if r['intent']])
    rx, ry = load_optionb_test(project)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, wake_x=wx, wake_groups=wg, nonwake_x=nx, nonwake_groups=ng,
                        personal_x=px, personal_y=py, reference_x=rx, reference_y=ry,
                        labels=np.array(LABELS), threshold=np.array(meta['wake']['local_operating_threshold']),
                        source_run=np.array(meta['source_run']), wake_sha256=np.array(meta['wake']['model']['sha256']),
                        intent_sha256=np.array(meta['intent']['model']['sha256']))
    print(json.dumps(dict(fixture=str(output), size_bytes=output.stat().st_size, source_run=meta['source_run'])))


def evaluate(fixture, deployment, destination):
    import onnxruntime as ort
    import platform
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    def session(name, expected):
        p = deployment / 'models' / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Model hash mismatch: {name}')
        return ort.InferenceSession(str(p), sess_options=options, providers=['CPUExecutionProvider'])
    def predict(s, features, wake=False):
        result = []
        for x in features:
            logits = s.run(None, {s.get_inputs()[0].name:x[None]})[0][0]
            if wake:
                exp = np.exp(logits - logits.max())
                result.append(float(exp[1]/exp.sum()))
            else:
                result.append(int(logits.argmax()))
        return np.array(result)
    with np.load(fixture, allow_pickle=False) as d:
        wake = session('wake_binary_int8.onnx',str(d['wake_sha256'].item()))
        intent = session('intent_31_int8.onnx',str(d['intent_sha256'].item()))
        threshold = float(d['threshold'])
        ws = predict(wake, d['wake_x'], True)
        ns = predict(wake, d['nonwake_x'], True)
        wp = np.array([ws[d['wake_groups'] == i].max() for i in np.unique(d['wake_groups'])])
        npersonal = np.array([ns[d['nonwake_groups'] == i].max() for i in np.unique(d['nonwake_groups'])])
        nr = predict(wake,d['reference_x'],True)
        result = dict(platform=platform.machine(),source_run=str(d['source_run'].item()),
                      wake_sha256=str(d['wake_sha256'].item()),intent_sha256=str(d['intent_sha256'].item()),
                      protocol='Frozen file-based features; five placements per personal WAV; reused reference test; speaker-overlapping personal holdout; no audio playback or device actions.',
                      wake=dict(threshold=threshold,wake_hits=int((wp>=threshold).sum()),wake_support=len(wp),
                                personal_false_accepts=int((npersonal>=threshold).sum()),personal_negative_support=len(npersonal),
                                reference_false_accepts=int((nr>=threshold).sum()),reference_negative_support=len(nr)),
                      reference_intent=class_metrics(d['reference_y'],predict(intent,d['reference_x']),d['labels'].tolist()),
                      personal_intent=class_metrics(d['personal_y'],predict(intent,d['personal_x']),d['labels'].tolist()))
    destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_intent')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--deployment',type=Path,default=Path('/home/dalmacio/Desktop/dandan'))
    parser.add_argument('--output',type=Path,default=Path('pi_saved_audio_results.json'))
    args=parser.parse_args()
    if args.prepare:
        prepare(Path(__file__).resolve().parents[1],args.fixture)
    else:
        evaluate(args.fixture,args.deployment,args.output)
