"""Reproducible random-start training, held-out evaluation and static INT8 export."""
import copy
import hashlib
import json
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from .config import ROOT, SAMPLES, SR
from .data import audit_dataset, build_features
from .frontend import Frontend
from .model import TinyDSCNN


def prepare(mode='synthetic_baseline', seed=231):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = ROOT/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True)
    rows, labels, audit = audit_dataset(mode, run)
    print(json.dumps(audit, indent=2), flush=True)
    splits = build_features(rows, labels)
    # Noise augmentation uses only TRAIN noise recordings, never test/val audio.
    fe = Frontend()
    x, y, waves = splits['train']
    noise_indices = [i for i,t in enumerate(y) if labels[t]=='_background_noise_']
    rng = np.random.default_rng(seed)
    extra = []
    for wave in waves:
        noise = waves[rng.choice(noise_indices)]
        snr = rng.uniform(10, 30)
        scale = np.sqrt((np.mean(wave**2)+1e-12)/(np.mean(noise**2)+1e-12)/10**(snr/10))
        extra.append(fe(np.clip(wave+noise*scale, -1, 1)))
    splits['augmented'] = (np.concatenate([x, extra]), np.tile(y,2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyDSCNN(len(labels)).to(device)  # random initialization; no checkpoint read
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    info = dict(seed=seed, device=str(device), torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                parameters=sum(p.numel() for p in model.parameters()),
                pretrained_weights_used=False, initial_state_sha256=initial_hash,
                architecture='TinyDSCNN-48', sample_rate=SR, samples=SAMPLES,
                feature_shape=[1,40,151], labels=labels, audit=audit)
    (run/'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Random-start {info["architecture"]}: {info["parameters"]:,} parameters on {device}. Run: {run.name}', flush=True)
    return dict(run=run, rows=rows, labels=labels, audit=audit, splits=splits,
                device=device, model=model, info=info, history=[])


@torch.inference_mode()
def predict(model, x, device, batch_size=128):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i+batch_size]).to(device)).cpu().numpy()
                           for i in range(0,len(x),batch_size)])


def train(ctx, epochs=100, batch_size=64):
    model, device = ctx['model'], ctx['device']
    x_np,y_np = ctx['splits']['augmented']
    x,y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
    vx,vy,_ = ctx['splits']['val']
    counts = np.bincount(y_np, minlength=len(ctx['labels']))
    weights = torch.tensor(np.sqrt(len(y_np)/np.maximum(counts,1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00003)
    best = -1.0
    for epoch in range(1,epochs+1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(y), device=device)
        total_loss = correct = 0
        for i in range(0,len(y),batch_size):
            ids = order[i:i+batch_size]
            xb = x[ids].clone()
            # Time/frequency masking is train-only and does not generate new speakers.
            f = random.randrange(0,37)
            t = random.randrange(0,141)
            xb[:,:,f:f+3,:] = 0
            xb[:,:,:,t:t+10] = 0
            xb = torch.roll(xb, random.randint(-8,8), dims=-1)
            logits = model(xb)
            loss = loss_fn(logits,y[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(ids)
            correct += (logits.argmax(1)==y[ids]).sum().item()
        val_logits = predict(model,vx,device)
        val_loss = nn.functional.cross_entropy(torch.from_numpy(val_logits),torch.from_numpy(vy)).item()
        val_acc = float(np.mean(val_logits.argmax(1)==vy))
        row = dict(epoch=epoch,train_loss=total_loss/len(y),train_accuracy=correct/len(y),
                   val_loss=val_loss,val_accuracy=val_acc,seconds=time.perf_counter()-started)
        ctx['history'].append(row)
        scheduler.step()
        if val_acc > best:
            best = val_acc
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ctx['run']/'best_fp32.pt')
            ctx['best_epoch'] = epoch
        (ctx['run']/'history.json').write_text(json.dumps(ctx['history'],indent=2),encoding='utf-8')
        print(f'Epoch {epoch:02}/{epochs} | loss {row["train_loss"]:.4f} | train {row["train_accuracy"]:.1%} | val {val_acc:.1%} | {row["seconds"]:.1f}s',flush=True)
    model.load_state_dict(torch.load(ctx['run']/'best_fp32.pt',map_location=device,weights_only=True))
    print(f'Best validation checkpoint: epoch {ctx["best_epoch"]}, {best:.2%}',flush=True)
    return ctx['history']


def ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def export_and_evaluate(ctx):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, quantize_static
    run,labels = ctx['run'],ctx['labels']
    model = copy.deepcopy(ctx['model']).cpu().eval()
    fp32_path, int8_path = run/'model_fp32.onnx',run/'model_int8.onnx'
    torch.onnx.export(model, torch.zeros(1,1,40,151), str(fp32_path), dynamo=False,
        input_names=['features'],output_names=['logits'],opset_version=17,
        do_constant_folding=True,external_data=False)
    class Reader(CalibrationDataReader):
        def __init__(self):
            x,y,_ = ctx['splits']['train']
            ids=[]
            for label in range(len(labels)):
                ids.extend(np.flatnonzero(y==label)[:12].tolist())
            self.items=iter(x[i:i+1] for i in ids)
        def get_next(self):
            x=next(self.items,None)
            return {'features':x} if x is not None else None
    quantize_static(str(fp32_path),str(int8_path),Reader(),quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,
        op_types_to_quantize=['Conv','Gemm'],extra_options={'ActivationSymmetric':True})
    graph = onnx.load(str(int8_path))
    onnx.checker.check_model(graph)
    initializers={t.name:t for t in graph.graph.initializer}
    producers={o:n for n in graph.graph.node for o in n.output}
    quantized_layers=[]
    for node in graph.graph.node:
        if node.op_type in ('Conv','Gemm'):
            dq=producers.get(node.input[1])
            if dq is None or dq.op_type!='DequantizeLinear' or initializers[dq.input[0]].data_type!=onnx.TensorProto.INT8:
                raise AssertionError(f'Non-INT8 learned layer: {node.name}')
            quantized_layers.append(node.name)
    assert int8_path.stat().st_size < 500000, 'Actual serialized model exceeds 500 KB'
    assert len(quantized_layers)==10, 'Expected nine convolutions and one classifier'
    session=ort_session(int8_path)
    tx,ty,tw=ctx['splits']['test']
    float_logits=predict(model,tx,'cpu')
    q_logits=np.concatenate([session.run(None,{'features':x[None]})[0] for x in tx])
    preds=q_logits.argmax(1)
    metrics=dict(fp32_test_accuracy=float(np.mean(float_logits.argmax(1)==ty)),
        int8_test_accuracy=float(np.mean(preds==ty)),test_count=len(ty),best_epoch=ctx['best_epoch'],
        int8_bytes=int8_path.stat().st_size,all_conv_and_dense_weights_int8=True,
        quantized_learned_layers=len(quantized_layers),
        prediction_agreement=float(np.mean(preds==float_logits.argmax(1))),
        classification_report=classification_report(ty,preds,target_names=labels,output_dict=True,zero_division=0),
        confusion_matrix=confusion_matrix(ty,preds,labels=np.arange(len(labels))).tolist(),
        raspberry_pi_measured=False,human_speech_measured=ctx['audit']['human_dataset_requirement_met'])
    # Held-out speech mixed with HELD-OUT noise, and deterministic SNR conditions.
    fe=Frontend()
    noise=tw[ty==labels.index('_background_noise_')]
    speech=np.flatnonzero([not labels[t].startswith('_') for t in ty])
    metrics['synthetic_noise_snr_accuracy']={}
    for snr in (20,10,0):
        correct=0
        for j,i in enumerate(speech):
            w=tw[i]; n=noise[j%len(noise)]
            scale=np.sqrt((np.mean(w*w)+1e-12)/(np.mean(n*n)+1e-12)/10**(snr/10))
            feat=fe(np.clip(w+n*scale,-1,1))[None]
            correct+=int(session.run(None,{'features':feat})[0].argmax()==ty[i])
        metrics['synthetic_noise_snr_accuracy'][str(snr)]=correct/len(speech)
    from .runtime import benchmark
    metrics['pc_benchmark']=benchmark(int8_path,iterations=200)
    metadata={**ctx['info'], 'created_at':datetime.now().isoformat(),
        'int8_sha256':hashlib.sha256(int8_path.read_bytes()).hexdigest(),
        'wake_threshold':0.80,'command_threshold':0.80,'margin':0.20,
        'thresholds_validated_on_humans':False,
        'model_file':'model_int8.onnx','metrics':metrics}
    (run/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (ROOT/'runs/latest.json').write_text(json.dumps({'run':run.name}),encoding='utf-8')
    ctx['metrics']=metrics
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix')},indent=2),flush=True)
    return metrics
