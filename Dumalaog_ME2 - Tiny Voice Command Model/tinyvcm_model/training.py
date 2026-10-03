"""From-scratch training pipeline for TinyDSCNN-48 VCM on Option B dataset."""
import copy
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASSES, LABELS, ROOT, SAMPLES, SR, TIME_STEPS, MELS, CHANNELS
from .data import audit_dataset, build_features
from .model import TinyDSCNN


def prepare(seed=231):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    run_dir = ROOT / 'runs' / f"vcm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows, labels, audit = audit_dataset(run_dir)
    print("Dataset audit completed:\n" + json.dumps({k: v for k, v in audit.items() if k != 'class_distribution'}, indent=2), flush=True)

    splits = build_features(rows, labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyDSCNN(classes=len(labels), channels=CHANNELS).to(device)

    # Initial random weights SHA-256 hash to prove from-scratch initialization
    initial_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

    run_info = {
        "model_name": "TinyDSCNN-48 VCM",
        "dataset": "Option B (100 human speakers, 17,986 audio clips)",
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "pretrained_weights_used": False,
        "initial_state_sha256": initial_hash,
        "classes_count": len(labels),
        "labels": labels,
        "sample_rate": SR,
        "window_duration_sec": 2.5,
        "samples": SAMPLES,
        "feature_shape": [1, MELS, TIME_STEPS],
        "created_at": datetime.now().isoformat()
    }
    (run_dir / 'run_info.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    print(f"Initialized {run_info['model_name']} ({run_info['parameters']:,} parameters) on {device}.", flush=True)

    return {
        "run_dir": run_dir,
        "rows": rows,
        "labels": labels,
        "audit": audit,
        "splits": splits,
        "device": device,
        "model": model,
        "info": run_info
    }


@torch.inference_mode()
def evaluate(model, features, targets, device, batch_size=128):
    model.eval()
    all_preds = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    n = len(targets)

    for i in range(0, n, batch_size):
        xb = torch.from_numpy(features[i:i + batch_size]).to(device)
        yb = torch.from_numpy(targets[i:i + batch_size]).to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(xb)
        all_preds.append(logits.argmax(1).cpu().numpy())

    preds = np.concatenate(all_preds)
    accuracy = float((preds == targets).mean())
    avg_loss = total_loss / n
    return accuracy, avg_loss, preds


def train(ctx, epochs=35, batch_size=64, lr=0.003):
    model = ctx['model']
    device = ctx['device']
    run_dir = ctx['run_dir']
    labels = ctx['labels']

    train_x_np, train_y_np, _ = ctx['splits']['train']
    val_x_np, val_y_np, _ = ctx['splits']['val']
    test_x_np, test_y_np, _ = ctx['splits']['test']

    # Send features directly to GPU memory for zero-overhead training
    print(f"Loading {len(train_y_np):,} training feature tensors into {device} memory...", flush=True)
    train_x = torch.from_numpy(train_x_np).to(device)
    train_y = torch.from_numpy(train_y_np).to(device)

    # Class balancing weights
    counts = np.bincount(train_y_np, minlength=len(labels))
    class_weights = torch.tensor(np.sqrt(len(train_y_np) / np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)

    best_val_acc = -1.0
    best_weights = None
    best_epoch = 0
    history = []

    print(f"\nStarting {epochs}-epoch training on {device} ({len(train_y):,} training samples, {len(val_y_np):,} val samples)...", flush=True)
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_y), device=device)
        running_loss = 0.0
        running_correct = 0

        for i in range(0, len(train_y), batch_size):
            ids = order[i:i + batch_size]
            xb = train_x[ids].clone()
            yb = train_y[ids]

            # SpecAugment: Random frequency & time masking + temporal jitter
            f_start = random.randrange(0, 36)
            t_start = random.randrange(0, 235)
            xb[:, :, f_start:f_start + 4, :] = 0.0
            xb[:, :, :, t_start:t_start + 15] = 0.0
            xb = torch.roll(xb, random.randint(-10, 10), dims=-1)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(ids)
            running_correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc = running_correct / len(train_y)
        train_loss = running_loss / len(train_y)
        epoch_time = time.perf_counter() - epoch_start

        # Validation evaluation
        val_acc, val_loss, _ = evaluate(model, val_x_np, val_y_np, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, run_dir / 'best_model.pt')

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "epoch_time_sec": round(epoch_time, 2),
            "lr": round(optimizer.param_groups[0]['lr'], 6)
        })

        best_mark = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% ({epoch_time:.2f}s){best_mark}", flush=True)

    total_training_time = time.perf_counter() - total_start
    print(f"\nTraining finished in {total_training_time:.1f}s ({total_training_time/60:.2f} min). Best Val Acc: {best_val_acc*100:.2f}% at epoch {best_epoch}.", flush=True)

    # Load best weights for final evaluation on test split (10 held-out human speakers)
    model.load_state_dict(best_weights)
    test_acc, test_loss, test_preds = evaluate(model, test_x_np, test_y_np, device)
    print(f"Final Held-Out Test Accuracy (10 unseen human speakers): {test_acc*100:.2f}% (Loss: {test_loss:.4f})", flush=True)

    report_text = classification_report(test_y_np, test_preds, target_names=labels, digits=4)
    report_dict = classification_report(test_y_np, test_preds, target_names=labels, output_dict=True)
    conf_mat = confusion_matrix(test_y_np, test_preds).tolist()

    (run_dir / 'classification_report.txt').write_text(report_text, encoding='utf-8')
    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "total_training_time_sec": round(total_training_time, 2),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat
    }
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (run_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "history": history,
        "run_dir": run_dir
    }
