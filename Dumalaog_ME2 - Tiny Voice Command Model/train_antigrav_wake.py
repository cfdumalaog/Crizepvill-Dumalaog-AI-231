"""Train 32-class TinyDSCNN-48 Antigrav (31 Option B commands + WAKE_WORD) and deploy to Pi 5."""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

PROJECT_ROOT = Path(r'C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME2 - Tiny Voice Command Model')
sys.path.insert(0, str(PROJECT_ROOT))

from tinyvcm_antigrav.config import LABELS, SAMPLES, SR, MELS, TIME_STEPS, CHANNELS, CLASSES
from tinyvcm_antigrav.frontend import Frontend, fit_audio
from tinyvcm_antigrav.model import TinyDSCNN
from tinyvcm_antigrav.export import export_models

LABEL_MAP = {
    'lights_on': 'LIGHT_ON',
    'lights_off': 'LIGHT_OFF',
    'question_weather': 'WEATHER',
    'question_time': 'TIME',
    'volume_up': 'VOLUME_UP',
    'volume_down': 'VOLUME_DOWN',
    'play_music': 'PLAY_MUSIC',
    'media_pause': 'PAUSE',
    'media_resume': 'PLAY_MUSIC',
    'media_next': 'NEXT',
    'timer_1min': 'TIMER_1m',
    'timer_5min': 'TIMER_1m',
    'timer_10min': 'TIMER_10s',
    'alarm_set': 'ALARM_8_00AM'
}

def load_data():
    frontend = Frontend()
    manifest_path = PROJECT_ROOT / 'data' / 'human' / 'manifest.csv'

    # 1. User command takes
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    user_cmd_features = []
    user_cmd_labels = []
    user_wake_features = []

    for r in rows:
        lbl = r['label']
        wav_path = PROJECT_ROOT / r['path']
        if not wav_path.exists():
            continue
        wav, sr = sf.read(wav_path, dtype='float32')
        fitted = fit_audio(wav, target_samples=SAMPLES)
        spec = frontend(fitted)

        if lbl == 'wake_word':
            user_wake_features.append(spec)
        elif lbl in LABEL_MAP:
            target_label = LABEL_MAP[lbl]
            target_idx = LABELS.index(target_label)
            user_cmd_features.append(spec)
            user_cmd_labels.append(target_idx)

    print(f"Loaded {len(user_wake_features)} user wake word takes and {len(user_cmd_features)} user command takes.")

    # 2. Base wake word files
    wake_files = list((PROJECT_ROOT / 'data/dataset/wake_word').glob('*.wav'))
    base_wake_features = []
    for f in wake_files:
        wav, sr = sf.read(f, dtype='float32')
        fitted = fit_audio(wav, target_samples=SAMPLES)
        base_wake_features.append(frontend(fitted))

    print(f"Loaded {len(base_wake_features)} base wake word takes.")

    wake_idx = LABELS.index('WAKE_WORD')
    assert wake_idx == 31, f"Expected WAKE_WORD to be class 31, got {wake_idx}"

    # Package wake word features
    all_wake_x = np.stack(base_wake_features + user_wake_features, axis=0).astype(np.float32)
    all_wake_y = np.full(len(all_wake_x), wake_idx, dtype=np.int64)

    # User wake takes repeated 8x for strong user pitch adaptation
    user_wake_x_rep = np.repeat(np.stack(user_wake_features, axis=0), 8, axis=0).astype(np.float32)
    user_wake_y_rep = np.full(len(user_wake_x_rep), wake_idx, dtype=np.int64)

    # User command takes repeated 5x
    user_cmd_x = np.stack(user_cmd_features, axis=0).astype(np.float32)
    user_cmd_y = np.array(user_cmd_labels, dtype=np.int64)
    user_cmd_x_rep = np.repeat(user_cmd_x, 5, axis=0)
    user_cmd_y_rep = np.repeat(user_cmd_y, 5, axis=0)

    # 3. Base Option B Cache (17,986 audio clips across 31 classes)
    cache_path = PROJECT_ROOT / 'data' / 'option_b' / 'features_cache.npz'
    cache = np.load(cache_path)
    train_x = cache['train_x'].astype(np.float32)
    train_y = cache['train_y'].astype(np.int64)
    val_x = cache['val_x'].astype(np.float32)
    val_y = cache['val_y'].astype(np.int64)
    test_x = cache['test_x'].astype(np.float32)
    test_y = cache['test_y'].astype(np.int64)

    # Split wake words: 80% train, 20% val
    rng = np.random.default_rng(231)
    perm = rng.permutation(len(all_wake_x))
    split_w = int(len(all_wake_x) * 0.8)
    wake_train_x, wake_val_x = all_wake_x[perm[:split_w]], all_wake_x[perm[split_w:]]
    wake_train_y, wake_val_y = all_wake_y[perm[:split_w]], all_wake_y[perm[split_w:]]

    # Combine into augmented splits
    full_train_x = np.concatenate([train_x, wake_train_x, user_wake_x_rep, user_cmd_x_rep], axis=0)
    full_train_y = np.concatenate([train_y, wake_train_y, user_wake_y_rep, user_cmd_y_rep], axis=0)

    full_val_x = np.concatenate([val_x, wake_val_x], axis=0)
    full_val_y = np.concatenate([val_y, wake_val_y], axis=0)

    # For testing, append wake validation samples
    full_test_x = np.concatenate([test_x, wake_val_x], axis=0)
    full_test_y = np.concatenate([test_y, wake_val_y], axis=0)

    print(f"Augmented Train: {len(full_train_y):,} samples across {len(LABELS)} classes.")
    print(f"Augmented Val:   {len(full_val_y):,} samples.")
    print(f"Augmented Test:  {len(full_test_y):,} samples.")

    user_data = {
        'wake_x': np.stack(user_wake_features, axis=0).astype(np.float32),
        'wake_y': np.full(len(user_wake_features), wake_idx, dtype=np.int64),
        'cmd_x': user_cmd_x,
        'cmd_y': user_cmd_y
    }

    return full_train_x, full_train_y, full_val_x, full_val_y, full_test_x, full_test_y, user_data

def main():
    print("=" * 70)
    print("  TRAINING 32-CLASS TINYDSCNN-48 ANTIGRAV WITH WAKE WORD GATING")
    print("=" * 70)

    train_x, train_y, val_x, val_y, test_x, test_y, user_data = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Instantiate 32-class TinyDSCNN-48
    model = TinyDSCNN(classes=len(LABELS), channels=CHANNELS).to(device)

    # Transfer 31-class backbone weights
    checkpoint_path = PROJECT_ROOT / 'runs' / 'antigrav-20260927-115145' / 'best_model.pt'
    if checkpoint_path.exists():
        print(f"Loading pretrained backbone from {checkpoint_path.name}...")
        old_state = torch.load(checkpoint_path, map_location=device)
        model_state = model.state_dict()
        for k, v in old_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k].copy_(v)
            elif k == 'classifier.1.weight':
                model_state[k][:31].copy_(v)
                model_state[k][31].copy_(v.mean(dim=0) + torch.randn(48, device=device) * 0.05)
            elif k == 'classifier.1.bias':
                model_state[k][:31].copy_(v)
                model_state[k][31] = 0.0
        model.load_state_dict(model_state)
        print("Backbone and classification head initialized successfully.")

    run_dir = PROJECT_ROOT / 'runs' / f"antigrav-wake-32class-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    epochs = 20
    batch_size = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    gpu_train_x = torch.from_numpy(train_x).to(device)
    gpu_train_y = torch.from_numpy(train_y).to(device)

    best_val_acc = -1.0
    best_weights = None

    print(f"\nTraining 32-class TinyDSCNN-48 for {epochs} epochs on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(gpu_train_y), device=device)
        running_loss = 0.0
        running_correct = 0

        for i in range(0, len(gpu_train_y), batch_size):
            ids = order[i:i + batch_size]
            xb = gpu_train_x[ids].clone()
            yb = gpu_train_y[ids]

            # SpecAugment
            f_start = np.random.randint(0, 36)
            t_start = np.random.randint(0, 235)
            xb[:, :, f_start:f_start + 4, :] = 0.0
            xb[:, :, :, t_start:t_start + 12] = 0.0

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(ids)
            running_correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        train_acc = running_correct / len(gpu_train_y)

        # Validation
        model.eval()
        with torch.no_grad():
            val_xb = torch.from_numpy(val_x).to(device)
            val_yb = torch.from_numpy(val_y).to(device)
            val_logits = model(val_xb)
            val_acc = (val_logits.argmax(1) == val_yb).float().mean().item()

            # Check user wake word recognition
            uw_xb = torch.from_numpy(user_data['wake_x']).to(device)
            uw_yb = torch.from_numpy(user_data['wake_y']).to(device)
            uw_logits = model(uw_xb)
            uw_acc = (uw_logits.argmax(1) == uw_yb).float().mean().item()

            # Check user command recognition
            uc_xb = torch.from_numpy(user_data['cmd_x']).to(device)
            uc_yb = torch.from_numpy(user_data['cmd_y']).to(device)
            uc_logits = model(uc_xb)
            uc_acc = (uc_logits.argmax(1) == uc_yb).float().mean().item()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, run_dir / 'best_model.pt')

        best_mark = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:2d}/{epochs} | Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | User Wake: {uw_acc*100:5.1f}% | User Cmd: {uc_acc*100:5.1f}%{best_mark}", flush=True)

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_xb = torch.from_numpy(test_x).to(device)
        test_yb = torch.from_numpy(test_y).to(device)
        test_acc = (model(test_xb).argmax(1) == test_yb).float().mean().item()

        final_uw = (model(torch.from_numpy(user_data['wake_x']).to(device)).argmax(1) == torch.from_numpy(user_data['wake_y']).to(device)).float().mean().item()
        final_uc = (model(torch.from_numpy(user_data['cmd_x']).to(device)).argmax(1) == torch.from_numpy(user_data['cmd_y']).to(device)).float().mean().item()

    print("\n" + "=" * 70)
    print("  32-CLASS MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Held-Out Test Accuracy (Option B 10 Unseen Speakers + Wake): {test_acc*100:.2f}%")
    print(f"User 'Hi Dandan' Wake Word Accuracy:                          {final_uw*100:.2f}% ({int(final_uw*len(user_data['wake_y']))}/{len(user_data['wake_y'])} correct)")
    print(f"User Voice Command Accuracy:                                  {final_uc*100:.2f}% ({int(final_uc*len(user_data['cmd_y']))}/{len(user_data['cmd_y'])} correct)")

    # Export ONNX
    print("\n" + "=" * 70)
    print("  EXPORTING STATIC INT8 ONNX MODEL (32 CLASSES)")
    print("=" * 70)
    export_summary = export_models(run_dir, val_x, test_x, test_y)

    models_dest = PROJECT_ROOT / 'models'
    src_int8 = run_dir / 'models' / 'antigrav_optionb_int8.onnx'
    src_fp32 = run_dir / 'models' / 'antigrav_optionb_fp32.onnx'

    shutil.copy2(src_int8, models_dest / 'antigrav_optionb_int8.onnx')
    shutil.copy2(src_fp32, models_dest / 'antigrav_optionb_fp32.onnx')
    print(f"Exported updated 32-class ONNX models to {models_dest / 'antigrav_optionb_int8.onnx'} ({src_int8.stat().st_size:,} bytes)")

    print("\nTraining and export completed successfully!")

if __name__ == '__main__':
    main()
