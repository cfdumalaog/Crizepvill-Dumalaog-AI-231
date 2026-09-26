"""
Training, Evaluation, and INT8 Quantization Export Pipeline for Tiny VCM.
Trains BC-ResNet-1 on the 24-class smart device command dataset,
evaluates noise robustness across SNRs, and exports optimized edge weights.
"""

import os
import time
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    DATA_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    COMMAND_CLASSES,
    IDX_TO_CLASS
)
from .model import build_model, count_parameters, model_size_kb
from .dataset import create_dataloaders, VoiceCommandDataset
from .audio import LogMelFrontend, mix_noise_at_snr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, (correct / total) * 100.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model performance and returns metrics and predictions."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def evaluate_snr_robustness(
    model: nn.Module,
    snr_levels: list = [30.0, 20.0, 10.0, 0.0, -5.0],
    dataset_dir: Path = DATA_DIR / "dataset",
    device: torch.device = torch.device("cpu")
) -> Dict[float, float]:
    """
    Evaluates command classification accuracy across various Signal-to-Noise Ratios
    using physical WAV files mixed with real background noise.
    """
    model.eval()
    frontend = LogMelFrontend().to(device)
    results = {}

    noise_dir = dataset_dir / "_background_noise_"
    noise_files = list(noise_dir.glob("*.wav")) if noise_dir.exists() else []

    for snr in snr_levels:
        correct = 0
        total = 0
        for cls_name in COMMAND_CLASSES:
            if cls_name in ["_silence_", "_background_noise_"]:
                continue
            cls_dir = dataset_dir / cls_name
            if not cls_dir.exists():
                continue
            wav_files = list(cls_dir.glob("*.wav"))[:10]
            cls_idx = COMMAND_CLASSES.index(cls_name)

            for wf in wav_files:
                clean_audio, _ = sf.read(str(wf))
                if clean_audio.ndim > 1:
                    clean_audio = np.mean(clean_audio, axis=1)
                
                if noise_files:
                    noise_audio, _ = sf.read(str(random.choice(noise_files)))
                    if noise_audio.ndim > 1:
                        noise_audio = np.mean(noise_audio, axis=1)
                    noisy = mix_noise_at_snr(clean_audio, noise_audio, snr)
                else:
                    noisy = clean_audio

                audio_tensor = torch.from_numpy(noisy).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    mel = frontend(audio_tensor)
                    logits = model(mel)
                    pred = torch.argmax(logits, dim=1).item()
                if pred == cls_idx:
                    correct += 1
                total += 1

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        results[snr] = acc

    return results


def run_training_pipeline(
    model_name: str = "bc_resnet",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    output_dir: Path = MODELS_DIR
) -> Dict[str, Any]:
    """Full training, benchmarking, quantization, and export pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device} | Architecture: {model_name.upper()}")

    # 1. Instantiate Model
    model = build_model(model_name, num_classes=NUM_CLASSES).to(device)
    params = count_parameters(model)
    fp32_size = model_size_kb(model, 32)
    print(f"[TRAIN] Model Parameters: {params:,} | Estimated FP32 Size: {fp32_size:.1f} KB")

    # 2. Prepare DataLoaders
    print(f"[TRAIN] Loading physical 16kHz WAV dataset from {DATA_DIR / 'dataset'}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=DATA_DIR / "dataset",
        batch_size=batch_size
    )
    print(f"[TRAIN] Samples -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 3. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_checkpoint_path = output_dir / f"{model_name}_best.pt"

    print(f"[TRAIN] Starting {epochs}-epoch training run...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dur = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s] - Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}% | Val Loss: {va_loss:.4f}, Acc: {va_acc:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_checkpoint_path)

    print(f"[TRAIN] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

    # 4. Final Test Set Evaluation
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)
    print(f"[EVAL] Test Accuracy: {test_acc:.2f}% (Loss: {test_loss:.4f})")

    # 5. Noise Robustness Benchmark
    print("[BENCHMARK] Evaluating Noise Robustness across SNR levels...")
    snr_results = evaluate_snr_robustness(model, device=device)
    for snr, acc in snr_results.items():
        print(f"  * SNR {snr:5.1f} dB -> Accuracy: {acc:5.1f}%")

    # 6. INT8 Dynamic Quantization (for Edge deployment on Raspberry Pi CPU)
    print("[EXPORT] Applying INT8 Quantization for Edge CPU deployment...")
    cpu_model = build_model(model_name, num_classes=NUM_CLASSES)
    cpu_model.load_state_dict(torch.load(best_checkpoint_path, map_location="cpu"))
    cpu_model.eval()

    quantized_model = torch.ao.quantization.quantize_dynamic(
        cpu_model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    quantized_path = output_dir / f"{model_name}_int8.pt"
    torch.save(quantized_model.state_dict(), quantized_path)
    int8_size = os.path.getsize(quantized_path) / 1024.0
    print(f"[EXPORT] Quantized Model Saved to {quantized_path} ({int8_size:.1f} KB)")

    # 7. ONNX Export
    onnx_path = output_dir / f"{model_name}.onnx"
    try:
        dummy_input = torch.randn(1, 1, 40, 151)
        torch.onnx.export(
            cpu_model,
            dummy_input,
            onnx_path,
            input_names=["spectrogram"],
            output_names=["logits"],
            dynamic_axes={"spectrogram": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[EXPORT] ONNX model exported to {onnx_path} ({os.path.getsize(onnx_path) / 1024.0:.1f} KB)")
    except Exception as e:
        print(f"[EXPORT] ONNX Export notice: {e}")

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "snr_robustness": snr_results,
        "history": history,
        "params": params,
        "int8_size_kb": int8_size,
        "checkpoint": str(best_checkpoint_path)
    }


if __name__ == "__main__":
    run_training_pipeline()
