"""MNIST data, training, evaluation, and visualization for ME1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST

from manual_cnn import ManualThreeLayerCNN, manual_cross_entropy


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass(frozen=True)
class MNISTTensors:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor


def set_seed(seed: int = 42) -> None:
    # cuBLAS needs this workspace policy before its first CUDA operation when
    # deterministic algorithms are requested.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_images(raw: torch.Tensor) -> torch.Tensor:
    images = raw.to(dtype=torch.float32).unsqueeze(1) / 255.0
    return (images - MNIST_MEAN) / MNIST_STD


def load_mnist(
    data_dir: str | Path,
    *,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> MNISTTensors:
    """Download/read official MNIST and return normalized BCHW tensors."""

    data_dir = Path(data_dir)
    training = MNIST(root=data_dir, train=True, download=True)
    testing = MNIST(root=data_dir, train=False, download=True)

    train_images = _prepare_images(training.data)
    train_labels = training.targets.to(dtype=torch.long)
    test_images = _prepare_images(testing.data)
    test_labels = testing.targets.to(dtype=torch.long)

    if limit_train is not None:
        train_images = train_images[:limit_train]
        train_labels = train_labels[:limit_train]
    if limit_test is not None:
        test_images = test_images[:limit_test]
        test_labels = test_labels[:limit_test]

    return MNISTTensors(train_images, train_labels, test_images, test_labels)


def _batch_indices(
    sample_count: int,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    order = (
        torch.randperm(sample_count, generator=generator)
        if shuffle
        else torch.arange(sample_count)
    )
    return list(order.split(batch_size))


@torch.no_grad()
def evaluate_model(
    model: ManualThreeLayerCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 512,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    total_loss = 0.0
    total_correct = 0
    predictions: list[torch.Tensor] = []

    for indices in _batch_indices(
        len(images), batch_size, shuffle=False, generator=generator
    ):
        batch_images = images[indices].to(model.device, non_blocking=True)
        batch_labels = labels[indices].to(model.device, non_blocking=True)
        logits = model(batch_images)
        loss = manual_cross_entropy(logits, batch_labels)
        batch_predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * len(indices)
        total_correct += int((batch_predictions == batch_labels).sum().item())
        predictions.append(batch_predictions.cpu())

    return {
        "loss": total_loss / len(images),
        "accuracy": total_correct / len(images),
        "predictions": torch.cat(predictions),
    }


def train_model(
    model: ManualThreeLayerCNN,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train for exactly ``epochs`` passes and return epoch-level history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        total_loss = 0.0
        total_correct = 0

        for indices in _batch_indices(
            len(train_images), batch_size, shuffle=True, generator=generator
        ):
            batch_images = train_images[indices].to(model.device, non_blocking=True)
            batch_labels = train_labels[indices].to(model.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images)
            loss = manual_cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item()) * len(indices)
            total_correct += int((logits.detach().argmax(dim=1) == batch_labels).sum().item())

        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": total_loss / len(train_images),
            "train_accuracy": total_correct / len(train_images),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss={epoch_result['train_loss']:.4f} | "
            f"accuracy={epoch_result['train_accuracy'] * 100:.2f}% | "
            f"time={epoch_result['seconds']:.1f}s"
        )

    return history


def confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, classes: int = 10
) -> torch.Tensor:
    encoded = targets.to(torch.long) * classes + predictions.to(torch.long)
    return torch.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def save_training_curves(
    history: list[dict[str, float]], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(item["epoch"]) for item in history]
    losses = [item["train_loss"] for item in history]
    accuracies = [item["train_accuracy"] * 100 for item in history]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, accuracies, marker="o", color="tab:green")
    axes[1].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_prediction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_path: str | Path,
    *,
    seed: int = 2026,
) -> tuple[Path, list[int]]:
    """Save 16 deterministic test samples in a 4×4 GT/prediction grid."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(len(images), generator=generator)[:16]

    figure, axes = plt.subplots(4, 4, figsize=(9, 9))
    for axis, index in zip(axes.flat, selected.tolist(), strict=True):
        image = images[index, 0] * MNIST_STD + MNIST_MEAN
        ground_truth = int(labels[index].item())
        prediction = int(predictions[index].item())
        axis.imshow(image.clamp(0, 1).numpy(), cmap="gray")
        axis.set_title(
            f"GT: {ground_truth} | Pred: {prediction}",
            color="green" if ground_truth == prediction else "red",
            fontsize=10,
        )
        axis.axis("off")
    figure.suptitle("MNIST: ground truth and manual-CNN prediction", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path, selected.tolist()


def save_metrics(
    output_path: str | Path,
    *,
    model: ManualThreeLayerCNN,
    history: list[dict[str, float]],
    test_result: dict[str, object],
    train_samples: int,
    test_samples: int,
    device: torch.device,
    selected_indices: list[int],
    epochs: int,
    seed: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student": "Crizepvill Dumalaog",
        "course": "AI 231",
        "exercise": "ME1",
        "dataset": "MNIST official train/test split",
        "epochs": epochs,
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "test_loss": float(test_result["loss"]),
        "test_accuracy": float(test_result["accuracy"]),
        "architecture": model.architecture(),
        "history": history,
        "prediction_grid_indices": selected_indices,
        "restrictions": {
            "cnn_library": False,
            "pytorch_neural_network_layers": False,
            "convolution": "Tensor.as_strided + einops.rearrange + torch.einsum",
            "pooling": "einops.reduce",
            "dense": "torch.einsum",
            "loss": "manual log-sum-exp cross-entropy",
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
