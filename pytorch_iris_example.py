"""Beginner-friendly PyTorch classification example using the Iris dataset.

Running this file creates the sample CSV (if needed), trains a small neural
network, evaluates it, saves plots and metrics, and writes a reusable model
checkpoint. The code is intentionally explicit so each training step is easy
to match with the companion PDF manual.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "iris_sample.csv"
ARTIFACT_DIR = PROJECT_ROOT / "output" / "pytorch_iris_example"
MODEL_PATH = ARTIFACT_DIR / "iris_mlp_checkpoint.pt"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Make this small example repeatable across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """Prefer a supported accelerator and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_sample_csv(path: Path = DATA_PATH) -> pd.DataFrame:
    """Export scikit-learn's bundled Iris data as an easy-to-inspect CSV."""
    iris = load_iris(as_frame=True)
    column_names = {
        "sepal length (cm)": "sepal_length_cm",
        "sepal width (cm)": "sepal_width_cm",
        "petal length (cm)": "petal_length_cm",
        "petal width (cm)": "petal_width_cm",
    }
    frame = iris.frame.rename(columns=column_names).copy()
    frame["species"] = frame["target"].map(
        {index: name for index, name in enumerate(iris.target_names)}
    )
    frame = frame.drop(columns="target")
    frame.to_csv(path, index=False)
    return frame


def load_and_split_data() -> dict[str, object]:
    """Load, stratify, split, and standardize without leaking test data."""
    frame = create_sample_csv()
    feature_names = [column for column in frame.columns if column != "species"]
    class_names = ["setosa", "versicolor", "virginica"]
    class_to_index = {name: index for index, name in enumerate(class_names)}

    features = frame[feature_names].to_numpy(dtype=np.float32)
    labels = frame["species"].map(class_to_index).to_numpy(dtype=np.int64)

    # First reserve 30% for validation + test, then divide it evenly.
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    # Learn scaling values from training data only.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    def make_dataset(x_values: np.ndarray, y_values: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.from_numpy(x_values), torch.from_numpy(y_values))

    return {
        "frame": frame,
        "feature_names": feature_names,
        "class_names": class_names,
        "mean": mean,
        "std": std,
        "train_dataset": make_dataset(x_train, y_train),
        "val_dataset": make_dataset(x_val, y_val),
        "test_dataset": make_dataset(x_test, y_test),
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
    }


class IrisMLP(nn.Module):
    """A small multilayer perceptron for four inputs and three classes."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # The output contains logits. CrossEntropyLoss applies the needed math.
        return self.layers(features)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(features)
            loss = loss_fn(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


def predict_one(
    model: nn.Module,
    measurements: list[float],
    mean: np.ndarray,
    std: np.ndarray,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, list[float]]:
    """Scale one flower, produce probabilities, and return the top class."""
    scaled = (np.asarray(measurements, dtype=np.float32) - mean) / std
    features = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=1).squeeze(0)
    values = probabilities.cpu().tolist()
    return class_names[int(probabilities.argmax().item())], values


def plot_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    colors = {"setosa": "#0f766e", "versicolor": "#f59e0b", "virginica": "#4f46e5"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for species, group in frame.groupby("species"):
        ax.scatter(
            group["petal_length_cm"],
            group["petal_width_cm"],
            label=species,
            color=colors[species],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_title("Iris dataset: petal measurements reveal class structure", loc="left", weight="bold")
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#0f766e")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="#f59e0b")
    axes[0].set_title("Loss", loc="left", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train", color="#0f766e")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation", color="#f59e0b")
    axes[1].set_title("Accuracy", loc="left", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fraction correct")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("Learning curves", x=0.07, ha="left", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_split_data()
    device = choose_device()

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        data["train_dataset"], batch_size=16, shuffle=True, generator=generator
    )
    val_loader = DataLoader(data["val_dataset"], batch_size=32, shuffle=False)
    test_loader = DataLoader(data["test_dataset"], batch_size=32, shuffle=False)

    model = IrisMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 30
    epochs_without_improvement = 0

    for epoch in range(1, 201):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | val accuracy {val_accuracy:.1%}"
            )
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)

    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.inference_mode():
        for features, labels in test_loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    class_names = data["class_names"]
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    prediction, probabilities = predict_one(
        model,
        measurements=[5.9, 3.0, 5.1, 1.8],
        mean=data["mean"],
        std=data["std"],
        class_names=class_names,
        device=device,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mean": torch.tensor(data["mean"], dtype=torch.float32),
        "std": torch.tensor(data["std"], dtype=torch.float32),
        "feature_names": data["feature_names"],
        "class_names": class_names,
        "hidden_sizes": [16, 8],
    }
    torch.save(checkpoint, MODEL_PATH)

    plot_dataset(data["frame"], ARTIFACT_DIR / "iris_scatter.png")
    plot_history(history, ARTIFACT_DIR / "training_curves.png")
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Test-set confusion matrix", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "split_sizes": data["split_sizes"],
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "sample_measurements": [5.9, 3.0, 5.1, 1.8],
        "sample_prediction": prediction,
        "sample_probabilities": {
            name: probability for name, probability in zip(class_names, probabilities)
        },
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.1%}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Sample prediction: {prediction}")
    print(f"Saved dataset: {DATA_PATH}")
    print(f"Saved model and figures: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
