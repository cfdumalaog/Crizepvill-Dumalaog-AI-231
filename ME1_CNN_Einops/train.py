"""Command-line runner for the manual three-layer MNIST CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment import (  # noqa: E402
    choose_device,
    evaluate_model,
    load_mnist,
    save_metrics,
    save_prediction_grid,
    save_training_curves,
    set_seed,
    train_model,
)
from manual_cnn import ManualThreeLayerCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN whose layers are implemented with tensors/Einops/einsum."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "runs" / "cli"
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the assignment run.",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for the assignment run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    dataset = load_mnist(
        args.data_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = ManualThreeLayerCNN(seed=args.seed, device=device)

    print(json.dumps(model.architecture(), indent=2))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"MNIST tensors: train={tuple(dataset.train_images.shape)}, "
        f"test={tuple(dataset.test_images.shape)}"
    )

    history = train_model(
        model,
        dataset.train_images,
        dataset.train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    test_result = evaluate_model(
        model,
        dataset.test_images,
        dataset.test_labels,
        batch_size=max(args.batch_size, 512),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = save_training_curves(history, args.output_dir / "training_curves.png")
    grid_path, selected = save_prediction_grid(
        dataset.test_images,
        dataset.test_labels,
        test_result["predictions"],
        args.output_dir / "predictions_4x4.png",
        seed=2026,
    )
    metrics_path = save_metrics(
        args.output_dir / "metrics.json",
        model=model,
        history=history,
        test_result=test_result,
        train_samples=len(dataset.train_images),
        test_samples=len(dataset.test_images),
        device=device,
        selected_indices=selected,
        epochs=args.epochs,
        seed=args.seed,
    )
    torch.save(
        {"architecture": model.architecture(), "state_dict": model.state_dict()},
        args.output_dir / "manual_cnn_checkpoint.pt",
    )

    print(f"Test loss: {float(test_result['loss']):.4f}")
    print(f"Test accuracy: {float(test_result['accuracy']) * 100:.2f}%")
    print(f"Metrics: {metrics_path}")
    print(f"Curves: {curves_path}")
    print(f"Prediction grid: {grid_path}")


if __name__ == "__main__":
    main()
