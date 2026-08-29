"""Build the reproducible ME1 notebook from the checked-in source modules."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "ME1_MNIST_Manual_CNN.ipynb"
MANUAL_SOURCE = (PROJECT_ROOT / "src" / "manual_cnn.py").read_text(encoding="utf-8")
EXPERIMENT_SOURCE = (PROJECT_ROOT / "src" / "experiment.py").read_text(
    encoding="utf-8"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {
        "display_name": "Python (AI 222 + AI 231)",
        "language": "python",
        "name": "ai222-231",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
notebook["cells"] = [
    markdown(
        """
        # ME1: Three-layer CNN for MNIST using Einops and Einsum

        **Student:** Crizepvill Dumalaog

        **Course:** AI 231

        **Training requirement:** exactly five epochs on the official MNIST
        training split, followed by one report on the official test split.

        ## Agent disclosure

        A coding agent created the repository structure, manual implementation,
        tests, notebook, training run, plots, Git history, and GitHub repository.
        The student is still responsible for inspecting and understanding every
        operation, tensor shape, result, and limitation before submission.

        ## Restrictions

        This notebook uses no CNN library and no PyTorch neural-network layer
        API. There is no built-in convolution, pooling, flatten, linear,
        activation, or loss layer. The three convolutional stages use a strided
        tensor window view, `einops.rearrange`, and `torch.einsum`; pooling uses
        `einops.reduce`; dense transformations use `einsum`; and cross-entropy
        is written directly from log-sum-exp. Torchvision is used only to
        download/read MNIST—not for a model, transform, or CNN operation. No NLP
        library or NLP operation is involved.
        """
    ),
    markdown(
        r"""
        ## How the manual convolution works

        For input $X$ and kernel $W$, an overlapping view exposes patches with
        shape `(batch, in_channel, out_h, out_w, kernel_h, kernel_w)`.
        Einops flattens the last three patch axes and the corresponding kernel
        axes. The learned cross-correlation is then

        $$Y_{b,h,w,o}=\sum_p X^{patch}_{b,h,w,p}W^{flat}_{p,o}+b_o.$$

        `einsum("bhwp,po->bhwo", patches, kernels)` performs exactly that
        contraction. Autograd follows the view, rearrangement, reduction, and
        contraction back to each ordinary leaf tensor parameter.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        candidates = [
            Path.cwd(),
            Path.cwd() / "ME1_CNN_Einops",
            Path.cwd() / "AI 231" / "ME1_CNN_Einops",
        ]
        PROJECT_ROOT = next(
            path.resolve() for path in candidates
            if (path / "src" / "manual_cnn.py").exists()
        )
        SRC_DIR = PROJECT_ROOT / "src"
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        print(f"Project: {PROJECT_ROOT}")
        print(f"Python: {sys.version.split()[0]}")
        """
    ),
    markdown(
        """
        ## Manual layer and model implementation

        The complete checked-in implementation is included in this code cell so
        the submission can be inspected directly. It intentionally contains no
        PyTorch neural-network layer import.
        """
    ),
    code(MANUAL_SOURCE),
    markdown(
        """
        ## Data, training, evaluation, and visualization utilities

        MNIST's official 60,000-image training split and 10,000-image test split
        are loaded without augmentation. The test split is not used for model
        selection or parameter updates.
        """
    ),
    code(EXPERIMENT_SOURCE),
    code(
        """
        SEED = 42
        EPOCHS = 5
        BATCH_SIZE = 256
        LEARNING_RATE = 1e-3

        set_seed(SEED)
        device = choose_device("auto")
        dataset = load_mnist(PROJECT_ROOT / "data")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Training images: {tuple(dataset.train_images.shape)}")
        print(f"Test images: {tuple(dataset.test_images.shape)}")
        print(f"Training labels: {tuple(dataset.train_labels.shape)}")
        print(f"Test labels: {tuple(dataset.test_labels.shape)}")
        assert len(dataset.train_images) == 60_000
        assert len(dataset.test_images) == 10_000
        """
    ),
    markdown(
        """
        ## Architecture and tensor-shape inspection

        “Three-layer CNN” means three convolutional stages. The classifier head
        contains two additional dense contractions, also implemented with
        `einsum` rather than a layer library.
        """
    ),
    code(
        """
        model = ManualThreeLayerCNN(seed=SEED, device=device)
        sample_logits, shape_trace = model.forward_with_shapes(
            dataset.train_images[:4].to(device)
        )
        print("Architecture:", model.architecture())
        print("Shape trace:")
        for name, shape in shape_trace.items():
            print(f"  {name:14s} -> {shape}")
        sanity_loss = manual_cross_entropy(
            sample_logits, dataset.train_labels[:4].to(device)
        )
        sanity_loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
        for parameter in model.parameters():
            parameter.grad = None
        print(f"Gradient sanity loss: {sanity_loss.item():.4f}")
        """
    ),
    markdown(
        """
        ## Five-epoch training run

        The following is the one required training call. It performs exactly
        five complete passes over all 60,000 training images.
        """
    ),
    code(
        """
        history = train_model(
            model,
            dataset.train_images,
            dataset.train_labels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            seed=SEED,
        )
        assert len(history) == 5
        assert [int(row["epoch"]) for row in history] == [1, 2, 3, 4, 5]
        """
    ),
    markdown(
        """
        ## Official test-split result

        Evaluation occurs after the fifth epoch. Test labels are used only for
        the reported metrics and visualization, never for training.
        """
    ),
    code(
        """
        test_result = evaluate_model(
            model,
            dataset.test_images,
            dataset.test_labels,
            batch_size=512,
        )
        TEST_ACCURACY = float(test_result["accuracy"])
        print(f"Test loss: {float(test_result['loss']):.4f}")
        print(f"Test accuracy: {TEST_ACCURACY * 100:.2f}%")

        ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
        curves_path = save_training_curves(
            history, ARTIFACT_DIR / "training_curves.png"
        )
        grid_path, selected_indices = save_prediction_grid(
            dataset.test_images,
            dataset.test_labels,
            test_result["predictions"],
            ARTIFACT_DIR / "predictions_4x4.png",
            seed=2026,
        )
        metrics_path = save_metrics(
            ARTIFACT_DIR / "metrics.json",
            model=model,
            history=history,
            test_result=test_result,
            train_samples=len(dataset.train_images),
            test_samples=len(dataset.test_images),
            device=device,
            selected_indices=selected_indices,
            epochs=EPOCHS,
            seed=SEED,
        )
        torch.save(
            {"architecture": model.architecture(), "state_dict": model.state_dict()},
            ARTIFACT_DIR / "manual_cnn_checkpoint.pt",
        )
        print(f"Metrics saved to: {metrics_path}")
        print(f"Training curves saved to: {curves_path}")
        print(f"Prediction grid saved to: {grid_path}")
        """
    ),
    markdown(
        """
        ## Ground truth and prediction grid

        Sixteen deterministically sampled test images are displayed in a 4×4
        grid. Each title contains `GT` (ground truth) and `Pred` (prediction).
        Green titles are correct; red titles are errors.
        """
    ),
    code(
        """
        from IPython.display import Image, display

        assert len(selected_indices) == 16
        display(Image(filename=str(grid_path)))
        """
    ),
    markdown(
        """
        ## Training curves

        The plots below show the loss and accuracy measured over the training
        split after each full epoch.
        """
    ),
    code(
        """
        display(Image(filename=str(curves_path)))
        """
    ),
    markdown(
        """
        ## Compliance and interpretation summary

        - Convolution: `as_strided` windows → Einops flattening → `einsum`.
        - Pooling: `einops.reduce(..., "max")`.
        - Dense layers: `einsum`.
        - Activation and loss: elementary tensor operations.
        - Optimizer: Adam updates the ordinary leaf tensors returned by the
          model's `parameters()` method.
        - Dataset utility: torchvision MNIST reader only.
        - CNN/NLP libraries: none.

        The automated test suite also parses the ME1 source abstract syntax tree
        to reject prohibited imports and layer constructors. Passing those tests
        does not replace understanding: inspect the window strides, contraction
        indices, shape trace, loss, epoch logs, test result, and prediction grid.
        """
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
