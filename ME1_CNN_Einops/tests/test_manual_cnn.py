from __future__ import annotations

import ast
from pathlib import Path

import torch

from manual_cnn import (
    ManualThreeLayerCNN,
    manual_conv2d,
    manual_cross_entropy,
    manual_linear,
    manual_max_pool2d,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"


def test_single_channel_convolution_has_known_values() -> None:
    image = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        requires_grad=True,
    )
    kernel = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]], requires_grad=True)
    bias = torch.tensor([0.5], requires_grad=True)
    result = manual_conv2d(image, kernel, bias)
    expected = torch.tensor([[[[-3.5, -3.5], [-3.5, -3.5]]]])
    torch.testing.assert_close(result, expected)

    result.sum().backward()
    assert image.grad is not None
    assert kernel.grad is not None
    assert bias.grad is not None


def test_multi_channel_convolution_shape_and_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    image = torch.randn((2, 3, 9, 9), generator=generator, requires_grad=True)
    kernel = torch.randn((5, 3, 3, 3), generator=generator, requires_grad=True)
    bias = torch.randn((5,), generator=generator, requires_grad=True)
    result = manual_conv2d(image, kernel, bias, stride=2, padding=1)
    assert result.shape == (2, 5, 5, 5)
    result.square().mean().backward()
    assert all(value.grad is not None for value in (image, kernel, bias))


def test_einops_max_pool_has_known_values() -> None:
    image = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    result = manual_max_pool2d(image, 2)
    expected = torch.tensor([[[[6.0, 8.0], [14.0, 16.0]]]])
    torch.testing.assert_close(result, expected)


def test_manual_linear_and_loss_are_differentiable() -> None:
    features = torch.tensor([[1.0, 2.0], [-1.0, 3.0]], requires_grad=True)
    weights = torch.tensor([[0.5, -0.5], [1.0, 0.25]], requires_grad=True)
    bias = torch.tensor([0.1, -0.2], requires_grad=True)
    targets = torch.tensor([1, 0])
    logits = manual_linear(features, weights, bias)
    loss = manual_cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is not None for value in (features, weights, bias))


def test_three_convolution_stage_architecture() -> None:
    model = ManualThreeLayerCNN(seed=11)
    images = torch.randn(4, 1, 28, 28)
    logits, shapes = model.forward_with_shapes(images)
    assert logits.shape == (4, 10)
    assert shapes == {
        "input": (4, 1, 28, 28),
        "conv1_relu": (4, 8, 28, 28),
        "pool1": (4, 8, 14, 14),
        "conv2_relu": (4, 16, 14, 14),
        "pool2": (4, 16, 7, 7),
        "conv3_relu": (4, 32, 7, 7),
        "flatten": (4, 1568),
        "hidden_relu": (4, 64),
        "logits": (4, 10),
    }
    loss = manual_cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_me1_source_imports_no_neural_network_or_cnn_library() -> None:
    """Use the AST so explanatory docstrings may name APIs without false alarms."""

    prohibited_import_roots = {
        "tensorflow",
        "keras",
    }
    prohibited_torch_modules = {
        "torch" + ".nn",
        "torch" + ".nn.functional",
    }

    for source_path in SOURCE_DIR.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        assert prohibited_import_roots.isdisjoint(
            {name.split(".")[0] for name in imports}
        )
        assert prohibited_torch_modules.isdisjoint(set(imports))

    implementation = (SOURCE_DIR / "manual_cnn.py").read_text(encoding="utf-8")
    assert "torch.einsum" in implementation
    assert "from einops import rearrange, reduce" in implementation
    assert ".as_strided" in implementation


def test_no_prohibited_layer_constructor_is_called() -> None:
    prohibited_names = {
        "Conv" + "1d",
        "Conv" + "2d",
        "Conv" + "3d",
        "Max" + "Pool2d",
        "Avg" + "Pool2d",
        "Linear",
        "Flatten",
        "Unfold",
        "Cross" + "EntropyLoss",
    }
    for source_path in SOURCE_DIR.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        called_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
        assert prohibited_names.isdisjoint(called_names)
