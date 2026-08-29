"""Manual three-convolution-layer MNIST model using tensors, Einops, and einsum.

This module deliberately avoids torch.nn, torch.nn.functional, and every
built-in convolution/pooling/linear/loss layer. Parameters are ordinary leaf
tensors tracked by autograd and updated by a standard optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from einops import rearrange, reduce
import torch


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a scalar or pair and reject non-positive dimensions."""

    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"Expected a positive integer or pair, received {value!r}.")
    return int(pair[0]), int(pair[1])


def manual_pad2d(x: torch.Tensor, padding: int | tuple[int, int]) -> torch.Tensor:
    """Zero-pad the final two dimensions using basic tensor allocation/copy."""

    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Padding cannot be negative.")
    if pad_h == 0 and pad_w == 0:
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, received shape {tuple(x.shape)}.")

    batch, channels, height, width = x.shape
    padded = x.new_zeros(
        (batch, channels, height + 2 * pad_h, width + 2 * pad_w)
    )
    padded[:, :, pad_h : pad_h + height, pad_w : pad_w + width] = x
    return padded


def sliding_windows_2d(
    x: torch.Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
) -> torch.Tensor:
    """Return a BCHW-compatible overlapping window view using ``as_strided``.

    Output shape is ``(batch, channels, out_h, out_w, kernel_h, kernel_w)``.
    No convolution helper or unfold layer is used.
    """

    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, received shape {tuple(x.shape)}.")
    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    batch, channels, height, width = x.shape
    if kernel_h > height or kernel_w > width:
        raise ValueError("Kernel cannot be larger than the padded input.")

    out_h = 1 + (height - kernel_h) // stride_h
    out_w = 1 + (width - kernel_w) // stride_w
    stride_b, stride_c, stride_y, stride_x = x.stride()
    shape = (batch, channels, out_h, out_w, kernel_h, kernel_w)
    strides = (
        stride_b,
        stride_c,
        stride_y * stride_h,
        stride_x * stride_w,
        stride_y,
        stride_x,
    )
    return x.as_strided(size=shape, stride=strides)


def manual_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> torch.Tensor:
    """Apply a learned 2-D cross-correlation via strided views and ``einsum``."""

    if x.ndim != 4 or weight.ndim != 4 or bias.ndim != 1:
        raise ValueError("Expected x=BCHW, weight=OIKK, and bias=O tensors.")
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    if x.shape[1] != in_channels:
        raise ValueError(
            f"Input has {x.shape[1]} channels but kernel expects {in_channels}."
        )
    if bias.shape[0] != out_channels:
        raise ValueError("Bias length must equal the number of output channels.")

    padded = manual_pad2d(x, padding)
    windows = sliding_windows_2d(padded, (kernel_h, kernel_w), stride)
    patches = rearrange(
        windows,
        "batch channel out_h out_w kernel_h kernel_w -> "
        "batch out_h out_w (channel kernel_h kernel_w)",
    )
    flat_kernels = rearrange(
        weight,
        "out_channel in_channel kernel_h kernel_w -> "
        "(in_channel kernel_h kernel_w) out_channel",
    )
    output = torch.einsum("bhwp,po->bhwo", patches, flat_kernels)
    output = output + bias
    return rearrange(output, "batch out_h out_w channel -> batch channel out_h out_w")


def manual_relu(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ReLU without a neural-network activation module."""

    return torch.where(x > 0, x, torch.zeros_like(x))


def manual_max_pool2d(
    x: torch.Tensor, pool_size: int | tuple[int, int] = 2
) -> torch.Tensor:
    """Non-overlapping max pooling implemented by ``einops.reduce``."""

    pool_h, pool_w = _pair(pool_size)
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW input, received shape {tuple(x.shape)}.")
    if x.shape[-2] % pool_h or x.shape[-1] % pool_w:
        raise ValueError("Spatial dimensions must be divisible by the pool size.")
    return reduce(
        x,
        "batch channel (out_h pool_h) (out_w pool_w) -> batch channel out_h out_w",
        "max",
        pool_h=pool_h,
        pool_w=pool_w,
    )


def manual_flatten(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch axes using Einops."""

    return rearrange(x, "batch ... -> batch (...)")


def manual_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Dense transformation implemented as an ``einsum`` contraction."""

    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("Expected x=BI, weight=IO, and bias=O tensors.")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != bias.shape[0]:
        raise ValueError("Incompatible dense-layer tensor shapes.")
    return torch.einsum("bi,io->bo", x, weight) + bias


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean multiclass negative log likelihood using log-sum-exp stabilization."""

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits=(batch, classes) and targets=(batch,).")
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    rows = torch.arange(targets.shape[0], device=targets.device)
    return -log_probabilities[rows, targets].mean()


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the fraction of correct class predictions."""

    return float((logits.argmax(dim=1) == targets).float().mean().item())


@dataclass(frozen=True)
class ManualCNNConfig:
    """Architecture definition for the three-convolution-stage model."""

    input_channels: int = 1
    image_size: int = 28
    conv1_channels: int = 8
    conv2_channels: int = 16
    conv3_channels: int = 32
    hidden_features: int = 64
    classes: int = 10


class ManualThreeLayerCNN:
    """Three convolutional stages and an einsum classifier without layer APIs."""

    def __init__(
        self,
        config: ManualCNNConfig | None = None,
        *,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or ManualCNNConfig()
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        c = self.config
        flattened_features = c.conv3_channels * (c.image_size // 4) ** 2
        self.conv1_weight = self._weight(
            (c.conv1_channels, c.input_channels, 5, 5),
            c.input_channels * 5 * 5,
            generator,
        )
        self.conv1_bias = self._bias(c.conv1_channels)
        self.conv2_weight = self._weight(
            (c.conv2_channels, c.conv1_channels, 3, 3),
            c.conv1_channels * 3 * 3,
            generator,
        )
        self.conv2_bias = self._bias(c.conv2_channels)
        self.conv3_weight = self._weight(
            (c.conv3_channels, c.conv2_channels, 3, 3),
            c.conv2_channels * 3 * 3,
            generator,
        )
        self.conv3_bias = self._bias(c.conv3_channels)
        self.hidden_weight = self._weight(
            (flattened_features, c.hidden_features),
            flattened_features,
            generator,
        )
        self.hidden_bias = self._bias(c.hidden_features)
        self.output_weight = self._weight(
            (c.hidden_features, c.classes), c.hidden_features, generator
        )
        self.output_bias = self._bias(c.classes)

    def _weight(
        self,
        shape: tuple[int, ...],
        fan_in: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        values = torch.randn(shape, generator=generator) * math.sqrt(2.0 / fan_in)
        return values.to(self.device).requires_grad_(True)

    def _bias(self, features: int) -> torch.Tensor:
        return torch.zeros(features, device=self.device, requires_grad=True)

    def named_parameters(self) -> list[tuple[str, torch.Tensor]]:
        """Return stable parameter names and their leaf tensors."""

        names = (
            "conv1_weight",
            "conv1_bias",
            "conv2_weight",
            "conv2_bias",
            "conv3_weight",
            "conv3_bias",
            "hidden_weight",
            "hidden_bias",
            "output_weight",
            "output_bias",
        )
        return [(name, getattr(self, name)) for name in names]

    def parameters(self) -> Iterable[torch.Tensor]:
        """Yield tensors in the format expected by PyTorch optimizers."""

        return (parameter for _, parameter in self.named_parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return detached CPU copies suitable for checkpoints."""

        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Copy checkpoint values into the existing leaf tensors."""

        expected = {name for name, _ in self.named_parameters()}
        if set(state) != expected:
            raise ValueError(f"State keys do not match model parameters: {set(state) ^ expected}")
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(state[name].to(self.device))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def architecture(self) -> dict[str, object]:
        return asdict(self.config) | {"trainable_parameters": self.parameter_count()}

    def forward_with_shapes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        """Forward pass plus named shapes for notebook inspection."""

        shapes: dict[str, tuple[int, ...]] = {"input": tuple(x.shape)}
        x = manual_conv2d(x, self.conv1_weight, self.conv1_bias, padding=2)
        x = manual_relu(x)
        shapes["conv1_relu"] = tuple(x.shape)
        x = manual_max_pool2d(x, 2)
        shapes["pool1"] = tuple(x.shape)

        x = manual_conv2d(x, self.conv2_weight, self.conv2_bias, padding=1)
        x = manual_relu(x)
        shapes["conv2_relu"] = tuple(x.shape)
        x = manual_max_pool2d(x, 2)
        shapes["pool2"] = tuple(x.shape)

        x = manual_conv2d(x, self.conv3_weight, self.conv3_bias, padding=1)
        x = manual_relu(x)
        shapes["conv3_relu"] = tuple(x.shape)

        x = manual_flatten(x)
        shapes["flatten"] = tuple(x.shape)
        x = manual_linear(x, self.hidden_weight, self.hidden_bias)
        x = manual_relu(x)
        shapes["hidden_relu"] = tuple(x.shape)
        logits = manual_linear(x, self.output_weight, self.output_bias)
        shapes["logits"] = tuple(logits.shape)
        return logits, shapes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_shapes(x)
        return logits
