"""Task 9H candidate architectures: one dense stack and one residual stack.

A small named pair, not a framework. New candidates arrive as new immutable
configurations, not as new abstraction layers.

There is no batch or layer normalization and no dropout, on purpose: both make a
single row's prediction depend on batch composition or on a random mask, which
is not a property a pricing function should have.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

import torch
from torch import nn

from .attempts import ARCHITECTURES

#: Activation name -> module factory.
ACTIVATIONS: Final = {
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "gelu": lambda: nn.GELU(approximate="none"),
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}

MAX_HIDDEN_DIMENSION: Final = 4096
MAX_BLOCKS: Final = 64
MAX_PARAMETERS: Final = 10_000_000


class ArchitectureError(ValueError):
    """Raised when an architecture section does not satisfy its contract."""


def _activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ArchitectureError(f"unknown activation {name!r}")
    return ACTIVATIONS[name]()


def _check_widths(widths: Sequence[int], where: str) -> tuple[int, ...]:
    if not widths:
        raise ArchitectureError(f"{where} must be non-empty")
    values = []
    for width in widths:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ArchitectureError(f"{where} entries must be positive integers")
        if width > MAX_HIDDEN_DIMENSION:
            raise ArchitectureError(f"{where} entry exceeds {MAX_HIDDEN_DIMENSION}")
        values.append(int(width))
    return tuple(values)


class DenseNetwork(nn.Module):
    """``Linear -> activation`` repeated, then a scalar linear."""

    def __init__(
        self, input_dimension: int, hidden_dimensions: Sequence[int], activation: str
    ) -> None:
        super().__init__()
        widths = _check_widths(hidden_dimensions, "hidden_dimensions")
        self.activation_name = activation
        modules: list[nn.Module] = []
        previous = int(input_dimension)
        for width in widths:
            modules.append(nn.Linear(previous, width, dtype=torch.float64))
            modules.append(_activation(activation))
            previous = width
        modules.append(nn.Linear(previous, 1, dtype=torch.float64))
        self.layers = nn.Sequential(*modules)

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        return self.layers(standardized_features).squeeze(-1)


class ResidualBlock(nn.Module):
    """``h + W2 * act(W1 h + b1) + b2``: an identity shortcut, nothing else."""

    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.inner = nn.Linear(width, width, dtype=torch.float64)
        self.outer = nn.Linear(width, width, dtype=torch.float64)
        self.activation = _activation(activation)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.outer(self.activation(self.inner(hidden)))


class ResidualNetwork(nn.Module):
    """Input projection, residual blocks of one width, scalar linear output."""

    def __init__(self, input_dimension: int, width: int, blocks: int, activation: str) -> None:
        super().__init__()
        (checked_width,) = _check_widths((width,), "width")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= MAX_BLOCKS:
            raise ArchitectureError(f"blocks must be an integer in [1, {MAX_BLOCKS}]")
        self.activation_name = activation
        self.projection = nn.Linear(int(input_dimension), checked_width, dtype=torch.float64)
        self.blocks = nn.ModuleList(ResidualBlock(checked_width, activation) for _ in range(blocks))
        self.output = nn.Linear(checked_width, 1, dtype=torch.float64)

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(standardized_features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden).squeeze(-1)


def declared_parameter_count(section: Mapping[str, Any], input_dimension: int) -> int:
    """Count parameters from the declaration alone, before allocating anything.

    Checked *before* construction on purpose: an over-large declaration would
    otherwise allocate gigabytes of weights only to be rejected on the next
    line, turning a configuration typo into an out-of-memory kill.
    """
    if section["name"] == "smooth_mlp":
        widths = _check_widths(section["hidden_dimensions"], "hidden_dimensions")
        dimensions = (int(input_dimension), *widths, 1)
        return sum(inputs * outputs + outputs for inputs, outputs in pairwise(dimensions))
    (width,) = _check_widths((section["width"],), "width")
    blocks = int(section["blocks"])
    projection = int(input_dimension) * width + width
    per_block = 2 * (width * width + width)
    return projection + blocks * per_block + width + 1


def build_network(section: Mapping[str, Any], input_dimension: int) -> nn.Module:
    """Dispatch one immutable ``[architecture]`` section to a network."""
    name = section.get("name")
    activation = section.get("activation")
    if not isinstance(name, str) or name not in ARCHITECTURES:
        raise ArchitectureError(f"architecture.name must be one of {list(ARCHITECTURES)}")
    if not isinstance(activation, str) or activation not in ACTIVATIONS:
        raise ArchitectureError(f"architecture.activation must be one of {sorted(ACTIVATIONS)}")
    if name == "smooth_mlp":
        if not isinstance(section.get("hidden_dimensions"), list):
            raise ArchitectureError("smooth_mlp requires hidden_dimensions")
    else:
        width = section.get("width")
        blocks = section.get("blocks")
        if (
            isinstance(width, bool)
            or isinstance(blocks, bool)
            or not isinstance(width, int)
            or not isinstance(blocks, int)
        ):
            raise ArchitectureError("smooth_residual requires integer width and blocks")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ArchitectureError(f"blocks must be an integer in [1, {MAX_BLOCKS}]")
    declared = declared_parameter_count(section, input_dimension)
    if declared > MAX_PARAMETERS:
        raise ArchitectureError(
            f"architecture declares {declared} parameters, over {MAX_PARAMETERS}"
        )
    if name == "smooth_mlp":
        network: nn.Module = DenseNetwork(
            input_dimension, section["hidden_dimensions"], activation
        )
    else:
        network = ResidualNetwork(
            input_dimension, int(section["width"]), int(section["blocks"]), activation
        )
    parameters = parameter_count(network)
    if parameters != declared:
        raise ArchitectureError(
            f"built network has {parameters} parameters but the declaration implies {declared}"
        )
    return network


def parameter_count(network: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in network.parameters()))
