"""Classical and neural derivative-pricing research."""

from ._core import DenseLayer, SmoothMlp, black_scholes
from .stages import ResearchStage

__all__ = [
    "DenseLayer",
    "ResearchStage",
    "SmoothMlp",
    "black_scholes",
]

__version__ = "0.1.0"

