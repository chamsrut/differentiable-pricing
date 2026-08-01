"""Classical and neural derivative-pricing research."""

from ._core import (
    DenseLayer,
    SmoothMlp,
    black_scholes,
    crr_diagnostics,
    crr_lattice_parameters,
    crr_price,
    crr_price_batch,
    lsm_price,
    lsm_training_memory_bytes,
)
from .stages import ResearchStage

__all__ = [
    "DenseLayer",
    "ResearchStage",
    "SmoothMlp",
    "black_scholes",
    "crr_diagnostics",
    "crr_lattice_parameters",
    "crr_price",
    "crr_price_batch",
    "lsm_price",
    "lsm_training_memory_bytes",
]

__version__ = "0.1.0"
