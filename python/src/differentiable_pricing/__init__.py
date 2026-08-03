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
from ._pde import (
    PsorConvergenceError,
    pde_discount_factor,
    pde_log_discount,
    pde_post_dividend_spot,
    pde_price,
    pde_segment_rate,
)
from .stages import ResearchStage

__all__ = [
    "DenseLayer",
    "PsorConvergenceError",
    "ResearchStage",
    "SmoothMlp",
    "black_scholes",
    "crr_diagnostics",
    "crr_lattice_parameters",
    "crr_price",
    "crr_price_batch",
    "lsm_price",
    "lsm_training_memory_bytes",
    "pde_discount_factor",
    "pde_log_discount",
    "pde_post_dividend_spot",
    "pde_price",
    "pde_segment_rate",
]

__version__ = "0.1.0"
