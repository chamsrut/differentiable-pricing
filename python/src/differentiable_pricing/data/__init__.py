"""Dataset generation and diagnostics for the European-option stage.

Importing :mod:`differentiable_pricing.data.generate` or
:mod:`differentiable_pricing.data.diagnose` requires the optional ``data``
dependency group (NumPy and PyArrow); the core pricing package does not.
Submodules are deliberately not re-exported here: both are also ``python -m``
entry points, and importing them from this package initializer would execute
them twice.

Command line::

    python -m differentiable_pricing.data.generate \\
        --config configs/european_option_dataset_v1.toml \\
        --output data/european-option-v1

    python -m differentiable_pricing.data.diagnose \\
        --dataset data/european-option-v1 \\
        --output data/european-option-v1/diagnostics.json
"""
