"""Dataset generation for the European-option stage.

Importing :mod:`differentiable_pricing.data.generate` requires the optional
``data`` dependency group (NumPy and PyArrow); the core pricing package does
not. Submodules are deliberately not re-exported here: ``generate`` is also a
``python -m`` entry point, and importing it from this package initializer
would execute it twice.

Command line::

    python -m differentiable_pricing.data.generate \\
        --config configs/european_option_dataset_v1.toml \\
        --output data/european-option-v1
"""
