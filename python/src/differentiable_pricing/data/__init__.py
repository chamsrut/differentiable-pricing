"""Dataset generation, diagnostics, and read-side schema contracts.

Generation lives here for the European-option stage only. The American CRR
tables under ``american-option-dataset/1`` are **read**, never generated, by
this repository: :mod:`differentiable_pricing.data.american_schema` carries
their schema contract and manifest validator, and
:mod:`differentiable_pricing.data.american_admission` carries the task 9E
integrity and leakage gates. Their generator lives on an unmerged branch and was
deliberately not ported, so nothing here can regenerate that dataset.

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
