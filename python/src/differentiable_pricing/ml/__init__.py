"""Neural-surrogate training and evaluation.

The submodules are intentionally not imported here. PyTorch is an optional
dependency, and both :mod:`differentiable_pricing.ml.train` and
:mod:`differentiable_pricing.ml.evaluate` are ``python -m`` entry points.
The :mod:`differentiable_pricing.ml.constrain` entry point derives a
European-bound-constrained artifact without retraining.
"""
