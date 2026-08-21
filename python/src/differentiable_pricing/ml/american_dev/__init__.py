"""Task 9H: adaptive, exploratory American **price**-model development.

Everything here is development infrastructure, not confirmatory research.
Selection happens repeatedly against ``validation``, so nothing this package
measures is an unbiased result.

Two rules bind every module:

* ``train`` and ``validation`` are the only reachable partitions. No code path
  here opens, hashes, stats, imports, counts or inspects ``interpolation_test``
  or any other final partition, and there is no final-evaluation entry point.
* training is a **manual, terminal-invoked human command**. No test, hook, CI
  job or repository check calls it.

Scope is deliberately price-only. Greeks, latency, implied volatility and
transfer learning are separate follow-up stages that begin only after a
candidate meets the development criterion, and none of their machinery is built
here.

Submodules are imported explicitly rather than re-exported, so the offline
attempt-log tool can load :mod:`attempts` without pulling in PyTorch.

See ``docs/tasks/active/task-9h-american-pricer-development.md``.
"""

from __future__ import annotations

__all__: list[str] = []
