# Contributing

Start with [CLAUDE.md](CLAUDE.md) and
[docs/research-contract.md](docs/research-contract.md). Every change should
state its model or software assumption, add a regression test, and pass:

```bash
./scripts/check.sh
```

The full gate ends with `python3 -m pytest -q`, so install the editable
package first with `python -m pip install -e '.[dev,train]'`. A missing pytest
or missing extension fails the gate instead of skipping the Python suite. The
`--quick` mode used by the pre-commit hook stops after the C++ tests; run the
full gate before opening a pull request.

Tests are split by directory so CI can keep one job PyTorch-free:
`python/tests/ml/` holds every PyTorch-dependent test, and everything else
under `python/tests/` must import only the `data` extra. Add new
PyTorch-dependent tests under `python/tests/ml/`;
`scripts/check_test_partition.py` fails the gate if a test outside that
directory reaches `torch` or `differentiable_pricing.ml`.

Pull requests should include:

- the problem and numerical assumptions;
- tests and commands run;
- price, Greek, convergence, or latency evidence as applicable;
- known limits and out-of-domain behaviour;
- independent code review and, for quant changes, numerical review.

Do not commit proprietary market data, licensed datasets, credentials, large
generated labels, or model checkpoints.
