# Contributing

Start with [CLAUDE.md](CLAUDE.md) and
[docs/research-contract.md](docs/research-contract.md). Every change should
state its model or software assumption, add a regression test, and pass:

```bash
./scripts/check.sh
```

The full gate ends with `python3 -m pytest -q`, so install the editable
package first with `python -m pip install -e '.[dev,data]'`. A missing pytest
or missing extension fails the gate instead of skipping the Python suite. The
`--quick` mode used by the pre-commit hook stops after the C++ tests; run the
full gate before opening a pull request.

Pull requests should include:

- the problem and numerical assumptions;
- tests and commands run;
- price, Greek, convergence, or latency evidence as applicable;
- known limits and out-of-domain behaviour;
- independent code review and, for quant changes, numerical review.

Do not commit proprietary market data, licensed datasets, credentials, large
generated labels, or model checkpoints.
