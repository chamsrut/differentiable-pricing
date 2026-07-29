# Contributing

Start with [CLAUDE.md](CLAUDE.md) and
[docs/research-contract.md](docs/research-contract.md). Every change should
state its model or software assumption, add a regression test, and pass:

```bash
./scripts/check.sh
```

For Python-binding tests, install the editable package with
`python -m pip install -e '.[dev]'` and run `pytest -q`.

Pull requests should include:

- the problem and numerical assumptions;
- tests and commands run;
- price, Greek, convergence, or latency evidence as applicable;
- known limits and out-of-domain behaviour;
- independent code review and, for quant changes, numerical review.

Do not commit proprietary market data, licensed datasets, credentials, large
generated labels, or model checkpoints.
