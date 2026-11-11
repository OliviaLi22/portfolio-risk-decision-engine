# Portfolio Risk Decision Engine

A Python decision-support engine for evaluating portfolio risk and constrained
rebalancing across competing objectives: tracking error, concentration,
historical stress losses, and trading costs.

The illustrative portfolio holds SPY, QQQ, IWM, IEF, and GLD against a SPY
benchmark, using daily data from 2018 through 2025.

## Features

- Risk diagnostics: volatility, tracking error, drawdown, and asset risk contributions.
- Historical stress tests for the COVID-19 selloff and the 2022 equity/bond drawdown.
- CVXPY optimization with tracking error, asset weight, and turnover constraints.
- Walk-forward comparison of no rebalancing, rules-based, and optimized strategies,
  including weight drift and estimated transaction costs.
- CSV tables, performance charts, and a decision summary.

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m portfolio_risk_decision_engine.run_analysis download-data
uv run python -m portfolio_risk_decision_engine.run_analysis analyze
```

The download uses Yahoo Finance and requires network access. Analysis reads saved
local Parquet files and writes reports to `results/`.
Market data, generated reports, dashboard data snapshots, and local credentials
are excluded from Git; regenerate the outputs locally with the commands above.

Edit portfolio weights, stress windows, and risk budgets in
[configs/portfolio.json](configs/portfolio.json).

## Dashboard and robustness

```bash
uv run python -m portfolio_risk_decision_engine.run_analysis dashboard
```

Open `dashboard/dist/index.html` to explore the observed results, allocation
trade-offs, and sensitivity checks. The snapshot works offline. Robustness tables
and interpretation are saved under `results/`; see
[robustness methodology](docs/robustness.md) for scope and uncertainty limits.

## Development

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests use synthetic fixtures and run without network access. See
[implementation and methodology notes](docs/implementation_notes.md) for the
architecture, modeling conventions, and limitations.

This is an illustrative research project. Results depend on data quality and
modeling assumptions; historical comparisons do not guarantee future performance.
