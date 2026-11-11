# Implementation and Methodology Notes

A command-line, first-version portfolio risk decision system for an **illustrative**
multi-asset ETF portfolio. It explores how tracking error, concentration, historical
stress losses, and turnover limits can point toward different rebalancing decisions.
This is not a client account, an alpha model, or an investment recommendation.

## Portfolio and setup

The example holds SPY 50%, QQQ 25%, IWM 10%, IEF 10%, and GLD 5%, with SPY as
the 100% benchmark. The configured price period is **2018-01-01 inclusive to
2026-01-01 exclusive**. All assumptions are in
[`configs/portfolio.json`](../configs/portfolio.json). Python 3.13, uv, and roughly
8 GB of memory are sufficient; no database or web service is required.

```bash
uv sync
uv run python -m portfolio_risk_decision_engine.run_analysis download-data
uv run python -m portfolio_risk_decision_engine.run_analysis analyze
```

The first command downloads market data and requires network access. The analysis
command reads **only saved local Parquet**. Once data are present, the last command
is the one-command report run. The installed `portfolio-risk-decision-engine`
console command accepts the same `download-data` and `analyze` arguments. Paths
can be overridden with `--config`, `--data-dir`, and `--results-dir`.

Run engineering checks with:

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

## Architecture

| Module | Responsibility |
| --- | --- |
| `config.py` | Validate dates, unique ticker keys, weights, budgets, and stress windows. |
| `market_data.py` | Download auto-adjusted Yahoo Finance closes; normalize varying column layouts; validate and store Parquet. |
| `risk.py` | Constant-weight return series, ex-post risk, active return, concentration, drawdown, and covariance risk contributions. |
| `stress.py` | Observed historical ETF returns within configured windows. |
| `optimization.py` | DCP-compliant constrained rebalancing and consistent trading-cost accounting. |
| `validation.py` | Walk-forward decisions and net strategy return comparison. |
| `reporting.py` | CSV tables, PNG figures, and decision summary. |
| `run_analysis.py` | Local CLI. |

`data/processed/prices.parquet` and `returns.parquet` are generated locally and
excluded from Git. The analysis verifies that saved returns match saved prices.
Figures appear in `results/figures/`; tables in `results/tables/`; interpretation
in `results/decision_summary.md`. Generated results should only be published after
reviewing the data and assumptions. Unit tests use deterministic synthetic data
and do not call Yahoo Finance.

## Methodology

Yahoo Finance is queried with `auto_adjust=True` and the adjusted `Close` field.
Prices must be positive, finite, increasing by unique date, complete for all
configured tickers, and close to the configured period endpoints. Missing cells
are rejected; they are never forward-filled. Returns are simple daily
close-to-close percentage changes. A missing *entire trading session* cannot be
identified without a separate exchange calendar.

The full-history risk report uses **constant portfolio weights, rebalanced at
each daily close without cost**, so each day's portfolio return is the dot
product of that day's ETF returns and the fixed weights. The benchmark follows
the same convention. Volatility and **ex-post tracking error** are sample daily
standard deviations multiplied by √252. Active return is the daily portfolio
return minus benchmark return; the displayed annual active return is its
arithmetic mean times 252. Correlation uses daily returns. Risk contribution
is `w_i (Σw)_i / sqrt(wᵀΣw)` from the final 252 observed daily returns, and
its components sum to portfolio daily volatility. Concentration diagnostics
include the largest ETF weight and weight Herfindahl index. ETF allocation
weights are holdings, **not statistical factor exposures**.

Historical stress windows are COVID-19 equity selloff (2020-02-19 to
2020-03-23) and the 2022 equity/bond drawdown (2022-01-03 to 2022-10-14).
For each ETF, the scenario return is the first observed close to the last
observed close inside the inclusive window. A scenario portfolio is established
at the first close and **held without rebalancing** until the last close, making
its cumulative return linear in starting weights. The stress test and optimizer
use this same convention. This differs intentionally from the daily constant
weight convention used in the descriptive risk report.

The optimizer minimizes a configurable weighted sum of annualized **predicted
tracking error**, worst observed-window loss (floored at zero), and estimated
trading cost. Predicted tracking error uses the final 252 daily observations to
estimate covariance and is `sqrt(252 (w-b)ᵀΣ(w-b))`. The historical stress vectors
are cumulative return fractions. Costs are 5 basis points per dollar bought
*and* per dollar sold; one-way turnover is half the sum of absolute weight
changes. All three objective terms are fractions of portfolio value. The
constraints require full investment, nonnegative weights, a 60% asset cap,
25% annual predicted tracking-error cap, and 15% one-way turnover cap. An
infeasible problem raises an error; constraints are never relaxed automatically.
The simple rules strategy first repairs drifted asset-cap breaches by allocating
the excess in proportion to available cap capacity, then transfers up to five
percentage points from the largest active overweight toward SPY using the
remaining turnover allowance. An unaffordable repair raises an error. The
no-rebalance strategy makes no trade. See [robustness notes](robustness.md).

The walk-forward evaluation starts in 2023, after **both** configured stress
windows ended. Every 63 trading days, it estimates covariance using only the
preceding 252 daily returns, then decides weights *before* the first held daily
return. The historical scenario vectors were already observable at every
decision date. All strategies start from the configured weights at the first
out-of-sample decision, with no initial acquisition cost. Between decisions,
ETF holdings drift with realized returns;
transaction costs are deducted on trade dates. The no-rebalance portfolio is
left to drift throughout the out-of-sample period. All strategies share the
same dates, ETF returns, and cost schedule. Walk-forward realized tracking
error is computed from net daily active returns, so trading costs enter the
realized comparison. The stress-exposure columns assess the same *past* windows
at each decision; they are diagnostics, not future losses or forecasts.

## Limitations and review decisions

The chosen benchmark, historical windows, 252-day covariance lookback, quarterly
rebalance cadence, TE budget, asset cap, turnover cap, 5-basis-point trading
cost, and objective coefficients are **modeling assumptions for review**.
They were not calibrated to a client mandate. A 25% TE budget may be loose for
some mandates; a tighter budget can make the problem infeasible, especially
with a 60% SPY cap and a 15% turnover limit.

Yahoo Finance data may contain errors, revisions, different distribution
treatment, and incomplete or inconsistent histories. The selected ETF universe
has selection and survivorship bias. The model does not include taxes, market
impact, cash drag, financing, liquidity, intraday execution, or uncertainty in
covariance estimates. The retrospective final-date optimizer uses all data
available by the end of 2025 and must not be read as an out-of-sample result.
Walk-forward results are historical and do not guarantee future improvement.
