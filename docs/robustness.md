# Dashboard and robustness methodology

Generate the standard reports, robustness outputs, and browser snapshot together:

```bash
uv run python -m portfolio_risk_decision_engine.run_analysis dashboard
```

The command reads validated local Parquet files. It never substitutes synthetic
history. Open `dashboard/dist/index.html` directly, or serve that directory using
`uv run python -m http.server 8000 --bind 127.0.0.1 --directory dashboard/dist`.
The dashboard uses local HTML, CSS, JavaScript, and a generated data snapshot;
it has no framework, database, CDN, or live market-data dependency.

## Evidence shown

- Descriptive full-history portfolio risk, with daily constant weights.
- The retrospective final-date optimized allocation and its predicted tracking
  error, historical stress exposure, turnover, cost, and weight concentration.
- The separate 2023–2025 walk-forward comparison, net of modeled trading costs.
- One-at-a-time changes to covariance lookback (126/252/504 observations), trading
  costs (0/5/25 bps), stress objective weight (0/0.5/1), rebalance interval
  (21/63/126 observations), predicted TE budget (2/3/5/10/25%), and a fixed 25%
  shrinkage of covariance toward its diagonal. There are 14 configurations
  including the base. Only one setting changes in each alternative.
- Annual strategy slices and paired moving-block uncertainty intervals.

The illustrated report embeds strategy comparisons, trailing realized tracking
error and drawdown, calendar-year outcomes, assumption sensitivity, a risk versus
stress exposure map, conditional bootstrap intervals, and final-date allocation
weights. Each chart includes its convention and interpretation limits. The six
robustness figures are regenerated as PNG and SVG files under `results/figures/`
when the dashboard command runs.

All variants use the same held-out dates. Infeasible configurations retain their
error and have no invented performance metrics. Changing costs affects both the
optimizer's decisions and subsequent net returns. Shrinkage is applied to prior
covariance estimates; the sensitivity table labels the sample-covariance risk
diagnostic separately from the shrunk covariance used by that variant's solver.

The rules comparator repairs any asset-cap breach caused by weight drift first,
redistributing the excess in proportion to available cap capacity. It then moves
up to five percentage points from the largest active overweight toward the
benchmark ETF, using the remaining turnover allowance. An unaffordable cap repair
fails explicitly. Baseline holdings can drift above the cap; the no-rebalance
strategy remains an unconstrained reference, and its decision flags expose this.
The rules comparator does not enforce the optimizer's tracking-error cap.

## Bootstrap interpretation

The fixed seed is 2026, with 1,000 paired resamples of 21-observation contiguous
blocks. Every resample uses the same date indices for optimized, no-rebalance,
and benchmark returns. The 2.5th and 97.5th percentiles summarize annual arithmetic
net-return difference and realized tracking-error improvement. Positive TE
improvement means no-rebalance TE exceeds optimized TE.

These intervals condition on the observed decision paths. They do not refit the
covariance model, rerun decisions in resampled histories, account for parameter
selection, or provide a forecast interval. One block length and three OOS years
are limited evidence. The grid is an exploratory diagnostic, not a validation
set for choosing whichever configuration performs best.

## Reading the conclusion

Lower historical stress exposure, lower predicted tracking error, lower realized
tracking error, and greater diversification are separate claims. The dashboard
reports their signs and magnitudes independently. A loose TE cap can remain
inactive while the stress term drives decisions. Per-asset limits do not ensure
lower Herfindahl concentration. Protection against two past stress windows does
not demonstrate subsequent crisis protection, especially over an OOS period
without a comparable selloff.

The latest measured assessment is in `results/robustness_summary.md`, with
`robustness_sensitivity.csv`, `robustness_annual.csv`, and
`robustness_bootstrap.json` under `results/tables/`. The downloadable dashboard
snapshot includes the source period, generation timestamp, settings, metrics,
and interpretation limits.
