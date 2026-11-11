"""Sensitivity checks and conditional sampling diagnostics for historical results."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from cvxpy.error import SolverError

from .config import OptimizationConfig, PortfolioConfig
from .optimization import RebalanceResult, optimize_rebalance
from .risk import tracking_error
from .validation import WalkForwardResult, walk_forward


@dataclass(frozen=True)
class RobustnessResult:
    sensitivity: pd.DataFrame
    bootstrap: dict[str, object]
    annual: pd.DataFrame
    findings: list[dict[str, str]]


def paired_block_bootstrap(
    daily_returns: pd.DataFrame,
    annualization_days: int = 252,
    block_days: int = 21,
    replicates: int = 1000,
    seed: int = 2026,
) -> dict[str, object]:
    """Resample aligned observed strategy paths using the same contiguous blocks.

    The resulting percentile intervals condition on the observed decisions and
    net returns. They do not refit covariance, rerun the optimizer, account for
    strategy selection, or represent a confidence interval for future results.
    """
    columns = ["optimized", "no_rebalancing", "benchmark"]
    values = daily_returns.loc[:, columns].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap needs at least two finite aligned observations")
    if block_days < 1 or replicates < 2:
        raise ValueError("block_days must be positive and replicates at least two")
    n = len(values)
    block_days = min(block_days, n)
    blocks_per_sample = int(np.ceil(n / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_days + 1, size=(replicates, blocks_per_sample))
    indices = (starts[:, :, None] + np.arange(block_days)).reshape(replicates, -1)[
        :, :n
    ]
    samples = values[indices]
    return_difference = (samples[:, :, 0] - samples[:, :, 1]).mean(
        axis=1
    ) * annualization_days
    te_improvement = (
        (samples[:, :, 1] - samples[:, :, 2]).std(axis=1, ddof=1)
        - (samples[:, :, 0] - samples[:, :, 2]).std(axis=1, ddof=1)
    ) * np.sqrt(annualization_days)

    def interval(values: np.ndarray, estimate: float) -> dict[str, float]:
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": float(estimate),
            "lower": float(lower),
            "upper": float(upper),
        }

    return {
        "method": "Paired moving-block bootstrap; 95% percentile intervals",
        "block_days": block_days,
        "replicates": replicates,
        "seed": seed,
        "observations": n,
        "annual_return_difference": interval(
            return_difference,
            float((values[:, 0] - values[:, 1]).mean() * annualization_days),
        ),
        "tracking_error_improvement": interval(
            te_improvement,
            float(
                (
                    (values[:, 1] - values[:, 2]).std(ddof=1)
                    - (values[:, 0] - values[:, 2]).std(ddof=1)
                )
                * np.sqrt(annualization_days)
            ),
        ),
        "conditional_on_observed_decisions": True,
        "caveat": "Fixed observed decision paths are resampled together. Covariance and decisions are not re-estimated; selection and future-regime uncertainty are not covered.",
    }


def _shrinkage_optimizer(
    current: np.ndarray,
    benchmark: np.ndarray,
    covariance: np.ndarray,
    scenarios: np.ndarray,
    config: OptimizationConfig,
    annualization_days: int,
) -> RebalanceResult:
    """A fixed 25% shrinkage toward the diagonal, applied only to prior inputs."""
    shrunk = 0.75 * covariance + 0.25 * np.diag(np.diag(covariance))
    return optimize_rebalance(
        current, benchmark, shrunk, scenarios, config, annualization_days
    )


def _variants(config: PortfolioConfig) -> list[tuple[str, str, PortfolioConfig]]:
    variants = [("base", "Base configuration", config)]
    for days in (126, 504):
        variants.append(
            (
                f"lookback_{days}",
                f"Covariance lookback: {days} days",
                replace(config, lookback_days=days),
            )
        )
    for bps in (0, 25):
        variants.append(
            (
                f"cost_{bps}",
                f"Trading cost: {bps} bps",
                replace(
                    config,
                    optimization=replace(config.optimization, transaction_cost_bps=bps),
                ),
            )
        )
    for weight in (0.0, 1.0):
        variants.append(
            (
                f"stress_{weight:g}",
                f"Stress objective weight: {weight:g}",
                replace(
                    config,
                    optimization=replace(
                        config.optimization, stress_loss_weight=weight
                    ),
                ),
            )
        )
    for days in (21, 126):
        variants.append(
            (
                f"cadence_{days}",
                f"Rebalance every {days} days",
                replace(config, rebalance_every_days=days),
            )
        )
    for budget in (0.02, 0.03, 0.05, 0.10):
        variants.append(
            (
                f"te_{budget:g}",
                f"Predicted TE budget: {budget:.0%}",
                replace(
                    config,
                    optimization=replace(
                        config.optimization, max_tracking_error=budget
                    ),
                ),
            )
        )
    variants.append(("shrinkage_25", "Covariance diagonal shrinkage: 25%", config))
    return variants


def _annual_results(daily: pd.DataFrame, annualization_days: int) -> pd.DataFrame:
    rows = []
    for year, sample in daily.groupby(daily.index.year):
        for strategy in daily.columns:
            rows.append(
                {
                    "year": int(year),
                    "strategy": strategy,
                    "observations": len(sample),
                    "cumulative_net_return": float((1 + sample[strategy]).prod() - 1),
                    "realized_tracking_error": tracking_error(
                        sample[strategy], sample["benchmark"], annualization_days
                    ),
                }
            )
    return pd.DataFrame(rows)


def _findings(
    sensitivity: pd.DataFrame, bootstrap: dict[str, object]
) -> list[dict[str, str]]:
    feasible = sensitivity.loc[sensitivity["status"] == "completed"]
    failed = sensitivity.loc[sensitivity["status"] != "completed"]
    improved_te = int((feasible.te_improvement_vs_no_rebalancing > 0).sum())
    improved_stress = int((feasible.stress_improvement_vs_no_rebalancing > 0).sum())
    findings = [
        {
            "title": "Tracking error across tested assumptions",
            "detail": f"Realized tracking error is lower than no rebalancing in {improved_te}/{len(feasible)} completed variants. This grid changes one assumption at a time and is not an exhaustive search.",
            "tone": "positive" if improved_te == len(feasible) else "caution",
        },
        {
            "title": "Historical stress exposure is a separate trade-off",
            "detail": f"Mean observed-window stress exposure is lower in {improved_stress}/{len(feasible)} completed variants. Lower tracking error does not by itself establish lower historical stress losses.",
            "tone": "positive" if improved_stress == len(feasible) else "caution",
        },
    ]
    bounds = bootstrap["annual_return_difference"]
    crosses_zero = bounds["lower"] <= 0 <= bounds["upper"]
    findings.append(
        {
            "title": "Return difference remains uncertain"
            if crosses_zero
            else "Return difference is conditional on the observed sample",
            "detail": f"The paired 21-day block interval for annual arithmetic net return difference is {bounds['lower']:.2%} to {bounds['upper']:.2%}. It {'includes' if crosses_zero else 'excludes'} zero, but does not cover re-estimation or strategy-selection uncertainty.",
            "tone": "caution",
        }
    )
    te_bounds = bootstrap["tracking_error_improvement"]
    findings.append(
        {
            "title": "Conditional tracking-error difference",
            "detail": f"The 95% paired-block interval for no-rebalance TE minus optimized TE is {te_bounds['lower']:.2%} to {te_bounds['upper']:.2%}. Negative values mean optimized tracking error is higher. These intervals resample fixed observed decisions.",
            "tone": "caution" if te_bounds["upper"] < 0 else "neutral",
        }
    )
    base = sensitivity.loc[sensitivity.variant == "base"].iloc[0]
    findings.append(
        {
            "title": "The configured risk budget may be slack",
            "detail": f"The base strategy's maximum predicted tracking error uses {base.max_te_utilization:.1%} of its {base.te_budget:.0%} budget; {int(base.te_budget_binding_decisions)} decisions are within one basis point of that cap. A loose cap does not imply tracking error was minimized independently of the stress objective.",
            "tone": "neutral",
        }
    )
    if len(failed):
        findings.append(
            {
                "title": "Some assumptions cannot be implemented",
                "detail": f"{len(failed)}/{len(sensitivity)} variants fail without relaxing constraints: {', '.join(failed.label)}. Failed variants are retained in the table.",
                "tone": "caution",
            }
        )
    findings.append(
        {
            "title": "Evidence is limited to one ETF universe and one holdout period",
            "detail": "The configured holdout uses earlier named stress windows. Regime changes, alternative universes, historical-window selection, and parameter tuning are outside this assessment.",
            "tone": "neutral",
        }
    )
    return findings


def run_robustness(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: PortfolioConfig,
    baseline_walk: WalkForwardResult | None = None,
) -> RobustnessResult:
    """Evaluate a predeclared one-at-a-time grid on identical held-out dates.

    Results are exploratory diagnostics, not parameter selection. All complete
    variants are compared to their own no-rebalance path under the same costs
    and dates. Infeasibility is recorded and no budget is silently changed.
    """
    base = (
        baseline_walk
        if baseline_walk is not None
        else walk_forward(prices, returns, config)
    )
    rows = []
    for key, label, variant in _variants(config):
        row: dict[str, object] = {
            "variant": key,
            "label": label,
            "status": "completed",
            "error": "",
        }
        try:
            if key == "base":
                result = base
            elif key == "shrinkage_25":
                result = walk_forward(
                    prices, returns, variant, optimizer=_shrinkage_optimizer
                )
            else:
                result = walk_forward(prices, returns, variant)
            if not result.daily_returns.index.equals(base.daily_returns.index):
                raise ValueError("sensitivity held-out dates differ from base")
            optimized = result.summary.loc["optimized"]
            reference = result.summary.loc["no_rebalancing"]
            row.update(
                {
                    "realized_tracking_error": float(optimized.realized_tracking_error),
                    "te_improvement_vs_no_rebalancing": float(
                        reference.realized_tracking_error
                        - optimized.realized_tracking_error
                    ),
                    "cumulative_net_return": float(optimized.cumulative_net_return),
                    "net_return_difference_vs_no_rebalancing": float(
                        optimized.cumulative_net_return
                        - reference.cumulative_net_return
                    ),
                    "mean_decision_stress_loss": float(
                        optimized.mean_decision_stress_loss
                    ),
                    "stress_improvement_vs_no_rebalancing": float(
                        reference.mean_decision_stress_loss
                        - optimized.mean_decision_stress_loss
                    ),
                    "total_one_way_turnover": float(optimized.total_one_way_turnover),
                    "total_transaction_cost": float(optimized.total_transaction_cost),
                    "te_budget": variant.optimization.max_tracking_error,
                }
            )
            decision_risk = result.decisions.loc[
                result.decisions.strategy == "optimized", "predicted_tracking_error"
            ]
            row["max_predicted_tracking_error"] = float(decision_risk.max())
            row["max_te_utilization"] = float(
                decision_risk.max() / variant.optimization.max_tracking_error
            )
            row["te_budget_binding_decisions"] = (
                int(
                    (
                        decision_risk >= variant.optimization.max_tracking_error - 1e-4
                    ).sum()
                )
                if key != "shrinkage_25"
                else None
            )
            row["te_prediction_basis"] = (
                "sample covariance diagnostic; optimizer uses shrunk covariance"
                if key == "shrinkage_25"
                else "optimization covariance"
            )
        except (ValueError, SolverError) as exc:
            row["status"] = (
                "infeasible" if "infeasible" in str(exc).lower() else "failed"
            )
            row["error"] = str(exc)
        rows.append(row)
    sensitivity = pd.DataFrame(rows)
    bootstrap = paired_block_bootstrap(base.daily_returns, config.annualization_days)
    return RobustnessResult(
        sensitivity,
        bootstrap,
        _annual_results(base.daily_returns, config.annualization_days),
        _findings(sensitivity, bootstrap),
    )
