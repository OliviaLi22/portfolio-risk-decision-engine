"""Write reproducible figures, tables, and a plain-language decision summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .optimization import (
    evaluate_weights,
    optimize_rebalance,
    rules_based_reduction,
)
from .risk import (
    covariance_matrix,
    cumulative_returns,
    portfolio_returns,
    risk_contributions,
    risk_diagnostics,
)
from .stress import historical_scenario_returns, scenario_diagnostics
from .validation import WalkForwardResult, walk_forward


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_reports(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: PortfolioConfig,
    root: Path,
) -> WalkForwardResult:
    """Generate historical diagnostics and a distinct out-of-sample comparison."""
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    weights = config.portfolio_vector()
    benchmark = config.benchmark_vector()
    scenarios = historical_scenario_returns(prices, config.stress_windows)
    covariance = covariance_matrix(returns.iloc[-config.lookback_days :])
    optimized = optimize_rebalance(
        weights,
        benchmark,
        covariance,
        scenarios.to_numpy(),
        config.optimization,
        config.annualization_days,
    )
    rules = rules_based_reduction(
        weights,
        benchmark,
        config.optimization.max_asset_weight,
        config.optimization.max_turnover,
    )
    decisions = {
        "no_rebalancing": evaluate_weights(
            weights,
            weights,
            benchmark,
            covariance,
            scenarios.to_numpy(),
            config.optimization,
            config.annualization_days,
        ),
        "rules_based": evaluate_weights(
            weights,
            rules,
            benchmark,
            covariance,
            scenarios.to_numpy(),
            config.optimization,
            config.annualization_days,
        ),
        "optimized": optimized,
    }
    diagnostics = risk_diagnostics(
        returns, weights, benchmark, config.annualization_days
    )
    pd.Series(diagnostics, name="value").to_csv(
        tables / "risk_diagnostics.csv", header=True
    )
    pd.DataFrame(
        {
            "asset_weight": weights,
            "benchmark_weight": benchmark,
            "active_weight": weights - benchmark,
            "volatility_contribution": risk_contributions(weights, covariance),
        },
        index=config.tickers,
    ).to_csv(tables / "asset_exposures.csv", index_label="ticker")
    scenario_diagnostics(scenarios, weights, benchmark).to_csv(
        tables / "historical_stress.csv", index_label="scenario"
    )
    pd.DataFrame(
        {name: result.weights for name, result in decisions.items()},
        index=config.tickers,
    ).to_csv(tables / "decision_weights.csv", index_label="ticker")
    pd.DataFrame(
        {
            name: {
                "predicted_tracking_error": result.predicted_tracking_error,
                "worst_historical_stress_loss": result.worst_stress_loss,
                "one_way_turnover": result.turnover,
                "estimated_transaction_cost": result.transaction_cost,
                "within_asset_cap": bool(
                    (
                        result.weights <= config.optimization.max_asset_weight + 1e-6
                    ).all()
                ),
                "within_te_budget": result.predicted_tracking_error
                <= config.optimization.max_tracking_error + 1e-6,
                "within_turnover_budget": result.turnover
                <= config.optimization.max_turnover + 1e-6,
                "solver_status": result.status,
            }
            for name, result in decisions.items()
        }
    ).T.to_csv(tables / "decision_comparison.csv", index_label="strategy")
    walk = walk_forward(prices, returns, config)
    walk.daily_returns.to_csv(
        tables / "walk_forward_daily_returns.csv", index_label="date"
    )
    walk.summary.to_csv(tables / "walk_forward_summary.csv", index_label="strategy")
    walk.decisions.to_csv(tables / "walk_forward_decisions.csv", index=False)

    # The full-history figure uses a daily constant-mix analytical convention.
    portfolio = cumulative_returns(portfolio_returns(returns, weights))
    reference = cumulative_returns(portfolio_returns(returns, benchmark))
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + portfolio).plot(ax=ax, label="Illustrative portfolio")
    (1 + reference).plot(ax=ax, label="SPY benchmark")
    ax.set(
        title="Historical wealth index (daily constant weights)", ylabel="Growth of $1"
    )
    ax.legend()
    _save_figure(fig, figures / "cumulative_performance.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(config.tickers, weights)
    axes[0].set(title="ETF weights", ylabel="Portfolio weight")
    axes[1].bar(config.tickers, risk_contributions(weights, covariance))
    axes[1].set(
        title="Contribution to daily volatility", ylabel="Volatility contribution"
    )
    _save_figure(fig, figures / "concentration_and_risk_contribution.png")

    stress_table = scenario_diagnostics(scenarios, weights, benchmark)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(stress_table))
    ax.bar(x - 0.18, stress_table["portfolio_return"], width=0.36, label="Portfolio")
    ax.bar(x + 0.18, stress_table["benchmark_return"], width=0.36, label="Benchmark")
    ax.set_xticks(x, stress_table.index, rotation=15, ha="right")
    ax.set(ylabel="Window return", title="Observed historical stress windows")
    ax.legend()
    _save_figure(fig, figures / "historical_stress_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(config.tickers))
    ax.bar(x - 0.18, weights, width=0.36, label="Before")
    ax.bar(x + 0.18, optimized.weights, width=0.36, label="Optimized")
    ax.set_xticks(x, config.tickers)
    ax.set(ylabel="Portfolio weight", title="Retrospective rebalancing decision")
    ax.legend()
    _save_figure(fig, figures / "before_after_weights.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ("no_rebalancing", "rules_based", "optimized", "benchmark"):
        (1 + walk.daily_returns[name]).cumprod().plot(
            ax=ax, label=name.replace("_", " ")
        )
    ax.set(
        title="Walk-forward wealth index, net of modeled trading costs",
        ylabel="Growth of $1",
    )
    ax.legend()
    _save_figure(fig, figures / "walk_forward_comparison.png")

    base = walk.summary.loc["no_rebalancing"]
    opt = walk.summary.loc["optimized"]
    summary = (
        f"# Decision summary: {config.name}\n\n"
        "Illustrative portfolio only; this is not a client account or an investment recommendation.\n\n"
        "## Retrospective decision at the final sample close\n\n"
        f"Predicted tracking error: {decisions['no_rebalancing'].predicted_tracking_error:.2%} "
        f"before; {optimized.predicted_tracking_error:.2%} after. "
        f"Worst observed-window loss: {decisions['no_rebalancing'].worst_stress_loss:.2%} "
        f"before; {optimized.worst_stress_loss:.2%} after. "
        f"One-way turnover: {optimized.turnover:.2%}; estimated trading cost: "
        f"{optimized.transaction_cost:.3%}.\n\n"
        "## Walk-forward comparison (decisions from prior data only)\n\n"
        f"No-rebalance net cumulative return {base['cumulative_net_return']:.2%}; "
        f"optimized {opt['cumulative_net_return']:.2%}. "
        f"Realized tracking error {base['realized_tracking_error']:.2%} versus "
        f"{opt['realized_tracking_error']:.2%}. "
        f"Mean historical-scenario loss exposure at decision dates "
        f"{base['mean_decision_stress_loss']:.2%} versus "
        f"{opt['mean_decision_stress_loss']:.2%}. "
        f"Optimized cumulative one-way turnover {opt['total_one_way_turnover']:.2%}; "
        f"modeled trading costs {opt['total_transaction_cost']:.3%}.\n\n"
        "The full-history metrics are descriptive. The retrospective terminal decision uses "
        "the full sample; it is not an out-of-sample result. Walk-forward decisions use "
        "only trailing covariance data and historical stress windows completed beforehand. "
        "Historical scenario exposure is a fixed-window diagnostic, not a forecast. "
        "Yahoo Finance ETF data can have errors, revisions, and survivorship/selection bias. "
        "The model ignores taxes, market impact, financing, cash drag, distributions "
        "not captured by adjusted prices, and intraday execution. Neither the optimizer "
        "nor this retrospective comparison guarantees better future outcomes.\n"
    )
    (root / "decision_summary.md").write_text(summary, encoding="utf-8")
    return walk
