"""Export observed results and robustness evidence for the static dashboard."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .optimization import evaluate_weights, optimize_rebalance, rules_based_reduction
from .risk import covariance_matrix, risk_diagnostics
from .robustness import run_robustness
from .robustness_charts import create_robustness_figures
from .robustness_report import write_robustness_report
from .stress import historical_scenario_returns
from .validation import WalkForwardResult, walk_forward


def _records(frame: pd.DataFrame) -> list[dict]:
    """Convert tabular results without nonstandard JSON NaN values."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def export_dashboard(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: PortfolioConfig,
    results_dir: Path,
    dashboard_dir: Path,
    walk: WalkForwardResult | None = None,
) -> dict:
    """Build a self-contained data snapshot from validated local observations."""
    walk = walk if walk is not None else walk_forward(prices, returns, config)
    robustness = run_robustness(prices, returns, config, baseline_walk=walk)
    weights, benchmark = config.portfolio_vector(), config.benchmark_vector()
    covariance = covariance_matrix(returns.iloc[-config.lookback_days :])
    scenarios = historical_scenario_returns(prices, config.stress_windows)
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
    targets = {
        "no_rebalancing": weights,
        "rules_based": rules,
        "optimized": optimized.weights,
    }
    comparison = []
    for name, target in targets.items():
        result = evaluate_weights(
            weights,
            target,
            benchmark,
            covariance,
            scenarios.to_numpy(),
            config.optimization,
            config.annualization_days,
        )
        comparison.append(
            {
                "strategy": name,
                "predicted_tracking_error": result.predicted_tracking_error,
                "worst_historical_stress_loss": result.worst_stress_loss,
                "one_way_turnover": result.turnover,
                "estimated_transaction_cost": result.transaction_cost,
                "weight_hhi": float(np.square(target).sum()),
                "largest_asset_weight": float(target.max()),
            }
        )
    stress = pd.DataFrame(
        {name: scenarios.to_numpy() @ target for name, target in targets.items()},
        index=scenarios.index,
    )
    stress["benchmark"] = scenarios.to_numpy() @ benchmark
    wealth = (1 + walk.daily_returns).cumprod()
    wealth.index.name = "date"
    base = walk.summary.loc["no_rebalancing"]
    opt = walk.summary.loc["optimized"]
    te_delta = opt.realized_tracking_error - base.realized_tracking_error
    stress_delta = base.mean_decision_stress_loss - opt.mean_decision_stress_loss
    return_delta = opt.cumulative_net_return - base.cumulative_net_return
    hhi_before = float(np.square(weights).sum())
    hhi_after = float(np.square(optimized.weights).sum())
    conclusions = [
        {
            "title": "Lower historical stress exposure comes with trade-offs",
            "detail": (
                f"Optimized realized tracking error changed by {te_delta * 100:+.2f} percentage points "
                f"and net cumulative return by {return_delta * 100:+.2f} points versus no rebalancing. "
                f"Mean exposure to the selected past stress losses fell by {stress_delta * 100:.2f} points."
            ),
            "tone": "caution",
        },
        {
            "title": "Concentration must be evaluated separately",
            "detail": f"The final-date decision changes weight HHI from {hhi_before:.3f} to {hhi_after:.3f}. A per-asset cap alone does not ensure diversification.",
            "tone": "caution" if hhi_after > hhi_before else "neutral",
        },
    ]
    completed = robustness.sensitivity.loc[robustness.sensitivity.status == "completed"]
    lower_te = int((completed.te_improvement_vs_no_rebalancing > 0).sum())
    conclusions.append(
        {
            "title": "The conclusion depends on the modeling choices",
            "detail": f"Only {lower_te} of {len(completed)} completed sensitivity variants lowers realized tracking error. The grid explores assumptions; it is not a probability of future success.",
            "tone": "caution",
        }
    )
    payload = {
        "status": "ready",
        "meta": {
            "name": config.name,
            "source": "Yahoo Finance · auto-adjusted ETF closes",
            "price_start": prices.index[0].date().isoformat(),
            "price_end": prices.index[-1].date().isoformat(),
            "oos_start": walk.daily_returns.index[0].date().isoformat(),
            "oos_end": walk.daily_returns.index[-1].date().isoformat(),
            "observations": len(prices),
            "generated_at": datetime.now(UTC).isoformat(),
            "observation_fingerprint": sha256(prices.to_csv().encode()).hexdigest(),
        },
        "config": {
            "tickers": list(config.tickers),
            "initial_weights": weights.tolist(),
            "benchmark_weights": benchmark.tolist(),
            "max_tracking_error": config.optimization.max_tracking_error,
            "max_asset_weight": config.optimization.max_asset_weight,
            "max_turnover": config.optimization.max_turnover,
            "transaction_cost_bps": config.optimization.transaction_cost_bps,
            "lookback_days": config.lookback_days,
            "rebalance_every_days": config.rebalance_every_days,
            "tracking_error_weight": config.optimization.tracking_error_weight,
            "stress_loss_weight": config.optimization.stress_loss_weight,
            "trading_cost_weight": config.optimization.trading_cost_weight,
            "annualization_days": config.annualization_days,
            "stress_windows": [
                {"name": w.name, "start": w.start.isoformat(), "end": w.end.isoformat()}
                for w in config.stress_windows
            ],
        },
        "diagnostics": risk_diagnostics(
            returns, weights, benchmark, config.annualization_days
        ),
        "terminal": {
            "comparison": comparison,
            "weights": _records(
                pd.DataFrame(targets, index=config.tickers)
                .rename_axis("ticker")
                .reset_index()
            ),
            "stress": _records(stress.rename_axis("scenario").reset_index()),
        },
        "walk_forward": {
            "summary": _records(walk.summary.rename_axis("strategy").reset_index()),
            "wealth": _records(wealth.reset_index()),
        },
        "robustness": {
            "sensitivity": _records(robustness.sensitivity),
            "bootstrap": robustness.bootstrap,
            "annual": _records(robustness.annual),
            "findings": robustness.findings,
        },
        "conclusions": conclusions,
        "limitations": [
            "Only three out-of-sample calendar years; selected ETFs, stress windows, and objectives were chosen retrospectively.",
            "Historical scenario loss is exposure to two past windows, not observed subsequent crisis protection or a forecast.",
            "Bootstrap intervals condition on the observed strategy paths. They do not re-estimate weights or capture model selection uncertainty.",
            "The final-date allocation is a retrospective decision; its predicted risk is distinct from out-of-sample realized risk.",
            "Yahoo adjusted prices may be revised. Costs omit taxes, market impact, liquidity, and intraday execution.",
            "Full-history risk uses daily constant weights; walk-forward holdings drift between decisions. Trading budgets apply at decision dates.",
        ],
    }
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    robustness.sensitivity.to_csv(tables / "robustness_sensitivity.csv", index=False)
    robustness.annual.to_csv(tables / "robustness_annual.csv", index=False)
    (tables / "robustness_bootstrap.json").write_text(
        json.dumps(robustness.bootstrap, indent=2, allow_nan=False), encoding="utf-8"
    )
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    (dashboard_dir / "snapshot.json").write_text(serialized + "\n", encoding="utf-8")
    (dashboard_dir / "data.js").write_text(
        "window.PORTFOLIO_DATA = " + serialized + ";\n", encoding="utf-8"
    )
    figure_files = create_robustness_figures(
        walk, robustness, comparison, results_dir / "figures", config.annualization_days
    )
    write_robustness_report(payload, figure_files, results_dir)
    return payload
