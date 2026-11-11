"""Uncertainty diagnostics preserve paired dates and expose infeasibility."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk_decision_engine import robustness
from portfolio_risk_decision_engine.config import load_config
from portfolio_risk_decision_engine.validation import WalkForwardResult


def test_paired_bootstrap_is_deterministic_and_preserves_strategy_differences():
    dates = pd.bdate_range("2024-01-02", periods=80)
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, len(dates))
    daily = pd.DataFrame(
        {
            "optimized": common + 0.001,
            "no_rebalancing": common,
            "benchmark": common / 2,
        },
        index=dates,
    )
    first = robustness.paired_block_bootstrap(daily, replicates=100)
    second = robustness.paired_block_bootstrap(daily, replicates=100)
    assert first == second
    bounds = first["annual_return_difference"]
    assert bounds["lower"] == pytest.approx(0.252)
    assert bounds["upper"] == pytest.approx(0.252)
    # A constant difference has no extra volatility: common rows must be resampled together.
    assert first["tracking_error_improvement"]["lower"] == pytest.approx(0, abs=1e-12)
    assert first["tracking_error_improvement"]["upper"] == pytest.approx(0, abs=1e-12)
    assert first["conditional_on_observed_decisions"] is True


def test_sensitivity_retains_infeasibility_without_changing_budgets(monkeypatch):
    config = load_config(Path(__file__).parents[1] / "configs/portfolio.json")
    dates = pd.bdate_range("2023-01-03", periods=45)
    daily = pd.DataFrame(
        {
            "optimized": np.sin(np.arange(len(dates))) * 0.01,
            "no_rebalancing": 0.001,
            "benchmark": 0.0005,
        },
        index=dates,
    )
    summary = pd.DataFrame(
        {
            "optimized": {
                "realized_tracking_error": 0.02,
                "cumulative_net_return": 0.1,
                "mean_decision_stress_loss": 0.15,
                "total_one_way_turnover": 0.3,
                "total_transaction_cost": 0.001,
            },
            "no_rebalancing": {
                "realized_tracking_error": 0.01,
                "cumulative_net_return": 0.11,
                "mean_decision_stress_loss": 0.2,
                "total_one_way_turnover": 0,
                "total_transaction_cost": 0,
            },
        }
    ).T
    decisions = pd.DataFrame(
        {"strategy": ["optimized"], "predicted_tracking_error": [0.03]}
    )
    base = WalkForwardResult(daily, summary, decisions)
    observed_budgets = []

    def fake_walk(prices, returns, variant, **kwargs):
        observed_budgets.append(variant.optimization.max_tracking_error)
        if variant.optimization.max_tracking_error == 0.02:
            raise ValueError("rebalancing infeasible or solver failed: infeasible")
        return base

    monkeypatch.setattr(robustness, "walk_forward", fake_walk)
    result = robustness.run_robustness(pd.DataFrame(), pd.DataFrame(), config, base)
    infeasible = result.sensitivity.loc[result.sensitivity.variant == "te_0.02"].iloc[0]
    assert infeasible.status == "infeasible"
    assert "infeasible" in infeasible.error
    assert np.isnan(infeasible.realized_tracking_error)
    assert 0.02 in observed_budgets
    assert config.optimization.max_tracking_error == 0.25
    assert len(result.sensitivity) == 14
    assert (
        result.sensitivity.loc[
            result.sensitivity.variant == "te_0.03", "te_budget_binding_decisions"
        ].iloc[0]
        == 1
    )
    assert len(result.annual) == 3
    assert any("0/13" in finding["detail"] for finding in result.findings)
