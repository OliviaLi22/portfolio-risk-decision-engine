"""Deterministic numerical behavior and hard optimization constraints."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk_decision_engine.config import OptimizationConfig, StressWindow
from portfolio_risk_decision_engine.optimization import (
    optimize_rebalance,
    rules_based_reduction,
    turnover,
)
from portfolio_risk_decision_engine.risk import (
    annualized_volatility,
    maximum_drawdown,
    portfolio_returns,
    predicted_tracking_error,
    risk_contributions,
    tracking_error,
)
from portfolio_risk_decision_engine.stress import (
    historical_scenario_returns,
    scenario_diagnostics,
)


def test_tracking_error_risk_contributions_and_drawdown():
    dates = pd.bdate_range("2024-01-02", periods=4)
    returns = pd.DataFrame(
        {"A": [0.02, -0.01, 0.03, -0.02], "B": [0, 0.01, -0.01, 0]}, index=dates
    )
    portfolio = portfolio_returns(returns, np.array([0.5, 0.5]))
    benchmark = portfolio_returns(returns, np.array([1.0, 0.0]))
    np.testing.assert_allclose(portfolio, returns.mean(axis=1))
    assert np.isclose(
        tracking_error(portfolio, benchmark),
        annualized_volatility(portfolio - benchmark),
    )
    covariance = returns.cov().to_numpy()
    contributions = risk_contributions(np.array([0.5, 0.5]), covariance)
    assert np.isclose(
        contributions.sum(),
        np.sqrt(np.array([0.5, 0.5]) @ covariance @ np.array([0.5, 0.5])),
    )
    assert (
        predicted_tracking_error(np.array([0.5, 0.5]), np.array([1, 0]), covariance) > 0
    )
    assert maximum_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_stress_uses_observed_first_and_last_close():
    prices = pd.DataFrame(
        {"A": [100, 90, 80], "B": [100, 100, 110]},
        index=pd.bdate_range("2024-01-02", periods=3),
    )
    windows = (StressWindow("fall", date(2024, 1, 2), date(2024, 1, 4)),)
    scenarios = historical_scenario_returns(prices, windows)
    assert scenarios.loc["fall", "A"] == pytest.approx(-0.2)
    diagnostics = scenario_diagnostics(
        scenarios, np.array([0.5, 0.5]), np.array([1, 0])
    )
    assert diagnostics.loc["fall", "portfolio_return"] == pytest.approx(-0.05)


def test_optimizer_respects_all_constraints_and_reports_cost():
    current = np.array([0.5, 0.3, 0.2])
    benchmark = np.array([1.0, 0, 0])
    covariance = np.diag([0.0001, 0.0004, 0.0002])
    scenarios = np.array([[-0.2, -0.4, 0.02], [-0.1, -0.2, -0.05]])
    config = OptimizationConfig(0.35, 0.65, 0.15, 5, 1, 0.5, 1)
    result = optimize_rebalance(current, benchmark, covariance, scenarios, config)
    assert result.status in ("optimal", "optimal_inaccurate")
    assert result.weights.min() >= -1e-6
    assert result.weights.max() <= config.max_asset_weight + 1e-6
    assert result.weights.sum() == pytest.approx(1)
    assert result.turnover <= config.max_turnover + 1e-6
    assert result.predicted_tracking_error <= config.max_tracking_error + 1e-6
    assert result.transaction_cost == pytest.approx(result.turnover * 2 * 5 / 10_000)
    infeasible = OptimizationConfig(0.0001, 0.65, 0, 5, 1, 0.5, 1)
    with pytest.raises(ValueError, match="infeasible"):
        optimize_rebalance(current, benchmark, covariance, scenarios, infeasible)


def test_rules_repair_cap_drift_without_exceeding_turnover():
    current = np.array([0.67, 0.20, 0.13])
    benchmark = np.array([1.0, 0.0, 0.0])
    target = rules_based_reduction(current, benchmark, 0.60, 0.15)
    assert target.sum() == pytest.approx(1.0)
    assert target.max() <= 0.60 + 1e-10
    assert turnover(current, target) <= 0.15
    with pytest.raises(ValueError, match="turnover budget"):
        rules_based_reduction(current, benchmark, 0.60, 0.01)
