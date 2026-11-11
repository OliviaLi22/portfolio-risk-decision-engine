"""Walk-forward decisions cannot observe their held-return date or future returns."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk_decision_engine.config import StressWindow, load_config
from portfolio_risk_decision_engine.optimization import evaluate_weights
from portfolio_risk_decision_engine.validation import walk_forward


def test_covariance_uses_only_prior_observations():
    config = load_config(Path(__file__).parents[1] / "configs/portfolio.json")
    config = replace(
        config,
        start=date(2024, 1, 1),
        end_exclusive=date(2024, 3, 1),
        minimum_price_observations=3,
        stress_windows=(StressWindow("known", date(2024, 1, 2), date(2024, 1, 5)),),
        walk_forward_start=date(2024, 1, 23),
        lookback_days=5,
        rebalance_every_days=3,
    )
    dates = pd.bdate_range("2024-01-02", periods=35)
    rng = np.random.default_rng(42)
    daily = rng.normal(0, 0.01, (len(dates) - 1, len(config.tickers)))
    prices = pd.DataFrame(
        np.vstack(
            [np.full(len(config.tickers), 100), 100 * np.cumprod(1 + daily, axis=0)]
        ),
        index=dates,
        columns=config.tickers,
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    seen: list[np.ndarray] = []

    def spy(
        current, benchmark, covariance, scenarios, optimization, annualization_days
    ):
        seen.append(covariance.copy())
        return evaluate_weights(
            current,
            current,
            benchmark,
            covariance,
            scenarios,
            optimization,
            annualization_days,
        )

    result = walk_forward(prices, returns, config, optimizer=spy)
    first_day = result.decisions.iloc[0]["first_hold_date"]
    position = returns.index.get_loc(first_day)
    np.testing.assert_allclose(
        seen[0], returns.iloc[position - 5 : position].cov().to_numpy()
    )
    assert (result.decisions.training_end_date < result.decisions.first_hold_date).all()
    altered = returns.copy()
    altered.iloc[position + 1 :] *= 3
    seen.clear()
    walk_forward(prices, altered, config, optimizer=spy)
    np.testing.assert_allclose(
        seen[0], returns.iloc[position - 5 : position].cov().to_numpy()
    )
