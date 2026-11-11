"""Observed historical stress windows and buy-and-hold scenario vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StressWindow


def historical_scenario_returns(
    prices: pd.DataFrame,
    windows: tuple[StressWindow, ...],
) -> pd.DataFrame:
    """Asset close-to-close return from first to last observed close in each window.

    Weights are set at the first close and held until the last close, making
    scenario portfolio return linear in initial weights. No observations are
    invented when a named date is a holiday.
    """
    scenarios: dict[str, pd.Series] = {}
    for window in windows:
        sample = prices.loc[window.start.isoformat() : window.end.isoformat()]
        if len(sample) < 2:
            raise ValueError(
                f"stress window {window.name} needs two price observations"
            )
        scenarios[window.name] = sample.iloc[-1] / sample.iloc[0] - 1
    return pd.DataFrame.from_dict(scenarios, orient="index").loc[:, prices.columns]


def scenario_diagnostics(
    scenarios: pd.DataFrame,
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
) -> pd.DataFrame:
    portfolio = scenarios.to_numpy() @ weights
    benchmark = scenarios.to_numpy() @ benchmark_weights
    return pd.DataFrame(
        {
            "portfolio_return": portfolio,
            "benchmark_return": benchmark,
            "active_return": portfolio - benchmark,
            "portfolio_loss": np.maximum(-portfolio, 0),
            "benchmark_loss": np.maximum(-benchmark, 0),
        },
        index=scenarios.index,
    )
