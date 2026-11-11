"""Historical walk-forward comparison with strictly prior decision inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .optimization import (
    RebalanceResult,
    optimize_rebalance,
    rules_based_reduction,
    trading_cost,
    turnover,
)
from .risk import covariance_matrix, predicted_tracking_error, tracking_error
from .stress import historical_scenario_returns


@dataclass(frozen=True)
class WalkForwardResult:
    daily_returns: pd.DataFrame
    summary: pd.DataFrame
    decisions: pd.DataFrame


def walk_forward(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: PortfolioConfig,
    optimizer: Callable[..., RebalanceResult] = optimize_rebalance,
) -> WalkForwardResult:
    """Decide before each held return, then drift holdings until next decision.

    The first held date is the first trading day on/after walk_forward.start.
    Every covariance sample ends at the preceding trading close. Only stress
    windows completed before that close are used.
    """
    if not returns.index.equals(prices.index[1:]) or list(returns.columns) != list(
        prices.columns
    ):
        raise ValueError("returns must align with prices and ticker order")
    first = int(returns.index.searchsorted(pd.Timestamp(config.walk_forward_start)))
    if first < config.lookback_days or first >= len(returns):
        raise ValueError("insufficient trailing history or no walk-forward dates")
    benchmark = config.benchmark_vector()
    scenarios = historical_scenario_returns(prices, config.stress_windows).to_numpy()
    strategy_weights = {
        name: config.portfolio_vector().copy()
        for name in (
            "no_rebalancing",
            "rules_based",
            "optimized",
        )
    }
    gross_and_net: dict[str, list[float]] = {name: [] for name in strategy_weights}
    turnover_totals = {name: 0.0 for name in strategy_weights}
    cost_totals = {name: 0.0 for name in strategy_weights}
    stress_exposures: dict[str, list[float]] = {name: [] for name in strategy_weights}
    decisions: list[dict[str, object]] = []
    index = returns.index[first:]
    for offset, day in enumerate(index):
        position = first + offset
        costs_today = {name: 0.0 for name in strategy_weights}
        if offset % config.rebalance_every_days == 0:
            train = returns.iloc[position - config.lookback_days : position]
            if train.index.max() >= day:
                raise ValueError("look-ahead in covariance training slice")
            if any(pd.Timestamp(w.end) >= day for w in config.stress_windows):
                raise ValueError(
                    "historical stress scenario not known at decision date"
                )
            covariance = covariance_matrix(train)
            for name, current in strategy_weights.items():
                if name == "no_rebalancing":
                    target = current.copy()
                elif name == "rules_based":
                    target = rules_based_reduction(
                        current,
                        benchmark,
                        config.optimization.max_asset_weight,
                        config.optimization.max_turnover,
                    )
                else:
                    target = optimizer(
                        current,
                        benchmark,
                        covariance,
                        scenarios,
                        config.optimization,
                        config.annualization_days,
                    ).weights
                trade = turnover(current, target)
                fee = trading_cost(
                    current, target, config.optimization.transaction_cost_bps
                )
                strategy_weights[name] = target
                costs_today[name] = fee
                turnover_totals[name] += trade
                cost_totals[name] += fee
                worst_loss = float(np.maximum(-(scenarios @ target), 0).max())
                predicted_te = predicted_tracking_error(
                    target, benchmark, covariance, config.annualization_days
                )
                stress_exposures[name].append(worst_loss)
                decisions.append(
                    {
                        "strategy": name,
                        "first_hold_date": day,
                        "training_end_date": train.index[-1],
                        "turnover": trade,
                        "transaction_cost": fee,
                        "worst_historical_stress_loss": worst_loss,
                        "predicted_tracking_error": predicted_te,
                        "within_te_budget": predicted_te
                        <= config.optimization.max_tracking_error + 1e-6,
                        "within_asset_cap": bool(
                            (
                                target <= config.optimization.max_asset_weight + 1e-6
                            ).all()
                        ),
                        "within_turnover_budget": trade
                        <= config.optimization.max_turnover + 1e-6,
                    }
                )
        asset_returns = returns.iloc[position].to_numpy(dtype=float)
        for name, weights in strategy_weights.items():
            gross = float(weights @ asset_returns)
            gross_and_net[name].append((1 - costs_today[name]) * (1 + gross) - 1)
            drifted = weights * (1 + asset_returns) / (1 + gross)
            strategy_weights[name] = drifted
    daily = pd.DataFrame(gross_and_net, index=index)
    benchmark_returns = returns.loc[index].to_numpy() @ benchmark
    daily["benchmark"] = benchmark_returns
    summary: dict[str, dict[str, float]] = {}
    for name in gross_and_net:
        series = daily[name]
        summary[name] = {
            "cumulative_net_return": float((1 + series).prod() - 1),
            "realized_tracking_error": tracking_error(
                series, daily["benchmark"], config.annualization_days
            ),
            "total_one_way_turnover": turnover_totals[name],
            "total_transaction_cost": cost_totals[name],
            "mean_decision_stress_loss": float(np.mean(stress_exposures[name])),
            "worst_decision_stress_loss": float(np.max(stress_exposures[name])),
        }
    summary["benchmark"] = {
        "cumulative_net_return": float((1 + daily["benchmark"]).prod() - 1),
        "realized_tracking_error": 0.0,
        "total_one_way_turnover": 0.0,
        "total_transaction_cost": 0.0,
        "mean_decision_stress_loss": float(
            np.maximum(-(scenarios @ benchmark), 0).max()
        ),
        "worst_decision_stress_loss": float(
            np.maximum(-(scenarios @ benchmark), 0).max()
        ),
    }
    return WalkForwardResult(
        daily, pd.DataFrame.from_dict(summary, orient="index"), pd.DataFrame(decisions)
    )
