"""Validated, explicit assumptions for the illustrative portfolio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class OptimizationConfig:
    max_tracking_error: float
    max_asset_weight: float
    max_turnover: float
    transaction_cost_bps: float
    tracking_error_weight: float
    stress_loss_weight: float
    trading_cost_weight: float


@dataclass(frozen=True)
class PortfolioConfig:
    name: str
    illustrative: bool
    weights: dict[str, float]
    benchmark_weights: dict[str, float]
    start: date
    end_exclusive: date
    annualization_days: int
    minimum_price_observations: int
    optimization: OptimizationConfig
    stress_windows: tuple[StressWindow, ...]
    walk_forward_start: date
    lookback_days: int
    rebalance_every_days: int

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.weights, *self.benchmark_weights)))

    def portfolio_vector(self) -> np.ndarray:
        return np.array([self.weights.get(t, 0.0) for t in self.tickers])

    def benchmark_vector(self) -> np.ndarray:
        return np.array([self.benchmark_weights.get(t, 0.0) for t in self.tickers])


def _weights(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a nonempty ticker-to-weight object")
    result: dict[str, float] = {}
    for ticker, weight in value.items():
        if (
            not isinstance(ticker, str)
            or not ticker.strip()
            or ticker != ticker.strip()
        ):
            raise ValueError(f"{label} has an invalid ticker")
        if ticker in result:
            raise ValueError(f"{label} has duplicate ticker {ticker}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"{label}[{ticker}] must be numeric")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"{label}[{ticker}] must be finite and nonnegative")
        result[ticker] = float(weight)
    if not np.isclose(sum(result.values()), 1.0, atol=1e-10):
        raise ValueError(f"{label} must sum to 1")
    return result


def _date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _positive_int(value: object, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def load_config(path: str | Path) -> PortfolioConfig:
    """Load a JSON config and reject inconsistent assumptions early."""
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file, object_pairs_hook=_reject_duplicate_keys)
    weights = _weights(raw["weights"], "weights")
    benchmark = _weights(raw["benchmark_weights"], "benchmark_weights")
    start = _date(raw["start"], "start")
    end = _date(raw["end_exclusive"], "end_exclusive")
    if start >= end:
        raise ValueError("start must precede end_exclusive")
    windows = tuple(
        StressWindow(
            x["name"], _date(x["start"], "stress start"), _date(x["end"], "stress end")
        )
        for x in raw["stress_windows"]
    )
    if not windows or any(not (start <= w.start <= w.end < end) for w in windows):
        raise ValueError("stress windows must lie within the analysis dates")
    if len({w.name for w in windows}) != len(windows):
        raise ValueError("stress window names must be unique")
    opt = OptimizationConfig(**raw["optimization"])
    if not (
        0 < opt.max_tracking_error
        and 0 < opt.max_asset_weight <= 1
        and 0 <= opt.max_turnover <= 1
    ):
        raise ValueError("invalid optimization budgets")
    if opt.max_asset_weight * len(set(weights) | set(benchmark)) < 1:
        raise ValueError("asset cap makes full investment impossible")
    if any(
        x < 0 or not np.isfinite(x)
        for x in (
            opt.transaction_cost_bps,
            opt.tracking_error_weight,
            opt.stress_loss_weight,
            opt.trading_cost_weight,
        )
    ) or not any(
        (opt.tracking_error_weight, opt.stress_loss_weight, opt.trading_cost_weight)
    ):
        raise ValueError(
            "optimization costs and objective weights must be finite and nonnegative"
        )
    annual = _positive_int(raw["annualization_days"], "annualization_days")
    minimum = _positive_int(
        raw["minimum_price_observations"], "minimum_price_observations", 2
    )
    walk = raw["walk_forward"]
    walk_start = _date(walk["start"], "walk_forward start")
    lookback = _positive_int(walk["lookback_days"], "lookback_days", 2)
    frequency = _positive_int(walk["rebalance_every_days"], "rebalance_every_days")
    if not (start < walk_start < end):
        raise ValueError("walk_forward start must be inside analysis dates")
    if any(w.end >= walk_start for w in windows):
        raise ValueError(
            "all configured stress windows must end before walk_forward start"
        )
    return PortfolioConfig(
        raw["name"],
        bool(raw["illustrative"]),
        weights,
        benchmark,
        start,
        end,
        annual,
        minimum,
        opt,
        windows,
        walk_start,
        lookback,
        frequency,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
