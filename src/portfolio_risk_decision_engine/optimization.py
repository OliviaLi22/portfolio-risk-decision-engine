"""Convex rebalancing with explicit ex-ante risk and trading constraints."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .config import OptimizationConfig
from .risk import predicted_tracking_error


@dataclass(frozen=True)
class RebalanceResult:
    weights: np.ndarray
    predicted_tracking_error: float
    worst_stress_loss: float
    turnover: float
    transaction_cost: float
    status: str


def turnover(current: np.ndarray, target: np.ndarray) -> float:
    """One-way turnover is half the absolute-weight change."""
    return float(np.abs(np.asarray(target) - np.asarray(current)).sum() / 2)


def trading_cost(current: np.ndarray, target: np.ndarray, cost_bps: float) -> float:
    """Charge both buys and sells at cost_bps per dollar traded."""
    return float(
        np.abs(np.asarray(target) - np.asarray(current)).sum() * cost_bps / 10_000
    )


def evaluate_weights(
    current: np.ndarray,
    target: np.ndarray,
    benchmark: np.ndarray,
    covariance: np.ndarray,
    scenarios: np.ndarray,
    config: OptimizationConfig,
    annualization_days: int = 252,
    status: str = "evaluated",
) -> RebalanceResult:
    target = np.asarray(target, dtype=float)
    return RebalanceResult(
        weights=target,
        predicted_tracking_error=predicted_tracking_error(
            target, benchmark, covariance, annualization_days
        ),
        worst_stress_loss=float(np.maximum(-(scenarios @ target), 0).max()),
        turnover=turnover(current, target),
        transaction_cost=trading_cost(current, target, config.transaction_cost_bps),
        status=status,
    )


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("covariance must be finite and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if eigenvalues.min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T


def optimize_rebalance(
    current: np.ndarray,
    benchmark: np.ndarray,
    covariance: np.ndarray,
    scenarios: np.ndarray,
    config: OptimizationConfig,
    annualization_days: int = 252,
) -> RebalanceResult:
    """Minimize weighted predicted TE, worst observed stress loss, and trading cost.

    All terms are dimensionless return fractions. TE uses a daily covariance
    estimate and sqrt(annualization_days). Scenarios are cumulative buy-and-hold
    ETF returns. The problem is convex and DCP compliant.
    """
    current = np.asarray(current, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    n = len(current)
    if current.shape != (n,) or benchmark.shape != (n,) or covariance.shape != (n, n):
        raise ValueError("weight and covariance dimensions do not match")
    if scenarios.ndim != 2 or scenarios.shape[1] != n or scenarios.shape[0] < 1:
        raise ValueError("scenarios must have at least one row and match assets")
    if not np.isfinite(scenarios).all() or not np.isfinite(current).all():
        raise ValueError("weights and scenarios must be finite")
    if (current < -1e-10).any() or not np.isclose(current.sum(), 1, atol=1e-8):
        raise ValueError("current weights must be long-only and sum to 1")
    if (benchmark < -1e-10).any() or not np.isclose(benchmark.sum(), 1, atol=1e-8):
        raise ValueError("benchmark weights must be long-only and sum to 1")
    factor = _covariance_factor(covariance)
    target = cp.Variable(n)
    predicted_te = cp.norm(factor @ (target - benchmark), 2) * np.sqrt(
        annualization_days
    )
    worst_loss = cp.max(cp.hstack([0, -(scenarios @ target)]))
    abs_trade = cp.norm1(target - current)
    cost = abs_trade * config.transaction_cost_bps / 10_000
    objective = cp.Minimize(
        config.tracking_error_weight * predicted_te
        + config.stress_loss_weight * worst_loss
        + config.trading_cost_weight * cost
    )
    problem = cp.Problem(
        objective,
        [
            target >= 0,
            target <= config.max_asset_weight,
            cp.sum(target) == 1,
            abs_trade / 2 <= config.max_turnover,
            predicted_te <= config.max_tracking_error,
        ],
    )
    if not problem.is_dcp():
        raise ValueError("optimization problem is not DCP compliant")
    problem.solve(solver=cp.CLARABEL)
    if (
        problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or target.value is None
    ):
        raise ValueError(f"rebalancing infeasible or solver failed: {problem.status}")
    values = np.asarray(target.value).reshape(-1)
    result = evaluate_weights(
        current,
        values,
        benchmark,
        covariance,
        scenarios,
        config,
        annualization_days,
        status=problem.status,
    )
    tolerance = 1e-5
    if (
        values.min() < -tolerance
        or values.max() > config.max_asset_weight + tolerance
        or abs(values.sum() - 1) > tolerance
        or result.turnover > config.max_turnover + tolerance
        or result.predicted_tracking_error > config.max_tracking_error + tolerance
    ):
        raise ValueError("solver returned weights outside requested constraints")
    return result


def rules_based_reduction(
    current: np.ndarray,
    benchmark: np.ndarray,
    max_asset_weight: float,
    max_turnover: float,
) -> np.ndarray:
    """Repair drifted cap breaches, then trim up to 5pp toward the benchmark.

    Excess above the asset cap is distributed in proportion to available cap
    capacity. If that mandatory repair exceeds turnover, fail explicitly.
    """
    current = np.asarray(current, dtype=float)
    target = np.minimum(current, max_asset_weight)
    excess = float(1 - target.sum())
    if excess > 1e-10:
        capacity = np.maximum(max_asset_weight - target, 0)
        if capacity.sum() < excess - 1e-10:
            raise ValueError("rules-based asset cap is infeasible")
        target += excess * capacity / capacity.sum()
    repair_turnover = turnover(current, target)
    if repair_turnover > max_turnover + 1e-8:
        raise ValueError("rules-based cap repair exceeds the turnover budget")
    source = int(np.argmax(target - benchmark))
    recipient = int(np.argmax(benchmark))
    if source == recipient:
        return target
    shift = min(
        0.05,
        max_turnover - repair_turnover,
        target[source],
        max_asset_weight - target[recipient],
    )
    if shift > 0:
        target[source] -= shift
        target[recipient] += shift
    return target
