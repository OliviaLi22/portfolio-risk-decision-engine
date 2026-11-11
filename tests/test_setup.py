import cvxpy as cp
import numpy as np


def test_optimization_solver():
    weights = cp.Variable(2, nonneg=True)

    target = np.array([0.6, 0.4])

    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(weights - target)),
        [cp.sum(weights) == 1],
    )

    problem.solve()

    assert problem.status == "optimal"
    np.testing.assert_allclose(weights.value, target, atol=1e-4)
