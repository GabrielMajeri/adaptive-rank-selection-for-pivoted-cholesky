from typing import overload

import numpy as np
from scipy.optimize import curve_fit


@overload
def interpolate_conditioning_number_estimate(
    rank: int,
    system_dimension: int,
    trace_estimate: float,
    pivot_estimate: float,
    exponent: float,
) -> float: ...


@overload
def interpolate_conditioning_number_estimate(
    rank: np.ndarray,
    system_dimension: int | np.ndarray,
    trace_estimate: np.ndarray,
    pivot_estimate: np.ndarray,
    exponent: float | np.ndarray,
) -> np.ndarray: ...


def interpolate_conditioning_number_estimate(
    rank, system_dimension, trace_estimate, pivot_estimate, exponent
):
    """Interpolates between the trace and pivot estimates of the conditioning number
    of the preconditioned and regularized kernel matrix,
    based on the rank of the greedily pivoted Cholesky decomposition used to precondition it.
    """
    p = (rank / system_dimension) ** exponent
    return (1 - p) * trace_estimate + p * pivot_estimate


def fit_interpolation_exponent(
    ranks: list[int],
    system_dimension: int,
    traces_estimates: list[float],
    pivots_estimate: list[float],
    eigenvalue_estimates: list[float],
    initial_guess: float = 0.1,
) -> float:
    """Fits the optimal interpolation exponent to minimize the difference
    between the interpolated conditioning number estimates and the eigenvalue-based estimate
    (close to the ground truth).
    """
    ranks_np = np.asarray(ranks)
    trace_estimates_np = np.asarray(traces_estimates)
    pivot_estimates_np = np.asarray(pivots_estimate)

    def interpolation_function(ranks: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        # Linearly interpolate between the ranks
        interpolated_traces_estimates = np.interp(ranks, ranks_np, trace_estimates_np)
        interpolated_pivot_estimates = np.interp(ranks, ranks_np, pivot_estimates_np)
        return interpolate_conditioning_number_estimate(
            ranks_np,
            system_dimension,
            interpolated_traces_estimates,
            interpolated_pivot_estimates,
            exponents,
        )

    # Non-linear least squares fitting to find the optimal exponent
    popt, _ = curve_fit(
        interpolation_function, ranks, eigenvalue_estimates, p0=[initial_guess]
    )
    return popt[0]
