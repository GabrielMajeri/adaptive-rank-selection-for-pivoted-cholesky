from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np


class TimeComplexityEstimator:
    """Estimates the time complexity of solving a dense linear system
    using the preconditioned conjugate gradient method
    with a pivoted Cholesky preconditioner.
    """

    # k^2 N constant
    c_k_squared_N: float
    # k constant
    c_k: float
    # k^3 constant
    c_k_cubed: float
    # k N constant
    c_k_N: float
    # k^2 constant
    c_k_squared: float
    # N constant
    c_N: float
    # N^2 constant
    c_N_squared: float

    def __init__(self) -> None:
        self.c_k_squared_N = 1.0
        self.c_k = 1.0
        self.c_k_cubed = 1.0
        self.c_k_N = 1.0
        self.c_k_squared = 1.0
        self.c_N = 1.0
        self.c_N_squared = 1.0

        # Not used, since we get good results using the default constants,
        # but we can measure them if needed
        # self._measure_constants()

    def _measure_constants(self) -> None:
        c_k_squared_N_coefficients: list[float] = []
        c_k_coefficients: list[float] = []
        c_k_cubed_coefficients: list[float] = []
        c_k_N_coefficients: list[float] = []
        c_k_squared_coefficients: list[float] = []
        c_N_coefficients: list[float] = []
        c_N_squared_coefficients: list[float] = []

        for k, N in ((10, 100), (50, 200), (100, 300), (200, 300), (250, 500)):
            # k^2 N constant
            M1 = np.ones((k, k))
            M2 = np.ones((k, N))

            # Repeat the measurement 5 times to reduce variance
            durations = self._measure_operation(lambda: M1 @ M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / (k**2 * N)
            c_k_squared_N_coefficients.append(coefficient)

            # k constant
            M1 = np.ones((k, k))
            M2 = np.eye(k)

            durations = self._measure_operation(lambda: M1 + M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / k
            c_k_coefficients.append(coefficient)

            # k^3 constant
            M1 = np.eye(k)

            durations = self._measure_operation(
                lambda: np.linalg.cholesky(M1),  # noqa: B023
                num_repeats=5,
            )
            coefficient = np.mean(durations) / (k**3)
            c_k_cubed_coefficients.append(coefficient)

            # k N constant
            M1 = np.ones((k, N))
            M2 = np.ones((N, 1))

            durations = self._measure_operation(lambda: M1 @ M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / (k * N)
            c_k_N_coefficients.append(coefficient)

            # k^2 constant
            M1 = np.ones((k, k))
            M2 = np.ones((k, 1))

            durations = self._measure_operation(lambda: M1 @ M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / (k**2)
            c_k_squared_coefficients.append(coefficient)

            # N constant
            M1 = np.ones((N, 1))

            durations = self._measure_operation(lambda: np.dot(M1, M1.T), num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / N
            c_N_coefficients.append(coefficient)

            # N^2 constant
            M1 = np.ones((N, N))
            M2 = np.ones((N, 1))

            durations = self._measure_operation(lambda: M1 @ M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / N
            c_N_squared_coefficients.append(coefficient)

            # N^2 constant
            M1 = np.ones((N, N))
            M2 = np.ones((N, 1))

            durations = self._measure_operation(lambda: M1 @ M2, num_repeats=5)  # noqa: B023
            coefficient = np.mean(durations) / (N**2)
            c_N_squared_coefficients.append(coefficient)

        self.c_k_squared_N = np.max(c_k_squared_N_coefficients)
        self.c_k_cubed = np.max(c_k_cubed_coefficients)
        self.c_k_N = np.max(c_k_N_coefficients)
        self.c_k_squared = np.max(c_k_squared_coefficients)
        self.c_N = np.max(c_N_coefficients)
        self.c_N_squared = np.max(c_N_squared_coefficients)

    def _measure_operation(
        self, operation: Callable[[], Any], num_repeats: int = 5
    ) -> float:
        "Measure the time taken to perform a given operation, repeated multiple times to reduce variance."
        durations: list[float] = []
        for _ in range(num_repeats):
            start_time = perf_counter()
            _ = operation()
            end_time = perf_counter()
            duration = end_time - start_time
            durations.append(duration)
        return np.mean(durations)

    def estimate_time(
        self, system_dimension: int, rank: int, num_iterations: int
    ) -> float:
        "Estimate the time complexity needed of the algorithm based on the rank and number of iterations."
        return (
            2 * self.c_k_squared_N * (rank**2) * system_dimension
            + self.c_k * rank
            + self.c_k_cubed * rank**3
            + num_iterations
            * (
                2 * self.c_k_N * rank * system_dimension
                + 2 * self.c_k_squared * (rank**2)
                + 6 * self.c_N * system_dimension
                + self.c_N_squared * system_dimension**2
            )
        )
