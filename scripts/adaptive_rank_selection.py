from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import numpy as np
import typer
from rich import print
from tqdm import tqdm

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.experiments import warm_up_code
from adaptive_rank.experiments.common import AdaptiveRankSelectionResults
from adaptive_rank.interface import LazyTensorKernelAdapter, NumPyArrayAdapter
from adaptive_rank.kernels import rbf_kernel, rbf_kernel_keops
from adaptive_rank.preconditioner import GreedilyPivotedCholeskyPreconditioner
from adaptive_rank.solver import PreconditionedConjugateGradientSolver


class TimeComplexityEstimator:
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

        self._measure_constants()

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


def main(
    dataset: Annotated[
        str, typer.Option(help="Identifier of dataset to use.")
    ] = "random-multivariate-normal",
    num_points: Annotated[
        int,
        typer.Option(
            "--num-points",
            "-N",
            help="Number of points of the dataset. Equal to the dimension of the kernel matrix.",
        ),
    ] = 1000,
    dimension: Annotated[
        int | None,
        typer.Option(
            "--dimension",
            "-D",
            help="Dimension of each point's feature vector. Only relevant for synthetic datasets.",
        ),
    ] = 16,
    seed: Annotated[int, typer.Option(help="Seed for random number generator")] = 42,
    tolerance: Annotated[
        float,
        typer.Option(
            help="Tolerance for the conjugate gradient solver. If the residual error norm is below this value, the solver is considered to have converged."
        ),
    ] = 1e-5,
    # max_iterations: Annotated[
    #     int,
    #     typer.Option(
    #         help="Maximum number of iterations for the conjugate gradient solver. If exceeded, the solver will be considered to have not converged."
    #     ),
    # ] = 1000,
    use_keops: Annotated[
        bool, typer.Option(help="Use PyKeOps for kernel matrix computation")
    ] = False,
    use_tqdm: Annotated[
        bool, typer.Option(help="Enable tqdm for interactive progress bars")
    ] = True,
) -> None:
    """Uses our method for adaptively selecting the rank of the low-rank approximation,
    in order to minimize the total elapsed time of the algorithm.
    """
    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    points = labeled_dataset.X_train
    b = labeled_dataset.y_train.squeeze()
    dimension = points.shape[1]

    regularization = 1e-5

    if use_keops:
        print("Constructing kernel matrix using PyKeOps...")
        K = rbf_kernel_keops(points, points)
        K_adapted = LazyTensorKernelAdapter(
            K,
            dtype=points.dtype,
            regularization_factor=regularization,
            get_row=lambda row: rbf_kernel(points[row : row + 1], points).squeeze(),
        )
    else:
        print("Constructing kernel matrix using NumPy...")
        K = rbf_kernel(points, points) + regularization * np.eye(
            num_points, dtype=np.float64
        )
        K_adapted = NumPyArrayAdapter(K)

    initial_residual_error = b - K_adapted @ np.ones(len(b))
    initial_residual_error_norm = np.linalg.norm(initial_residual_error)

    N = num_points

    # Warm-up code to ensure that the preconditioned conjugate gradient solver is ready for timing
    warm_up_code(N)

    start_time = perf_counter()

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=N,
        regularization_factor=regularization,
    )

    time_complexity_estimator = TimeComplexityEstimator()

    print("Estimating the best rank for the preconditioner...")

    best_rank = -1
    best_estimated_time = np.inf
    estimated_times: list[float] = []

    ranks_iterator = range(preconditioner.max_rank)
    if use_tqdm:
        ranks_iterator = tqdm(ranks_iterator, desc="Ranks")

    for rank in ranks_iterator:
        preconditioner.update_inner()

        latest_pivot = preconditioner._pivots[-1]
        estimated_cond = 1 + latest_pivot / regularization
        # print("kappa ~=", estimated_cond)

        num_iterations = int(
            np.ceil(
                0.5
                * np.sqrt(estimated_cond)
                * np.log(initial_residual_error_norm / tolerance)
            )
        )

        estimated_time = time_complexity_estimator.estimate_time(
            N, rank, num_iterations
        )
        estimated_times.append(estimated_time)

        if estimated_time < best_estimated_time:
            best_estimated_time = estimated_time
            best_rank = rank

        if estimated_time > 1.25 * best_estimated_time:
            print(f"Exiting early at rank {rank}")
            break

    end_time = perf_counter()
    duration = end_time - start_time

    print("Minimal estimated time is achieved for k =", best_rank)
    print("Minimal estimated time (number of operations):", best_estimated_time)
    print(f"Time taken to estimate the best rank: {duration:.4g} seconds")

    # Solve the problem using PCG with the preconditioner of best found rank, and measure the real elapsed time
    start_time = perf_counter()

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=best_rank,
        regularization_factor=regularization,
    )
    preconditioner.compute_full()

    solver = PreconditionedConjugateGradientSolver(
        K_adapted, b, preconditioner, tolerance
    )
    solver.solve()

    end_time = perf_counter()
    duration = end_time - start_time
    print(
        f"Time taken to solve the system (with selected preconditioner rank k = {best_rank}): {duration:.4g} seconds"
    )
    real_elapsed_time = duration

    ranks = list(range(preconditioner.max_rank))
    results = AdaptiveRankSelectionResults(
        ranks=ranks,
        estimated_times=estimated_times,
        best_rank=best_rank,
        best_estimated_time=best_estimated_time,
        real_elapsed_time=real_elapsed_time,
    )

    print("Saving results to disk...")

    results_directory = Path("results/adaptive_rank_selection") / dataset
    results_directory.mkdir(parents=True, exist_ok=True)

    results_path = results_directory / f"N_{num_points}.json"

    with open(results_path, "w") as file:
        file.write(results.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
