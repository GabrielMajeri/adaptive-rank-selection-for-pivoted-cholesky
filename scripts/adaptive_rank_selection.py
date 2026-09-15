from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import typer
from rich import print
from tqdm import tqdm

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.experiments import warm_up_code
from adaptive_rank.experiments.common import AdaptiveRankSelectionResults
from adaptive_rank.interface import (
    LazyTensorKernelAdapter,
    MatrixInterface,
    NumPyArrayAdapter,
)
from adaptive_rank.kernels import rbf_kernel, rbf_kernel_keops
from adaptive_rank.model import TimeComplexityEstimator
from adaptive_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    IterativePreconditioner,
)
from adaptive_rank.solver import PreconditionedConjugateGradientSolver


def main(
    dataset: Annotated[
        str, typer.Option(help="Identifier of dataset to use.")
    ] = "random-multivariate-normal",
    num_points: Annotated[
        int | None,
        typer.Option(
            "--num-points",
            "-N",
            help="Number of points of the dataset. Equal to the dimension of the kernel matrix.",
        ),
    ] = 1000,
    all: Annotated[
        bool,
        typer.Option(
            help="Use all points of the dataset. Overrides the `num_points` option."
        ),
    ] = False,
    dimension: Annotated[
        int | None,
        typer.Option(
            "--dimension",
            "-D",
            help="Dimension of each point's feature vector. Only relevant for synthetic datasets.",
        ),
    ] = 16,
    seed: Annotated[
        int | None, typer.Option(help="Seed for random number generator")
    ] = 42,
    kernel_matrix_regularization_factor: Annotated[
        float,
        typer.Option(
            help="Regularization factor for the kernel matrix. Added to the diagonal of the kernel matrix to ensure positive definiteness and improve numerical stability."
        ),
    ] = 1e-5,
    preconditioner_regularization_factor: Annotated[
        float,
        typer.Option(
            help="Regularization factor for the preconditioner. Added to the diagonal of the preconditioner matrix to ensure positive definiteness and improve numerical stability."
        ),
    ] = 1e-5,
    tolerance: Annotated[
        float,
        typer.Option(
            help="Tolerance for the conjugate gradient solver. If the residual error norm is below this value, the solver is considered to have converged."
        ),
    ] = 1e-5,
    max_iterations: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations for the conjugate gradient solver. If exceeded, the solver will be considered to have not converged."
        ),
    ] = 5000,
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
    if all:
        num_points = None

    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    points = labeled_dataset.X_train
    b = labeled_dataset.y_train.squeeze()

    N: int = points.shape[0]
    D: int = points.shape[1]
    print(f"Dataset loaded. Using N = {N} vectors, each of dimension D = {D}")

    if use_keops:
        print("Constructing kernel matrix using PyKeOps...")
        K = rbf_kernel_keops(points, points)
        K_adapted: MatrixInterface = LazyTensorKernelAdapter(
            K,
            dtype=points.dtype,
            regularization_factor=kernel_matrix_regularization_factor,
            get_row=lambda row: rbf_kernel(points[row : row + 1], points).squeeze(),
        )
    else:
        print("Constructing kernel matrix using NumPy...")
        start_time = perf_counter()
        K = rbf_kernel(points, points) + kernel_matrix_regularization_factor * np.eye(
            N, dtype=np.float64
        )
        K_adapted = NumPyArrayAdapter(K)
        end_time = perf_counter()
        duration = end_time - start_time
        print(f"Kernel matrix constructed in {duration:.4g} seconds.")

    initial_residual_error = b - K_adapted @ np.ones(len(b))
    initial_residual_error_norm = np.linalg.norm(initial_residual_error)

    # Warm-up code to ensure that the preconditioned conjugate gradient solver is ready for timing
    warm_up_code(N)

    start_time = perf_counter()

    preconditioner: IterativePreconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=N,
        regularization_factor=preconditioner_regularization_factor,
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
        estimated_cond = 1 + latest_pivot / kernel_matrix_regularization_factor
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

    print("Solving the system using PCG with the preconditioner of best found rank...")

    # Solve the problem using PCG with the preconditioner of best found rank, and measure the real elapsed time
    start_time = perf_counter()

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=best_rank,
        regularization_factor=preconditioner_regularization_factor,
    )
    preconditioner.compute_full()

    solver = PreconditionedConjugateGradientSolver(
        K_adapted, b, preconditioner, tolerance
    )
    solution, num_iterations = solver.solve(max_iterations)

    residual_error_norm = np.linalg.vector_norm(K_adapted @ solution - b).item()
    print("Residual error norm:", residual_error_norm)
    print("Number of iterations:", num_iterations)

    converged = (num_iterations < max_iterations) and (residual_error_norm <= tolerance)
    print("Converged:", converged)

    end_time = perf_counter()
    duration = end_time - start_time
    print(
        f"Time taken to solve the system (with selected preconditioner rank k = {best_rank}): {duration:.4g} seconds"
    )
    real_elapsed_time = duration

    ranks = list(range(preconditioner.max_rank))
    results = AdaptiveRankSelectionResults(
        dataset=dataset,
        num_points=N,
        dimension=D,
        seed=seed,
        kernel_matrix_regularization_factor=kernel_matrix_regularization_factor,
        preconditioner_regularization_factor=preconditioner_regularization_factor,
        tolerance=tolerance,
        max_iterations=max_iterations,
        ranks=ranks,
        estimated_times=estimated_times,
        best_rank=best_rank,
        best_estimated_time=best_estimated_time,
        num_iterations=num_iterations,
        residual_error_norm=residual_error_norm,
        converged=converged,
        real_elapsed_time=real_elapsed_time,
    )

    print("Saving results to disk...")

    results_directory = Path("results/adaptive_rank_selection") / dataset
    results_directory.mkdir(parents=True, exist_ok=True)

    results_path = results_directory / f"N_{N}.json"

    with open(results_path, "w") as file:
        file.write(results.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
