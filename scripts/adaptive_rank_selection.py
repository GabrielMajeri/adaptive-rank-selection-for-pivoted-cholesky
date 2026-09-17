from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy
import typer
from matplotlib.axes import Axes
from rich import print
from tqdm import tqdm

from adaptive_rank.condition_number import (
    fit_interpolation_exponent,
    interpolate_conditioning_number_estimate,
)
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
)
from adaptive_rank.solver import PreconditionedConjugateGradientSolver


def fit_interpolation_exponent_on_subset(
    vectors: np.ndarray,
    kernel_matrix_regularization_factor: float,
    preconditioner_regularization_factor: float,
    rank_step_size: int = 50,
) -> float:
    """Fits the exponent used for interpolating between trace-based and pivot-based estimates
    of the kernel matrix conditioning number, by comparing them to the eigenvalue-based estimates
    on a smaller subset of the data.
    """
    n = vectors.shape[0]

    K = rbf_kernel(vectors, vectors) + kernel_matrix_regularization_factor * np.eye(
        n, dtype=np.float64
    )

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        NumPyArrayAdapter(K),
        max_rank=n,
        regularization_factor=preconditioner_regularization_factor,
    )

    ranks: list[int] = []
    eigenvalue_estimates: list[float] = []
    pivot_estimates: list[float] = []
    trace_estimates: list[float] = []

    for rank in range(n):
        preconditioner.update_inner()

        if rank % rank_step_size != 0:
            continue

        ranks.append(rank)

        U = preconditioner._preconditioner_upper[: rank + 1, :]
        remainder_matrix = K - U.T @ U
        largest_eigenvalue = scipy.linalg.eigh(
            remainder_matrix, eigvals_only=True, subset_by_index=[n - 1, n - 1]
        )[0]
        eigenvalue_estimates.append(
            1 + largest_eigenvalue / kernel_matrix_regularization_factor
        )

        latest_pivot = preconditioner.latest_pivot

        # 1 + d_k / mu
        pivot_estimates.append(1 + latest_pivot / kernel_matrix_regularization_factor)

        # 1 + sum(d_i) / mu
        trace = np.sum(preconditioner._matrix_diagonal)
        trace_estimates.append(1 + trace / kernel_matrix_regularization_factor)

    return fit_interpolation_exponent(
        ranks,
        n,
        trace_estimates,
        pivot_estimates,
        eigenvalue_estimates,
    )


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
    interpolation_exponent_fitting_subset_size: Annotated[
        int,
        typer.Option(
            help="Size of the subset of the dataset used to fit the interpolation exponent."
        ),
    ] = 1000,
    interpolation_exponent_fitting_step_size: Annotated[
        int,
        typer.Option(
            help="Step size for the ranks used to fit the interpolation exponent."
        ),
    ] = 100,
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

    print("Fitting the interpolation exponent on a subset of the dataset...")

    start_time = perf_counter()

    subset_size = min(interpolation_exponent_fitting_subset_size, N)
    subset_indices = np.random.choice(N, size=subset_size, replace=False)
    subset_vectors = points[subset_indices]
    interpolation_exponent = fit_interpolation_exponent_on_subset(
        subset_vectors,
        kernel_matrix_regularization_factor,
        preconditioner_regularization_factor,
        interpolation_exponent_fitting_step_size,
    )

    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Interpolation exponent fitted in {duration:.4g} seconds.")

    print(f"Fitted interpolation exponent: {interpolation_exponent:.4g}")

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

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=N,
        regularization_factor=preconditioner_regularization_factor,
    )

    time_complexity_estimator = TimeComplexityEstimator()

    print("Estimating the best rank for the preconditioner...")

    best_rank = -1
    best_estimated_time = np.inf
    ranks: list[int] = []
    estimated_conditioning_numbers: list[float] = []
    estimated_numbers_of_iterations: list[int] = []
    estimated_times: list[float] = []

    ranks_iterator = range(preconditioner.max_rank)
    if use_tqdm:
        ranks_iterator = tqdm(ranks_iterator, desc="Ranks")

    for rank in ranks_iterator:
        preconditioner.update_inner()
        ranks.append(rank)

        trace_estimate = (
            1
            + float(np.sum(preconditioner.residual_matrix_diagonal))
            / kernel_matrix_regularization_factor
        )
        pivot_estimate = (
            1 + preconditioner.latest_pivot / kernel_matrix_regularization_factor
        )

        estimated_cond = interpolate_conditioning_number_estimate(
            rank, N, trace_estimate, pivot_estimate, interpolation_exponent
        )
        estimated_conditioning_numbers.append(estimated_cond)

        num_iterations = int(
            np.ceil(
                0.5
                * np.sqrt(estimated_cond)
                * np.log(initial_residual_error_norm / tolerance)
            )
        )
        estimated_numbers_of_iterations.append(num_iterations)

        estimated_time = time_complexity_estimator.estimate_time(
            N, rank, num_iterations
        )
        estimated_times.append(estimated_time)

        if estimated_time < best_estimated_time:
            best_estimated_time = estimated_time
            best_rank = rank
        elif estimated_time > best_estimated_time:
            print(f"Minimum found after evaluating rank {rank}")
            break

    end_time = perf_counter()
    duration = end_time - start_time

    print("Minimal estimated time is achieved for k =", best_rank)
    print("Minimal estimated time (number of operations):", best_estimated_time)
    print(f"Time taken to estimate the best rank: {duration:.4g} seconds")

    print("Plotting the theoretical model parameters...")
    plots_directory = Path("plots/adaptive_search") / dataset
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle("Adaptive rank selection -- Theoretical model")

    ax = cast(Axes, axes[0])
    ax.set_title("Conditioning number model")

    ax.plot(ranks, estimated_conditioning_numbers)
    ax.set_xlabel("Rank of pivoted Cholesky preconditioner")
    ax.set_ylabel("Estimated conditioning number")

    ax.set_yscale("log")

    ax.grid()

    ax = cast(Axes, axes[1])
    ax.set_title("Number of iterations model")

    ax.plot(ranks, estimated_numbers_of_iterations)

    ax.set_xlabel("Rank of pivoted Cholesky preconditioner")
    ax.set_ylabel("Estimated number of iterations")

    ax.set_yscale("log")

    ax.grid()

    ax = cast(Axes, axes[2])
    ax.set_title("Time model")

    ax.plot(ranks, estimated_times)

    ax.set_xlabel("Rank of pivoted Cholesky preconditioner")
    ax.set_ylabel("Estimated time (arbitrary units)")

    ax.grid()

    fig.tight_layout()
    fig.savefig(plots_directory / f"N_{N}.pdf")

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
        estimated_num_iterations=estimated_numbers_of_iterations,
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
