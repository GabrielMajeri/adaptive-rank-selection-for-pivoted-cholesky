from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import typer
from rich import print

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.experiments import warm_up_code
from adaptive_rank.experiments.common import SingleRankSolveResults
from adaptive_rank.interface import NumPyArrayAdapter
from adaptive_rank.kernels import rbf_kernel
from adaptive_rank.preconditioner import GreedilyPivotedCholeskyPreconditioner
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
    rank: Annotated[
        int | None,
        typer.Option(
            help="Rank for the pivoted Cholesky preconditioner. If `None`, will be equal to one tenth of the system size."
        ),
    ] = None,
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
    ] = 1000,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="If True, will overwrite existing results file. If False, will abort if results file already exists."
        ),
    ] = False,
) -> None:
    """Solves a kernel ridge regression (KRR) problem with the preconditioned conjugate gradient method,
    using the low-rank approximation provided by the pivoted Cholesky decomposition as a preconditioner.
    """
    if all:
        num_points = None

    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    N: int = num_points if num_points is not None else labeled_dataset.X_train.shape[0]
    D: int = labeled_dataset.X_train.shape[1]

    print(
        f"Dataset '{dataset}' loaded. Using N = {N} points, each of dimension D = {D}"
    )

    if rank is None:
        rank = N // 10
        print(f"Rank not specified. Using rank = {rank}.")

    results_directory = Path("results/single_rank_solve") / dataset
    results_directory.mkdir(parents=True, exist_ok=True)

    results_file_path = results_directory / f"N_{N}_k_{rank}.json"

    if results_file_path.exists() and not overwrite:
        print(f"Results file already exists at {results_file_path}.")
        typer.confirm("Do you want to overwrite it?", abort=True)

    points = labeled_dataset.X_train
    b = labeled_dataset.y_train.squeeze()

    print("Constructing kernel matrix using NumPy...")
    K = rbf_kernel(points, points) + kernel_matrix_regularization_factor * np.eye(
        N, dtype=np.float64
    )
    K_adapted = NumPyArrayAdapter(K)

    print("Warming up code...")
    warm_up_code(N)

    start_time = perf_counter()

    print(f"Computing pivoted Cholesky preconditioner of rank {rank}...")
    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted, rank, preconditioner_regularization_factor, rank
    )
    preconditioner.compute_full()

    print("Solving linear system using preconditioned conjugate gradient method...")
    solver = PreconditionedConjugateGradientSolver(
        K_adapted, b, preconditioner, tolerance
    )

    solution, num_iterations = solver.solve(max_iterations)

    end_time = perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    residual_error_norm = np.linalg.vector_norm(K_adapted @ solution - b).item()

    converged = (num_iterations < max_iterations) and (residual_error_norm <= tolerance)
    print(f"Solver converged: {converged}")

    results = SingleRankSolveResults(
        dataset=dataset,
        num_points=N,
        dimension=D,
        seed=seed,
        kernel_matrix_regularization_factor=kernel_matrix_regularization_factor,
        preconditioner_regularization_factor=preconditioner_regularization_factor,
        rank=rank,
        tolerance=tolerance,
        max_iterations=max_iterations,
        num_iterations=num_iterations,
        residual_error_norm=residual_error_norm,
        converged=converged,
        real_elapsed_time=elapsed_time,
    )

    print("Saving results to disk...")
    with open(results_file_path, "w") as results_file:
        results_file.write(results.model_dump_json(indent=2))


if __name__ == "__main__":
    typer.run(main)
