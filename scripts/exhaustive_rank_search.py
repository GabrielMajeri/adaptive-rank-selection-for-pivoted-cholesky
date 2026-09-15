from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.figure import Figure
from rich import print
from tqdm import tqdm

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.experiments import warm_up_code
from adaptive_rank.experiments.common import ExhaustiveSearchResults
from adaptive_rank.interface import (
    LazyTensorKernelAdapter,
    MatrixInterface,
    NumPyArrayAdapter,
)
from adaptive_rank.kernels import rbf_kernel, rbf_kernel_keops
from adaptive_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    PivotedCholeskyPreconditioner,
)
from adaptive_rank.solver import PreconditionedConjugateGradientSolver
from adaptive_rank.types import Vector


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
    max_rank: Annotated[
        int | None,
        typer.Option(help="Rank up to which to exhaustively check solve time"),
    ] = None,
    rank_step: Annotated[
        int,
        typer.Option(
            help="Number of ranks by which to increase preconditioner's rank before again determining solver's full elapsed time"
        ),
    ] = 50,
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
    use_keops: Annotated[
        bool, typer.Option(help="Use PyKeOps for kernel matrix computation")
    ] = False,
    use_tqdm: Annotated[
        bool, typer.Option(help="Enable tqdm for interactive progress bars")
    ] = True,
    plot_only: Annotated[
        bool,
        typer.Option(
            help="Skip performing experiment and just plot results from a previous run"
        ),
    ] = False,
    hide_title: Annotated[
        bool,
        typer.Option(
            help="Hide title in plot. Useful for including figures in LaTeX documents."
        ),
    ] = False,
) -> None:
    """Tries to find the optimal rank for solving a dense linear system
    (using the conjugate gradient method together with
    the pivoted Cholesky decomposition as a preconditioner)
    by repeatedly solving the system with preconditioners of various fixed ranks.
    """
    results_directory = Path("results/exhaustive_search") / dataset

    if plot_only:
        if not num_points:
            print(
                "[red]Error: --num-points must be specified when using --plot-only[/red]"
            )
            raise typer.Exit(1)

        N = num_points

        max_rank = max_rank if max_rank is not None else N // 2

        results_path = (
            results_directory / f"N_{N}_max_k_{max_rank}_step_{rank_step}.json"
        )

        if not results_path.exists():
            print(f"[red]Error: results file '{results_path}' not found[/red]")
            raise typer.Exit(1)

        with open(results_path, "r") as file:
            results = ExhaustiveSearchResults.model_validate_json(file.read())

    else:
        print(f"Loading dataset '{dataset}'...")
        labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

        points = labeled_dataset.X_train
        b = labeled_dataset.y_train.squeeze()

        N: int = points.shape[0]
        D: int = points.shape[1]

        max_rank = max_rank if max_rank is not None else N // 2

        results_path = (
            results_directory / f"N_{N}_max_k_{max_rank}_step_{rank_step}.json"
        )

        if use_keops:
            print("Constructing kernel matrix using PyKeOps...")
            K = rbf_kernel_keops(points, points)
            K_adapted = LazyTensorKernelAdapter(
                K,
                dtype=points.dtype,
                regularization_factor=kernel_matrix_regularization_factor,
                get_row=lambda row: rbf_kernel(points[row : row + 1], points).squeeze(),
            )
        else:
            print("Constructing kernel matrix using NumPy...")
            K = rbf_kernel(
                points, points
            ) + kernel_matrix_regularization_factor * np.eye(N, dtype=np.float64)
            K_adapted = NumPyArrayAdapter(K)

        warm_up_code(N)

        ranks = list(range(0, max_rank, rank_step))
        elapsed_times: list[float] = []
        convergences: list[bool] = []
        pivots: list[float] = []

        def construct_greedily_pivoted_cholesky_preconditioner(
            matrix: MatrixInterface, rank: int
        ) -> GreedilyPivotedCholeskyPreconditioner:
            return GreedilyPivotedCholeskyPreconditioner(
                matrix, rank, preconditioner_regularization_factor, rank
            )

        if use_tqdm:
            ranks_iterator = tqdm(ranks)
        else:
            ranks_iterator = iter(ranks)

        print(
            f"Solving system using pivoted Cholesky preconditioner with every rank from 0 to {max_rank}"
        )
        for target_rank in ranks_iterator:
            elapsed_time, converged, last_pivot = (
                solve_system_using_pivoted_cholesky_preconditioner(
                    K_adapted,
                    b,
                    target_rank,
                    construct_greedily_pivoted_cholesky_preconditioner,
                    tolerance,
                    max_iterations,
                )
            )
            elapsed_times.append(elapsed_time)
            convergences.append(converged)
            pivots.append(last_pivot)

        results = ExhaustiveSearchResults(
            dataset=dataset,
            num_points=N,
            dimension=D,
            seed=seed,
            kernel_matrix_regularization_factor=kernel_matrix_regularization_factor,
            preconditioner_regularization_factor=preconditioner_regularization_factor,
            tolerance=tolerance,
            max_iterations=max_iterations,
            ranks=ranks,
            elapsed_times=elapsed_times,
            convergences=convergences,
            pivots=pivots,
        )

        print("Saving results to disk...")
        results_directory.mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as file:
            file.write(results.model_dump_json(indent=2))

    print("Plotting results...")
    plots_directory = Path("plots/exhaustive_search") / dataset
    plots_directory.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(dpi=200)

    if not hide_title:
        figure.suptitle(
            "Solving kernel ridge regression problem\n"
            "using CG with pivoted Cholesky preconditioner"
        )

    plot_results(figure, results)
    figure.tight_layout()
    figure.savefig(plots_directory / f"N_{N}_max_k_{max_rank}_step_{rank_step}.pdf")


def solve_system_using_pivoted_cholesky_preconditioner(
    A: MatrixInterface,
    b: Vector,
    rank: int,
    preconditioner_factory: Callable[
        [MatrixInterface, int], PivotedCholeskyPreconditioner
    ],
    tolerance: float,
    max_iterations: int,
) -> tuple[float, bool, float]:
    """Solves the given linear system using the conjugate gradient method,
    with the partial pivoted Cholesky factorization as a preconditioner.
    """
    start_time = perf_counter()

    preconditioner = preconditioner_factory(A, rank)
    preconditioner.compute_full()

    last_pivot = preconditioner.latest_pivot

    solver = PreconditionedConjugateGradientSolver(A, b, preconditioner, tolerance)

    solution, num_iterations = solver.solve(max_iterations)

    end_time = perf_counter()
    elapsed_time = end_time - start_time

    residual_error_norm = np.linalg.vector_norm(A @ solution - b).item()

    converged = (num_iterations < max_iterations) and (residual_error_norm <= tolerance)

    return elapsed_time, converged, last_pivot


def plot_results(figure: Figure, results: ExhaustiveSearchResults) -> None:
    ranks = np.asarray(results.ranks)
    elapsed_times = np.asarray(results.elapsed_times)
    # pivots = np.asarray(results.pivots)
    convergences = np.asarray(results.convergences)

    not_converged = np.where(~np.asarray(convergences, dtype=np.bool))[0]
    elapsed_times[not_converged] = elapsed_times.max()

    ax = figure.add_subplot(1, 1, 1)

    ax.plot(ranks, elapsed_times, label="Elapsed time")

    if len(not_converged) > 0:
        ax.plot(
            ranks[not_converged],
            elapsed_times[not_converged],
            color="red",
            label="Not converged",
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Elapsed time")

    ax.legend()
    ax.grid()

    # ax = figure.add_subplot(1, 2, 2)

    # ax.plot(ranks, pivots, color="orange", label="Pivot value")

    # ax.set_xlabel("Rank")
    # ax.set_ylabel("Pivot norm")

    # ax.legend()
    # ax.grid()


if __name__ == "__main__":
    typer.run(main)
