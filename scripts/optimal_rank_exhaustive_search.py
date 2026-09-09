import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.figure import Figure
from rich import print
from tqdm import tqdm

from optimal_rank.datasets.libsvm import (
    LibSVMDatasetKind,
    load_and_process_libsvm_dataset,
)
from optimal_rank.datasets.sgdml import load_and_process_sgdml_dataset
from optimal_rank.interface import MatrixInterface, NumPyArrayAdapter
from optimal_rank.kernels import rbf_kernel
from optimal_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    PivotedCholeskyPreconditioner,
)
from optimal_rank.solver import PreconditionedConjugateGradientSolver
from optimal_rank.types import Vector


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
    if plot_only:
        results_directory = Path("results/exhaustive_search") / dataset
        results_path = results_directory / f"N_{num_points}.pkl"
        if not results_path.exists():
            print(f"[red]Error: results file '{results_path}' not found[/red]")
            raise typer.Exit(1)

        with open(results_path, "rb") as file:
            results = pickle.load(file)  # pyright: ignore[reportAny]

        if not isinstance(results, ExhaustiveSearchResults):
            print("[red]Error: deserialized results object is of wrong type[/red]")
            raise typer.Exit(1)

    else:
        generator = np.random.default_rng(seed)

        if dataset == "random-multivariate-normal":
            if dimension is None:
                raise ValueError(
                    "Dimension must be specified for `random-multivariate-normal` dataset"
                )

            points = generator.normal(size=(num_points, dimension))

        elif dataset.startswith("libsvm"):
            dataset_identifier = dataset.split("-", 1)[1]
            labeled_dataset = load_and_process_libsvm_dataset(
                LibSVMDatasetKind.REGRESSION,
                dataset_identifier,
                max_vectors=num_points,
                train_size=1.0,
                test_size=0.0,
            )
            points = labeled_dataset.X_train

        elif dataset.startswith("sgdml"):
            dataset_identifier = dataset.split("-", 1)[1]
            labeled_dataset = load_and_process_sgdml_dataset(
                dataset_identifier,
                max_vectors=num_points,
                train_size=1.0,
                test_size=0.0,
            )
            points = labeled_dataset.X_train

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        regularization = 1e-5

        K = rbf_kernel(points, points) + regularization * np.eye(
            num_points, dtype=np.float64
        )
        K_adapted = NumPyArrayAdapter(K)
        b = generator.normal(size=num_points)

        N = num_points

        # TODO: why do we need this warm-up?
        warm_up_code(N)

        if max_rank is None:
            max_rank = N // 2

        ranks = list(range(0, max_rank, rank_step))
        elapsed_times: list[float] = []
        convergences: list[bool] = []
        pivots: list[float] = []

        preconditioner_regularization_factor = 1e-3

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

        results = ExhaustiveSearchResults(ranks, elapsed_times, convergences, pivots)

        print("Saving results to disk...")
        results_directory = Path("results/exhaustive_search")
        results_directory.mkdir(parents=True, exist_ok=True)

        results_path = results_directory / f"N_{num_points}.pkl"
        with open(results_path, "wb") as file:
            pickle.dump(results, file)

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
    figure.savefig(plots_directory / f"N_{num_points}.pdf")


def warm_up_code(dimension: int) -> None:
    """Warms-up the preconditioned conjugate gradient code by
    constructing a small preconditioner and then performing one step of the PCG method.
    """

    A = np.ones((dimension, dimension), dtype=np.float64)
    A_adapted = NumPyArrayAdapter(A)

    b = np.ones(dimension, dtype=np.float64)

    # Warm-up preconditioner computation code
    preconditioner = GreedilyPivotedCholeskyPreconditioner(A_adapted, 1, 1e-3)
    preconditioner.compute_full()

    tolerance = 1e-5

    # Warm-up solver code
    solver = PreconditionedConjugateGradientSolver(
        A_adapted, b, preconditioner, tolerance
    )

    solver.step()


@dataclass
class ExhaustiveSearchResults:
    ranks: list[int]
    elapsed_times: list[float]
    convergences: list[bool]
    pivots: list[float]


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
