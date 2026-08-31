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

from optimal_rank.interface import MatrixInterface, NumPyArrayAdapter
from optimal_rank.kernels import rbf_kernel
from optimal_rank.preconditioner import (
    GreedilyPivotedCholeskyPreconditioner,
    PivotedCholeskyPreconditioner,
)
from optimal_rank.solver import PreconditionedConjugateGradientSolver
from optimal_rank.types import Vector


def main(
    num_points: Annotated[
        int,
        typer.Option(
            help="Number of points of the dataset. Equal to the dimension of the kernel matrix."
        ),
    ] = 1000,
    dimension: Annotated[
        int, typer.Option(help="Dimension of each point's feature vector")
    ] = 16,
    seed: Annotated[int, typer.Option(help="Seed for random number generator")] = 42,
    plot_only: Annotated[
        bool,
        typer.Option(
            help="Skip performing experiment and just plot results from a previous run"
        ),
    ] = False,
) -> None:
    """Tries to find the optimal rank for solving a dense linear system
    (using the conjugate gradient method together with
    the pivoted Cholesky decomposition as a preconditioner)
    by repeatedly solving the system with preconditioners of various fixed ranks.
    """
    if plot_only:
        results_directory = Path("results/exhaustive_search")
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

        points = generator.normal(size=(num_points, dimension))

        regularization = 1e-5

        K = rbf_kernel(points, points) + regularization * np.eye(
            num_points, dtype=np.float64
        )
        K_adapted = NumPyArrayAdapter(K)
        b = generator.normal(size=num_points)

        N = num_points

        ranks = list(range(0, N // 2, 50))
        elapsed_times: list[float] = []
        convergences: list[bool] = []
        pivots: list[float] = []

        tolerance = 1e-3
        max_iterations = 1000
        preconditioner_regularization_factor = 1e-3

        def construct_greedily_pivoted_cholesky_preconditioner(
            matrix: MatrixInterface, rank: int
        ) -> GreedilyPivotedCholeskyPreconditioner:
            return GreedilyPivotedCholeskyPreconditioner(
                matrix, rank, preconditioner_regularization_factor, rank
            )

        for target_rank in tqdm(
            ranks, desc="Solving system using various preconditioner ranks"
        ):
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
    plots_directory = Path("plots/exhaustive_search")
    plots_directory.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(dpi=200)
    plot_results(figure, results)
    figure.savefig(plots_directory / f"N_{num_points}.pdf")


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
    ax = figure.add_subplot()

    lines1 = ax.plot(results.ranks, results.elapsed_times, label="Elapsed time")

    twin_ax = ax.twinx()
    lines2 = twin_ax.plot(
        results.ranks, results.pivots, color="orange", label="Pivot value"
    )
    twin_ax.set_ylabel("Pivot norm")

    not_converged = np.where(~np.asarray(results.convergences, dtype=np.bool))[0]
    if len(not_converged) > 0:
        ax.plot(
            results.ranks[not_converged],
            results.elapsed_times[not_converged],
            "*",
            color="red",
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Elapsed time")

    lines = lines1 + lines2
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels)

    ax.grid()


if __name__ == "__main__":
    typer.run(main)
