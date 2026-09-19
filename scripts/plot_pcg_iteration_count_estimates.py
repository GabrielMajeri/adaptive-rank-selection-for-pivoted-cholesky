from bisect import bisect
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer
from rich import print

from adaptive_rank.experiments.common import (
    AdaptiveRankSelectionResults,
    ExhaustiveSearchResults,
    conjugate_gradient_iterations_bound,
)
from adaptive_rank.kernels import KernelFunction
from adaptive_rank.preconditioner import PivotedCholeskyStrategy


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
    kernel_function: Annotated[
        KernelFunction,
        typer.Option(
            help=f"Kernel function to use for constructing the kernel matrix. Options: {', '.join([k.value for k in KernelFunction])}"
        ),
    ] = KernelFunction.RBF,
    pivoting_strategy: Annotated[
        PivotedCholeskyStrategy,
        typer.Option(
            help=f"Preconditioner to use for the conjugate gradient solver. Options: {', '.join([p.value for p in PivotedCholeskyStrategy])}"
        ),
    ] = PivotedCholeskyStrategy.GREEDY,
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
) -> None:
    """Plots the real vs. theoretically-predicted iteration counts
    for solving a kernel ridge regression problem using the preconditioned conjugate gradient method.

    Can be used to compare the real number of iterations required by conjugate gradient
    to converge in practice, versus the theoretical Chebyshev upper-bound.
    """
    results_directory = Path("results")

    if max_rank is None:
        max_rank = num_points // 2

    N = num_points

    exhaustive_search_results_file = (
        results_directory
        / "exhaustive_search"
        / f"kernel_{kernel_function.value}"
        / f"pivoting_{pivoting_strategy.value}"
        / dataset
        / f"N_{N}_max_k_{max_rank}_step_{rank_step}.json"
    )

    if not exhaustive_search_results_file.exists():
        raise FileNotFoundError(
            f"Exhaustive search results file not found: {exhaustive_search_results_file}"
        )

    adaptive_rank_selection_results_file = (
        results_directory
        / "adaptive_rank_selection"
        / f"kernel_{kernel_function.value}"
        / f"pivoting_{pivoting_strategy.value}"
        / dataset
        / f"N_{N}.json"
    )

    if not adaptive_rank_selection_results_file.exists():
        raise FileNotFoundError(
            f"Adaptive rank selection results file not found: {adaptive_rank_selection_results_file}"
        )

    exhaustive_search_results = ExhaustiveSearchResults.model_validate_json(
        exhaustive_search_results_file.read_text()
    )

    adaptive_rank_selection_results = AdaptiveRankSelectionResults.model_validate_json(
        adaptive_rank_selection_results_file.read_text()
    )

    plots_directory = Path("plots") / "real_vs_estimated_iteration_counts" / dataset
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    exhaustive_results_max_index = bisect(
        exhaustive_search_results.ranks, adaptive_rank_selection_results.best_rank
    )

    ax.plot(
        exhaustive_search_results.ranks[:exhaustive_results_max_index],
        exhaustive_search_results.num_iterations[:exhaustive_results_max_index],
        label="Measured",
    )

    raw_num_iteration_estimates = [
        conjugate_gradient_iterations_bound(
            condition_number_estimate,
            adaptive_rank_selection_results.initial_residual_error_norm,
            adaptive_rank_selection_results.tolerance,
            scaling_constant=1,
        )
        for condition_number_estimate in adaptive_rank_selection_results.estimated_conditioning_numbers
    ]
    ax.plot(
        adaptive_rank_selection_results.ranks,
        raw_num_iteration_estimates,
        label="Theoretical estimate",
    )

    ax.plot(
        adaptive_rank_selection_results.ranks,
        adaptive_rank_selection_results.estimated_num_iterations,
        linestyle="dashed",
        label="Scaled estimate",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Rank of pivoted Cholesky preconditioner", fontsize=15)
    ax.set_ylabel("PCG iteration count", fontsize=15)

    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.grid()
    ax.legend(fontsize=15)

    fig.tight_layout()
    plot_path = plots_directory / f"N_{N}.pdf"
    fig.savefig(plot_path, dpi=300)

    print(f"Plot saved to '{plot_path}'")


if __name__ == "__main__":
    typer.run(main)
