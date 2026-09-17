from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import typer

from adaptive_rank.experiments.common import (
    AdaptiveRankSelectionResults,
    ExhaustiveSearchResults,
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
    """Plots the real and theoretically-predicted iteration counts
    for the exhaustive and adaptive rank selection methods.

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
        / dataset
        / f"N_{N}_max_k_{max_rank}_step_{rank_step}.json"
    )

    if not exhaustive_search_results_file.exists():
        raise FileNotFoundError(
            f"Exhaustive search results file not found: {exhaustive_search_results_file}"
        )

    adaptive_rank_selection_results_file = (
        results_directory / "adaptive_rank_selection" / dataset / f"N_{N}.json"
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

    ax.plot(
        exhaustive_search_results.ranks,
        exhaustive_search_results.num_iterations,
        label="Real",
    )
    ax.plot(
        adaptive_rank_selection_results.ranks,
        adaptive_rank_selection_results.estimated_num_iterations,
        label="Estimated",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Iteration Count")

    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_directory / f"N_{N}.pdf", dpi=300)


if __name__ == "__main__":
    typer.run(main)
