from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes

from adaptive_rank.experiments.common import ExhaustiveSearchResults


def load_exhaustive_search_results(results_file: Path) -> ExhaustiveSearchResults:
    "Loads the results of a run of the exhaustive rank search script from a JSON file."
    with open(results_file, "r") as f:
        results = ExhaustiveSearchResults.model_validate_json(f.read())
    return results


def plot_exhaustive_search_results(
    results: ExhaustiveSearchResults, index: int, ax: Axes
) -> None:
    # Plot the results of the first run
    ranks = np.asarray(results.ranks)
    elapsed_times = np.asarray(results.elapsed_times)
    convergences = np.asarray(results.convergences)

    not_converged = np.where(~np.asarray(convergences, dtype=np.bool))[0]
    elapsed_times[not_converged] = elapsed_times.max()

    ax.plot(ranks, elapsed_times, label="Elapsed time")

    if len(not_converged) > 0:
        ax.plot(
            ranks[not_converged],
            elapsed_times[not_converged],
            color="red",
            label="Not converged",
        )

    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.legend(loc="upper right", fontsize=15)
    ax.grid()


def main(
    first_results_file: Annotated[Path, typer.Argument()],
    second_results_file: Annotated[Path, typer.Argument()],
) -> None:
    """Plots the results of multiple runs of the exhaustive rank search script,
    in two separate subplots.
    """
    first_run_results = load_exhaustive_search_results(first_results_file)
    second_run_results = load_exhaustive_search_results(second_results_file)

    all_results: list[ExhaustiveSearchResults] = [first_run_results, second_run_results]

    fig, axes = plt.subplots(2, 1)

    for index, results in enumerate(all_results):
        plot_exhaustive_search_results(results, index, axes[index])

    fig.supxlabel("Rank of pivoted Cholesky preconditioner", fontsize=15)
    fig.supylabel("Elapsed time (seconds)", fontsize=15)

    fig.tight_layout()

    plots_directory = Path("plots")
    plots_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_directory / "combined_exhaustive_searches.pdf")


if __name__ == "__main__":
    typer.run(main)
