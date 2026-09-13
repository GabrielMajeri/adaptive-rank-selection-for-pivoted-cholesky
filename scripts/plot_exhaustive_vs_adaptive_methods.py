import pickle
from bisect import bisect
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer


def main(
    dataset: Annotated[
        str, typer.Option(help="Identifier of dataset to use.")
    ] = "sgdml-benzene",
    num_points: Annotated[
        int,
        typer.Option(
            "--num-points",
            "-N",
            help="Number of points of the dataset. Equal to the dimension of the kernel matrix.",
        ),
    ] = 1000,
) -> None:
    """Compares the results of the exhaustive search and adaptive rank selection experiments
    for a given dataset and number of points. Makes plots of the elapsed time vs. rank for both methods,
    and marks the best rank found by the adaptive method.
    """
    exhaustive_search_results_path = Path(
        f"results/exhaustive_search/{dataset}/N_{num_points}.pkl"
    )
    with open(exhaustive_search_results_path, "rb") as file:
        exhaustive_search_results = pickle.load(file)

    adaptive_rank_selection_results_path = Path(
        f"results/adaptive_rank_selection/{dataset}/N_{num_points}.pkl"
    )
    with open(adaptive_rank_selection_results_path, "rb") as file:
        adaptive_rank_selection_results = pickle.load(file)

    elapsed_times = np.asarray(exhaustive_search_results.elapsed_times)

    fig, ax = plt.subplots(figsize=(8, 6))

    not_converged = np.where(
        ~np.asarray(exhaustive_search_results.convergences, dtype=np.bool)
    )[0]
    elapsed_times[not_converged] = elapsed_times.max()

    ax.plot(
        exhaustive_search_results.ranks,
        elapsed_times,
        label="Elapsed time",
    )

    # Mark points that did not converge
    not_converged = np.array(exhaustive_search_results.convergences) == False
    if len(not_converged) > 0:
        ax.plot(
            np.array(exhaustive_search_results.ranks)[not_converged],
            np.array(elapsed_times)[not_converged],
            color="red",
            label="Not converged",
        )

    best_rank_index = bisect(
        exhaustive_search_results.ranks, adaptive_rank_selection_results.best_rank
    )
    best_elapsed_time = (
        elapsed_times[best_rank_index - 1] + elapsed_times[best_rank_index]
    ) / 2

    ax.scatter(
        adaptive_rank_selection_results.best_rank,
        best_elapsed_time,
        30**2,
        color="orange",
        marker="*",
        label="Estimated time",
        zorder=3,
    )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Elapsed time")

    legend = ax.legend()
    legend.legend_handles[-1]._sizes = [10**2]  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    ax.grid()

    fig.tight_layout()

    plots_directory = Path("plots/exhaustive_vs_adaptive") / dataset
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig.savefig(plots_directory / f"comparison_{dataset}_N_{num_points}.pdf")


if __name__ == "__main__":
    typer.run(main)
