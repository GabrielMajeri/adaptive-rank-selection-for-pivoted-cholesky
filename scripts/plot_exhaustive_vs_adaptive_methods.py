from bisect import bisect
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from adaptive_rank.experiments.common import (
    AdaptiveRankSelectionResults,
    ExhaustiveSearchResults,
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
    seed: Annotated[
        int | None, typer.Option(help="Seed for random number generator")
    ] = 42,
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
        typer.Option(help="Rank up to which solve time was exhaustively checked."),
    ] = None,
    rank_step: Annotated[
        int,
        typer.Option(
            help="Number of ranks by which the preconditioner's rank was increased in the exhaustive search experiment."
        ),
    ] = 50,
) -> None:
    """Compares the results of the exhaustive search and adaptive rank selection experiments
    for a given dataset and number of points. Makes plots of the elapsed time vs. rank for both methods,
    and marks the best rank found by the adaptive method.
    """
    if dataset == "random-multivariate-normal":
        if seed is None:
            raise ValueError(
                "Seed must be specified for `random-multivariate-normal` dataset"
            )

        seed_component = f"_seed_{seed}"
    else:
        seed_component = ""

    print(
        f"Comparing exhaustive search and adaptive rank selection for dataset {dataset} with {num_points} points"
    )
    print(f"Kernel function: {kernel_function.value}")
    print(f"Pivoting strategy: {pivoting_strategy.value}")
    print(f"Max rank for exhaustive search: {max_rank}")
    print(f"Rank step for exhaustive search: {rank_step}")

    max_rank = max_rank if max_rank is not None else num_points // 2

    exhaustive_search_results_path = Path(
        f"results/exhaustive_search/kernel_{kernel_function.value}/pivoting_{pivoting_strategy.value}/{dataset}/N_{num_points}{seed_component}_max_k_{max_rank}_step_{rank_step}.json"
    )
    with open(exhaustive_search_results_path, "r") as file:
        exhaustive_search_results = ExhaustiveSearchResults.model_validate_json(
            file.read()
        )

    adaptive_rank_selection_results_path = Path(
        f"results/adaptive_rank_selection/kernel_{kernel_function.value}/pivoting_{pivoting_strategy.value}/{dataset}/N_{num_points}{seed_component}.json"
    )
    with open(adaptive_rank_selection_results_path, "r") as file:
        adaptive_rank_selection_results = (
            AdaptiveRankSelectionResults.model_validate_json(file.read())
        )

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

    # Mark the empirical minimum of the elapsed times from the exhaustive search results
    empirical_minimum = int(np.argmin(elapsed_times))
    print(
        "Empirical minimum rank from exhaustive search:",
        exhaustive_search_results.ranks[empirical_minimum],
    )

    ax.scatter(
        exhaustive_search_results.ranks[empirical_minimum],
        elapsed_times[empirical_minimum],
        20**2,
        color="green",
        marker="s",
        label="Empirical minimum",
        zorder=3,
    )

    # Determine at which y value to mark the best rank found by the adaptive method
    # We will use the average of the two closest elapsed times from the exhaustive search results
    best_rank_index = bisect(
        exhaustive_search_results.ranks, adaptive_rank_selection_results.best_rank
    )
    if best_rank_index == 0:
        best_elapsed_time = elapsed_times[best_rank_index]
    elif best_rank_index == len(elapsed_times):
        best_elapsed_time = elapsed_times[best_rank_index - 1]
    else:
        best_elapsed_time = (
            elapsed_times[best_rank_index - 1] + elapsed_times[best_rank_index]
        ) / 2

    print(
        "Best rank found by adaptive method:",
        adaptive_rank_selection_results.best_rank,
    )

    ax.scatter(
        adaptive_rank_selection_results.best_rank,
        best_elapsed_time,
        30**2,
        color="orange",
        marker="*",
        label="Estimated time",
        zorder=3,
    )

    rank_difference = abs(
        adaptive_rank_selection_results.best_rank
        - exhaustive_search_results.ranks[empirical_minimum]
    )
    print(
        "Rank offset between empirical minimum and adaptive method's best rank:",
        rank_difference,
    )

    print(
        "Difference in solve time between empirical minimum and adaptive method's best rank:",
        best_elapsed_time - elapsed_times[empirical_minimum],
    )

    print(f"Relative error: {rank_difference / num_points * 100:.2f}%")

    ax.set_xlabel("Rank of pivoted Cholesky preconditioner", fontsize=15)
    ax.set_ylabel("Elapsed time (seconds)", fontsize=15)

    legend = ax.legend(loc="upper right", fontsize=15)
    # Make square and star markers smaller
    legend.legend_handles[-2]._sizes = [15**2]  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    legend.legend_handles[-1]._sizes = [15**2]  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid()

    fig.tight_layout()

    plots_directory = (
        Path("plots/exhaustive_vs_adaptive")
        / f"kernel_{kernel_function.value}"
        / f"pivoting_{pivoting_strategy.value}"
        / dataset
    )
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        plots_directory / f"comparison_{dataset}_N_{num_points}{seed_component}.pdf"
    )


if __name__ == "__main__":
    typer.run(main)
