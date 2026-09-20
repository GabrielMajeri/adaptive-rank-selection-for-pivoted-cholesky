import re
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pydantic import ValidationError

from adaptive_rank.experiments.common import (
    AdaptiveRankSelectionResults,
    ExhaustiveSearchResults,
)

EXHAUSTIVE_RESULT_PATTERN = re.compile(
    r"N_(?P<num_points>\d+)_max_k_(?P<max_rank>\d+)_step_(?P<rank_step>\d+)\.json$"
)


def extract_result(
    exhaustive_results_path: Path,
    adaptive_results_path: Path,
) -> None:
    try:
        exhaustive_results = ExhaustiveSearchResults.model_validate_json(
            exhaustive_results_path.read_text()
        )
        adaptive_results = AdaptiveRankSelectionResults.model_validate_json(
            adaptive_results_path.read_text()
        )
    except ValidationError as err:
        print(f"Skipping {exhaustive_results_path}: invalid result data ({err.title})")
        return

    elapsed_times = np.asarray(exhaustive_results.elapsed_times, dtype=float)
    convergences = np.asarray(exhaustive_results.convergences, dtype=bool)
    elapsed_times[~convergences] = elapsed_times.max()
    empirical_minimum_index = int(np.argmin(elapsed_times))
    empirical_optimal_rank = exhaustive_results.ranks[empirical_minimum_index]
    adaptive_rank = adaptive_results.best_rank
    absolute_rank_error = abs(adaptive_rank - empirical_optimal_rank)
    relative_rank_error = absolute_rank_error / exhaustive_results.num_points * 100

    match = EXHAUSTIVE_RESULT_PATTERN.search(exhaustive_results_path.name)
    if match is None:
        raise ValueError(
            f"Unexpected exhaustive result filename: {exhaustive_results_path}"
        )

    print(f"Dataset: {exhaustive_results.dataset}")
    print(f"Number of points: {match['num_points']}")
    print(f"Maximum rank: {match['max_rank']}")
    print(f"Rank step: {match['rank_step']}")
    print(f"Kernel function: {exhaustive_results.kernel_function}")
    print(f"Pivoting strategy: {exhaustive_results.pivoting_strategy}")
    print(f"Empirically optimal rank: {empirical_optimal_rank}")
    print(f"Adaptively selected rank: {adaptive_rank}")
    print(f"Absolute rank error: {absolute_rank_error}")
    print(f"Relative rank error: {relative_rank_error:.2f}%")
    print()


def main(
    results_directory: Annotated[
        Path,
        typer.Option(help="Directory containing the exhaustive and adaptive results."),
    ] = Path("results"),
) -> None:
    """Print rank-selection results for every completed experiment pair."""
    exhaustive_directory = results_directory / "exhaustive_search"

    for exhaustive_results_path in sorted(
        exhaustive_directory.glob("kernel_*/pivoting_*/*/N_*_max_k_*_step_*.json")
    ):
        relative_path = exhaustive_results_path.relative_to(exhaustive_directory)
        kernel_directory, pivoting_directory, dataset, _ = relative_path.parts
        match = EXHAUSTIVE_RESULT_PATTERN.search(exhaustive_results_path.name)
        if match is None:
            continue

        adaptive_results_path = (
            results_directory
            / "adaptive_rank_selection"
            / kernel_directory
            / pivoting_directory
            / dataset
            / f"N_{match['num_points']}.json"
        )

        if not adaptive_results_path.exists():
            print(f"Skipping {exhaustive_results_path}: adaptive result is missing")
            continue

        extract_result(exhaustive_results_path, adaptive_results_path)


if __name__ == "__main__":
    typer.run(main)
