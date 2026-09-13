from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import typer
from rich import print
from tqdm import tqdm

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.experiments import warm_up_code
from adaptive_rank.experiments.common import AdaptiveRankSelectionResults
from adaptive_rank.interface import NumPyArrayAdapter
from adaptive_rank.kernels import rbf_kernel
from adaptive_rank.preconditioner import GreedilyPivotedCholeskyPreconditioner
from adaptive_rank.solver import PreconditionedConjugateGradientSolver


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
    tolerance: Annotated[
        float,
        typer.Option(
            help="Tolerance for the conjugate gradient solver. If the residual error norm is below this value, the solver is considered to have converged."
        ),
    ] = 1e-5,
    # max_iterations: Annotated[
    #     int,
    #     typer.Option(
    #         help="Maximum number of iterations for the conjugate gradient solver. If exceeded, the solver will be considered to have not converged."
    #     ),
    # ] = 1000,
    use_tqdm: Annotated[
        bool, typer.Option(help="Enable tqdm for interactive progress bars")
    ] = True,
) -> None:
    """Uses our method for adaptively selecting the rank of the low-rank approximation,
    in order to minimize the total elapsed time of the algorithm.
    """
    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    points = labeled_dataset.X_train
    b = labeled_dataset.y_train.squeeze()

    regularization = 1e-5

    K = rbf_kernel(points, points) + regularization * np.eye(
        num_points, dtype=np.float64
    )
    K_adapted = NumPyArrayAdapter(K)

    initial_residual_error = b - K_adapted @ np.ones(len(b))
    initial_residual_error_norm = np.linalg.norm(initial_residual_error)

    N = num_points

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=N,
        regularization_factor=regularization,
    )

    def compute_time_complexity(k: int, num_iterations: int) -> float:
        c_1 = 0.5
        return (
            3 * c_1 * (k**2) * N
            + (1 / 6) * k**3
            + num_iterations * (6 * N + N**2 + 2 * k * N + 2 * (k**2))
        )

    best_rank = -1
    best_estimated_time = np.inf
    estimated_times: list[float] = []

    ranks_iterator = range(preconditioner.max_rank)
    if use_tqdm:
        ranks_iterator = tqdm(ranks_iterator, desc="Ranks")

    for rank in ranks_iterator:
        preconditioner.update_inner()

        latest_pivot = preconditioner._pivots[-1]
        estimated_cond = 1 + latest_pivot / regularization
        # print("kappa ~=", estimated_cond)

        num_iterations = int(
            np.ceil(
                0.5
                * np.sqrt(estimated_cond)
                * np.log(initial_residual_error_norm / tolerance)
            )
        )

        estimated_time = compute_time_complexity(rank, num_iterations)
        estimated_times.append(estimated_time)

        if estimated_time < best_estimated_time:
            best_estimated_time = estimated_time
            best_rank = rank

        if estimated_time > 1.25 * best_estimated_time:
            print(f"Exiting early at rank {rank}")
            break

    print("Minimal estimated time is achieved for k =", best_rank)
    print("Minimal estimated time (number of operations):", best_estimated_time)

    # Warm-up code to ensure that the preconditioned conjugate gradient solver is ready for timing
    warm_up_code(N)

    # Solve the problem using PCG with the preconditioner of best found rank, and measure the real elapsed time
    start_time = perf_counter()

    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=best_rank,
        regularization_factor=regularization,
    )
    preconditioner.compute_full()

    solver = PreconditionedConjugateGradientSolver(
        K_adapted, b, preconditioner, tolerance
    )
    solver.solve()

    end_time = perf_counter()
    duration = end_time - start_time
    real_elapsed_time = duration

    ranks = list(range(preconditioner.max_rank))
    results = AdaptiveRankSelectionResults(
        ranks=ranks,
        estimated_times=estimated_times,
        best_rank=best_rank,
        best_estimated_time=best_estimated_time,
        real_elapsed_time=real_elapsed_time,
    )

    print("Saving results to disk...")

    results_directory = Path("results/adaptive_rank_selection") / dataset
    results_directory.mkdir(parents=True, exist_ok=True)

    results_path = results_directory / f"N_{num_points}.json"

    with open(results_path, "w") as file:
        file.write(results.model_dump_json())


if __name__ == "__main__":
    typer.run(main)
