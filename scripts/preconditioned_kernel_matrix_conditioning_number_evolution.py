from pathlib import Path
from time import perf_counter
from typing import Annotated, overload

import matplotlib.pyplot as plt
import numpy as np
import scipy
import typer
from pydantic import BaseModel
from scipy.optimize import curve_fit
from tqdm import tqdm

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.interface import NumPyArrayAdapter
from adaptive_rank.kernels import rbf_kernel
from adaptive_rank.preconditioner import GreedilyPivotedCholeskyPreconditioner


class KernelMatrixConditioningNumberEstimates(BaseModel):
    ranks: list[int]
    "Ranks of the greedily pivoted Cholesky decomposition used to precondition the kernel matrix."

    eigenvalue_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed based on the largest eigenvalue of the remainder matrix."

    pivot_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed based on the latest pivot of the greedily pivoted Cholesky decomposition."

    scaled_pivot_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed based on the latest pivot of the greedily pivoted Cholesky decomposition, multiplied by the dimension of the system matrix."

    trace_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed based on the trace of the remainder matrix."

    normalized_trace_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed based on the trace of the remainder matrix, normalized by the system dimension."

    interpolated_estimates: list[float]
    "Estimates of the conditioning number of the preconditioned kernel matrix, computed by interpolating between the trace and pivot estimates, based on the rank of the greedily pivoted Cholesky decomposition used to precondition it."


@overload
def interpolate_conditioning_number_estimate(
    rank: int,
    system_dimension: int,
    trace: float,
    pivot: float,
    exponent: float,
) -> float: ...


@overload
def interpolate_conditioning_number_estimate(
    rank: np.ndarray,
    system_dimension: int | np.ndarray,
    trace: np.ndarray,
    pivot: np.ndarray,
    exponent: float | np.ndarray,
) -> np.ndarray: ...


def interpolate_conditioning_number_estimate(
    rank, system_dimension, trace, pivot, exponent
):
    """Interpolates between the trace and pivot estimates of the conditioning number
    of the preconditioned and regularized kernel matrix,
    based on the rank of the greedily pivoted Cholesky decomposition used to precondition it.
    """
    p = (rank / system_dimension) ** exponent
    return (1 - p) * trace + p * pivot


def fit_interpolation_exponent(
    ranks: list[int],
    system_dimension: int,
    traces: list[float],
    pivots: list[float],
    eigenvalue_estimates: list[float],
    initial_guess: float = 0.1,
) -> float:
    """Fits the optimal interpolation exponent to minimize the difference
    between the interpolated conditioning number estimates and the eigenvalue-based estimate
    (close to the ground truth).
    """
    ranks_np = np.asarray(ranks)
    traces_np = np.asarray(traces)
    pivots_np = np.asarray(pivots)

    def interpolation_function(ranks: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        # Linearly interpolate between the ranks
        traces = np.interp(ranks_np, ranks_np, traces_np)
        pivots = np.interp(ranks_np, ranks_np, pivots_np)
        return interpolate_conditioning_number_estimate(
            ranks_np,
            system_dimension,
            traces,
            pivots,
            exponents,
        )

    # Non-linear least squares fitting to find the optimal exponent
    popt, _ = curve_fit(
        interpolation_function, ranks, eigenvalue_estimates, p0=[initial_guess]
    )
    return popt[0]


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
    estimate_computation_step: Annotated[
        int,
        typer.Option(
            help="Step size for computing conditioning number estimates. Estimates are computed every 'estimate_computation_step' iterations of the preconditioner update."
        ),
    ] = 20,
) -> None:
    """Estimates the conditioning number of a regularized kernel matrix
    as a function of the rank of the greedily pivoted Cholesky decomposition
    used to precondition it.
    """

    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    points = labeled_dataset.X_train

    N: int = points.shape[0]
    D: int = points.shape[1]
    print(f"Dataset loaded. Using N = {N} vectors, each of dimension D = {D}")

    print("Constructing kernel matrix using NumPy...")
    start_time = perf_counter()
    K = rbf_kernel(points, points) + kernel_matrix_regularization_factor * np.eye(
        N, dtype=np.float64
    )
    K_adapted = NumPyArrayAdapter(K)
    end_time = perf_counter()
    duration = end_time - start_time
    print(f"Kernel matrix constructed in {duration:.4g} seconds.")

    print("Starting to construct greedily pivoted Cholesky preconditioner...")
    preconditioner = GreedilyPivotedCholeskyPreconditioner(
        K_adapted,
        max_rank=N,
        regularization_factor=preconditioner_regularization_factor,
    )

    ranks: list[int] = []
    eigenvalue_estimates: list[float] = []
    pivot_estimates: list[float] = []
    scaled_pivot_estimates: list[float] = []
    trace_estimates: list[float] = []
    normalized_trace_estimates: list[float] = []

    print("Computing conditioning number estimates as we go...")

    for rank in tqdm(range(N)):
        preconditioner.update_inner()

        if rank % estimate_computation_step != 0:
            continue

        ranks.append(rank)

        U = preconditioner._preconditioner_upper[: rank + 1, :]
        remainder_matrix = K - U.T @ U
        largest_eigenvalue = scipy.linalg.eigh(
            remainder_matrix, eigvals_only=True, subset_by_index=[N - 1, N - 1]
        )[0]
        eigenvalue_estimates.append(
            1 + largest_eigenvalue / kernel_matrix_regularization_factor
        )

        latest_pivot = preconditioner.latest_pivot

        # 1 + d_k / mu
        pivot_estimates.append(1 + latest_pivot / kernel_matrix_regularization_factor)

        # 1 + N d_k / mu
        scaled_pivot_estimates.append(
            1 + N * latest_pivot / kernel_matrix_regularization_factor
        )

        # 1 + sum(d_i) / mu
        trace = np.sum(preconditioner._matrix_diagonal)
        trace_estimates.append(1 + trace / kernel_matrix_regularization_factor)

        # 1 + sum(d_i) / (N * mu)
        normalized_trace_estimates.append(
            1 + trace / (N * kernel_matrix_regularization_factor)
        )

    print("Fitting interpolation exponent...")
    interpolation_exponent = fit_interpolation_exponent(
        ranks,
        N,
        trace_estimates,
        pivot_estimates,
        eigenvalue_estimates,
    )
    print(f"Optimal interpolation exponent: {interpolation_exponent:.4g}")

    # Interpolate between the trace and the pivot estimates
    interpolated_estimates: list[float] = interpolate_conditioning_number_estimate(
        np.array(ranks),
        N,
        np.array(trace_estimates),
        np.array(pivot_estimates),
        interpolation_exponent,
    ).tolist()

    print("Saving results to disk...")
    results_directory = (
        Path("results/preconditioned_kernel_matrix_conditioning_number_estimates")
        / dataset
    )
    results_directory.mkdir(parents=True, exist_ok=True)

    results = KernelMatrixConditioningNumberEstimates(
        ranks=ranks,
        eigenvalue_estimates=eigenvalue_estimates,
        pivot_estimates=pivot_estimates,
        scaled_pivot_estimates=scaled_pivot_estimates,
        trace_estimates=trace_estimates,
        normalized_trace_estimates=normalized_trace_estimates,
        interpolated_estimates=interpolated_estimates,
    )

    with open(results_directory / f"N_{N}.json", "w") as f:
        f.write(results.model_dump_json(indent=2))

    print("Plotting results")
    plots_directory = (
        Path("plots/preconditioned_kernel_matrix_conditioning_number_estimates")
        / dataset
    )
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    fig.suptitle(
        "Conditioning number estimates for kernel matrix\n"
        + f"Dataset: {dataset}, N = {N}"
    )

    ax.plot(
        ranks,
        eigenvalue_estimates,
        label="Eigenvalue-derived bound $1 + \\lambda_{max}(R_k) / \\mu$",
    )
    ax.plot(ranks, pivot_estimates, label="Pivot estimate $1 + d_k / \\mu$")
    # ax.plot(
    #     ranks, scaled_pivot_estimates, label="Scaled pivot estimate $1 + N d_k / \\mu$"
    # )
    ax.plot(ranks, trace_estimates, label="Trace estimate $1 + tr(R_k)/ \\mu$")
    # ax.plot(
    #     ranks,
    #     normalized_trace_estimates,
    #     label="Normalized trace estimate $1 + tr(R_k) / (N \\mu)$",
    # )
    ax.plot(
        ranks,
        interpolated_estimates,
        label="Interpolated estimate\n$1 + (1 - p) tr(R_k)/\\mu + p d_k / \\mu$, $p = (k/N)^{"
        + f"{interpolation_exponent:.4g}"
        + "}$",
        linestyle="--",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Conditioning number estimate")

    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_directory / f"N_{N}.pdf")


if __name__ == "__main__":
    typer.run(main)
