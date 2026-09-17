from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from adaptive_rank.datasets.utils import load_dataset
from adaptive_rank.kernels import rbf_kernel


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
) -> None:

    print(f"Loading dataset '{dataset}'...")
    labeled_dataset = load_dataset(dataset, num_points, 1.0, 0.0, dimension, seed)

    points = labeled_dataset.X_train

    N: int = points.shape[0]
    D: int = points.shape[1]
    print(f"Dataset loaded. Using N = {N} vectors, each of dimension D = {D}")

    print("Constructing kernel matrix using NumPy...")
    K = rbf_kernel(points, points) + kernel_matrix_regularization_factor * np.eye(
        N, dtype=np.float64
    )

    print("Computing the spectrum of the kernel matrix...")
    eigenvalues = np.linalg.eigvalsh(K)

    print("Plotting the spectrum of the kernel matrix...")

    plots_directory = Path("plots/kernel_matrix_spectrum") / dataset
    plots_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    # ax.hist(eigenvalues, bins=50, label="Kernel matrix spectrum")
    ax.plot(np.arange(len(eigenvalues)), eigenvalues, label="Kernel matrix spectrum")

    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")

    ax.set_yscale("log")

    ax.set_title("Spectrum of the kernel matrix")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(plots_directory / f"N_{N}.pdf")


if __name__ == "__main__":
    typer.run(main)
