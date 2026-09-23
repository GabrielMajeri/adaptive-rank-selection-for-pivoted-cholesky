import subprocess
from textwrap import dedent
from typing import Annotated

import typer

from adaptive_rank.kernels import KernelFunction
from adaptive_rank.preconditioner import PivotedCholeskyStrategy

DATASETS = (
    "random-multivariate-normal",
    "libsvm-cadata",
    "libsvm-cpusmall",
    "libsvm-YearPredictionMSD",
    "sgdml-aspirin",
    "sgdml-azobenzene",
    "sgdml-benzene",
    "sgdml-ethanol",
    "sgdml-malonaldehyde",
    "sgdml-naphthalene",
    "sgdml-paracetamol",
    "sgdml-salicylic_acid",
    "sgdml-toluene",
    "sgdml-uracil",
)


def launch_slurm_job(
    dataset: str,
    num_points: int,
    kernel_function: KernelFunction,
    pivoting_strategy: PivotedCholeskyStrategy,
    max_rank: int,
    rank_step: int,
) -> None:
    """Launch a SLURM job for running the full pipeline with the given dataset,
    number of points, kernel function, pivoting strategy, max rank, and rank step.
    """

    script = dedent(f"""
    #!/usr/bin/env bash

    #SBATCH --job-name=adaptive_rank_selection_full_pipeline_{dataset}_{num_points}_{kernel_function.value}_{pivoting_strategy.value}_{max_rank}_{rank_step}
    #SBATCH --output=logs/full_pipeline_%j.log
    #SBATCH --error=logs/full_pipeline_%j.err
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=256
    #SBATCH --mem=1T
    #SBATCH --time=24:00:00
    #SBATCH --account=acc-d-fiz-26-001
    #SBATCH --partition=cpu-wide

    set -ex

    # Disable Python output buffering to see real-time output
    export PYTHONUNBUFFERED=1

    # Run the exhaustive rank search script
    uv run scripts/exhaustive_rank_search.py \
        --dataset {dataset} \
        --num-points {num_points} \
        --kernel-function {kernel_function.value} \
        --pivoting-strategy {pivoting_strategy.value} \
        --max-rank {max_rank} \
        --rank-step {rank_step}

    # Run the adaptive rank selection method
    uv run scripts/adaptive_rank_selection.py \
        --dataset {dataset} \
        --num-points {num_points} \
        --kernel-function {kernel_function.value} \
        --pivoting-strategy {pivoting_strategy.value}

    # Plot a comparison of the exhaustive and adaptive methods
    uv run scripts/plot_exhaustive_vs_adaptive_methods.py \
        --dataset {dataset} \
        --num-points {num_points} \
        --kernel-function {kernel_function.value} \
        --pivoting-strategy {pivoting_strategy.value} \
        --max-rank {max_rank} \
        --rank-step {rank_step}
    """).strip()

    try:
        _ = subprocess.run("sbatch", input=script.encode("utf-8"), check=True)
    except subprocess.CalledProcessError as err:
        print(f"An error occurred while launching Slurm job for {dataset}: {err}")


def main(
    datasets: Annotated[
        str, typer.Option(help="Datasets to process, separated by commas.")
    ] = ",".join(DATASETS),
    num_points: Annotated[
        str,
        typer.Option(
            help="Number of points to use for the datasets, separated by commas."
        ),
    ] = "20000",
    kernel_functions: Annotated[
        str,
        typer.Option(
            help=f"Kernel functions to use, separated by commas. Options: {', '.join(k.value for k in KernelFunction)}."
        ),
    ] = "rbf",
    pivoting_strategies: Annotated[
        str,
        typer.Option(
            help=f"Pivoting strategies to use, separated by commas. Options: {', '.join(s.value for s in PivotedCholeskyStrategy)}."
        ),
    ] = ",".join(s.value for s in PivotedCholeskyStrategy),
    max_ranks: Annotated[
        str,
        typer.Option(
            help="Maximum ranks to use for the pivoted Cholesky decomposition, separated by commas. Number of options must match the number of points list."
        ),
    ] = "10000",
    rank_steps: Annotated[
        str,
        typer.Option(
            help="Rank steps to use for the pivoted Cholesky decomposition, separated by commas. Number of options must match the number of points and max ranks list."
        ),
    ] = "100",
    dry_run: Annotated[
        bool,
        typer.Option(
            help="If set, the script will only print the Slurm job scripts without submitting them."
        ),
    ] = False,
) -> None:
    for dataset in datasets.split(","):
        for num_point, max_rank, rank_step in zip(
            num_points.split(","),
            max_ranks.split(","),
            rank_steps.split(","),
            strict=True,
        ):
            if dataset == "libsvm-cpusmall" and int(num_point) > 8192:
                print(
                    "`libsvm-cpusmall` dataset has a maximum of 8192 points. Adjusting num_points to 8192."
                )
                num_point = "8192"
                if int(max_rank) > 8192:
                    print(
                        "`libsvm-cpusmall` dataset has a maximum of 8192 points. Adjusting max_rank to 4096."
                    )
                    max_rank = "4096"

            for kernel_function in kernel_functions.split(","):
                for pivoting_strategy in pivoting_strategies.split(","):
                    if dry_run:
                        print(
                            f"Dry run: would launch Slurm job for dataset={dataset}, num_points={num_point}, kernel_function={kernel_function}, pivoting_strategy={pivoting_strategy}, max_rank={max_rank}, rank_step={rank_step}"
                        )
                    else:
                        launch_slurm_job(
                            dataset=dataset,
                            num_points=int(num_point),
                            kernel_function=KernelFunction(kernel_function),
                            pivoting_strategy=PivotedCholeskyStrategy(
                                pivoting_strategy
                            ),
                            max_rank=int(max_rank),
                            rank_step=int(rank_step),
                        )


if __name__ == "__main__":
    typer.run(main)
