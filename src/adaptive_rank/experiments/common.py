import numpy as np
from pydantic import BaseModel


class ExperimentResults(BaseModel):
    "Base class for experiment results."

    dataset: str
    num_points: int
    dimension: int
    seed: int | None

    kernel_function: str
    kernel_matrix_regularization_factor: float

    pivoting_strategy: str
    preconditioner_regularization_factor: float

    tolerance: float
    max_iterations: int


class SingleRankSolveResults(ExperimentResults):
    "Data class to store the results of the single rank solve experiment."

    rank: int
    num_iterations: int
    residual_error_norm: float
    converged: bool
    real_elapsed_time: float


class ExhaustiveSearchResults(ExperimentResults):
    "Data class to store the results of the exhaustive search experiment."

    ranks: list[int]
    num_iterations: list[int]
    residual_error_norms: list[float]
    elapsed_times: list[float]
    convergences: list[bool]
    pivots: list[float]


class AdaptiveRankSelectionResults(ExperimentResults):
    "Data class to store the results of the adaptive rank selection experiment."

    interpolation_exponent: float
    iteration_count_scaling_constant: float

    ranks: list[int]

    initial_residual_error_norm: float
    estimated_conditioning_numbers: list[float]
    estimated_num_iterations: list[int]

    estimated_times: list[float]
    best_rank: int
    best_estimated_time: float

    num_iterations: int
    residual_error_norm: float
    converged: bool
    real_elapsed_time: float


def conjugate_gradient_iterations_bound(
    conditioning_number: float,
    initial_residual_error_norm: float,
    tolerance: float,
    scaling_constant: float = 1.0,
) -> int:
    """Computes an upper bound on the number of iterations required for the conjugate gradient solver
    to converge to a solution with residual error norm below the specified tolerance,
    given an estimate of the conditioning number of the system matrix.
    """
    return int(
        np.ceil(
            scaling_constant
            * 0.5
            * np.sqrt(conditioning_number)
            * np.log(initial_residual_error_norm / tolerance)
        )
    )
