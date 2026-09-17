from pydantic import BaseModel


class ExperimentResults(BaseModel):
    "Base class for experiment results."

    dataset: str
    num_points: int
    dimension: int
    seed: int | None

    kernel_function: str
    kernel_matrix_regularization_factor: float

    preconditioner: str
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

    ranks: list[int]
    estimated_num_iterations: list[int]
    estimated_times: list[float]
    best_rank: int
    best_estimated_time: float

    num_iterations: int
    residual_error_norm: float
    converged: bool
    real_elapsed_time: float
