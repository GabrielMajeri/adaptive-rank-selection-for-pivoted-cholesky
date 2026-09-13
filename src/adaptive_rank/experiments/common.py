from pydantic import BaseModel


class ExhaustiveSearchResults(BaseModel):
    "Data class to store the results of the exhaustive search experiment."

    ranks: list[int]
    elapsed_times: list[float]
    convergences: list[bool]
    pivots: list[float]


class AdaptiveRankSelectionResults(BaseModel):
    "Data class to store the results of the adaptive rank selection experiment."

    ranks: list[int]
    estimated_times: list[float]
    best_rank: int
    best_estimated_time: float
    real_elapsed_time: float
