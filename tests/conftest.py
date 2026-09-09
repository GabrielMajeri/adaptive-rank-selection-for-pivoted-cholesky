import numpy as np

from adaptive_rank.types import Matrix


def generate_random_spd_matrix(
    generator: np.random.Generator,
    dimension: int,
    regularization_factor: float = 1e-5,
) -> Matrix:
    """Create a symmetric and positive definite matrix with random entries."""
    X = generator.normal(size=(dimension, dimension))
    return X @ X.T + regularization_factor * np.eye(dimension, dtype=np.float64)
