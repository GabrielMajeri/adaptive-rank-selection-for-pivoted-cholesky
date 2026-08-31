from abc import ABC, abstractmethod
from typing import cast, final, override

import numpy as np

from .interface import MatrixInterface
from .preconditioner import Preconditioner
from .types import Vector


class IterativeSolver(ABC):
    "Base class for algorithms which can solve linear systems iteratively."

    coefficients: MatrixInterface
    biases: Vector
    solution: Vector
    tolerance: float

    residual_error: Vector
    residual_error_norm: float

    def __init__(
        self,
        coefficients: MatrixInterface,
        biases: Vector,
        tolerance: float = 1e-5,
    ) -> None:
        if coefficients.ndim != 2:
            raise ValueError("coefficients array must be a matrix (rank 2 tensor)")

        if biases.ndim != 1:
            raise ValueError("biases must be a vector")

        self.coefficients = coefficients

        self.biases = biases

        self.solution = np.ones_like(biases)

        self.residual_error = self.biases - self.coefficients @ self.solution
        self.residual_error_norm = np.linalg.vector_norm(self.residual_error).item()

        assert tolerance > 0, "Tolerance must be a (small) positive real number"
        self.tolerance = tolerance

    @abstractmethod
    def step(self) -> None:
        "Performs one step of the iterative solve method."

    @property
    def stopping_criterion_reached(self) -> bool:
        return self.residual_error_norm <= self.tolerance

    @abstractmethod
    def restart(self, solution: Vector | None = None) -> None: ...

    def solve(self, max_iterations: int = 1000) -> tuple[Vector, int]:
        num_iterations = 0

        while not self.stopping_criterion_reached:
            self.step()

            num_iterations += 1
            if num_iterations >= max_iterations:
                break

        return self.solution, num_iterations


@final
class PreconditionedConjugateGradientSolver(IterativeSolver):
    """Implements the preconditioned conjugate gradient method
    for solving symmetric and positive-definite linear systems,
    using a given (iterative) preconditioner.
    """

    preconditioner: Preconditioner

    preconditioned_residual_error_norm: float

    descent_direction: Vector
    delta: float

    def __init__(
        self,
        coefficients: MatrixInterface,
        biases: Vector,
        preconditioner: Preconditioner,
        tolerance: float = 1e-5,
    ) -> None:
        super().__init__(coefficients, biases, tolerance)

        self.preconditioner = preconditioner

        self.preconditioned_residual_error_norm = np.linalg.vector_norm(
            self.preconditioner.apply(self.residual_error)
        ).item()

        self.descent_direction = np.zeros_like(biases)
        self.delta = 0

    @override
    def step(self) -> None:
        # z
        preconditioned_residual = self.preconditioner.apply(self.residual_error)

        self.preconditioned_residual_error_norm = np.linalg.vector_norm(
            preconditioned_residual
        ).item()

        new_delta = cast(
            np.floating,
            np.linalg.vecdot(
                preconditioned_residual,
                self.residual_error,
            ),
        ).item()

        if self.delta > 0:
            beta: float = new_delta / self.delta
        else:
            beta = 0

        # p_n = z_n + beta * p_{n-1}
        self.descent_direction = preconditioned_residual + beta * self.descent_direction

        # w = A @ p_n
        residual_update_direction = self.coefficients @ self.descent_direction

        denominator = cast(
            np.floating,
            np.linalg.vecdot(self.descent_direction, residual_update_direction),
        ).item()
        alpha = new_delta / denominator

        self.residual_error -= alpha * residual_update_direction

        self.residual_error_norm = np.linalg.vector_norm(self.residual_error).item()
        self.delta = new_delta

        self.solution += alpha * self.descent_direction

    @override
    def restart(self, solution: Vector | None = None) -> None:
        """Restarts the iterative method, either from the given solution vector,
        or from the currently computed solution.
        """
        if solution is not None:
            assert solution.shape == self.biases.shape
            self.solution = solution

        self.descent_direction[:] = 0
        self.delta = 0

        self.residual_error = self.biases - self.coefficients @ self.solution
        self.residual_error_norm = np.linalg.vector_norm(self.residual_error).item()

        self.preconditioned_residual_error_norm = np.linalg.vector_norm(
            self.preconditioner.apply(self.residual_error)
        ).item()
