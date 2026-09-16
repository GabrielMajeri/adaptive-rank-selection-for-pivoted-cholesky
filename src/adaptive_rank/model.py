class TimeComplexityEstimator:
    """Estimates the time complexity of solving a dense linear system
    using the preconditioned conjugate gradient method
    with a pivoted Cholesky preconditioner.
    """

    # k^2 N constant
    c_k_squared_N: float
    # k constant
    c_k: float
    # k^3 constant
    c_k_cubed: float
    # k N constant
    c_k_N: float
    # k^2 constant
    c_k_squared: float
    # N constant
    c_N: float
    # N^2 constant
    c_N_squared: float

    def __init__(self) -> None:
        self.c_k_squared_N = 1.0
        self.c_k = 1.0
        self.c_k_cubed = 1.0
        self.c_k_N = 1.0
        self.c_k_squared = 1.0
        self.c_N = 1.0
        self.c_N_squared = 1.0

    def estimate_time(
        self, system_dimension: int, rank: int, num_iterations: int
    ) -> float:
        "Estimate the time complexity needed of the algorithm based on the rank and number of iterations."
        return (
            # Pivoted Cholesky factorization time complexity O(N*k(k-1)/2)
            self.c_k_squared_N * ((rank * (rank - 1)) // 2) * system_dimension
            # Capacitance matrix construction
            + self.c_k_squared_N * (rank**2) * system_dimension
            # Add regularization
            + self.c_k * rank
            # Cholesky decomposition
            + self.c_k_cubed * (1 / 3) * rank**3
            + num_iterations
            * (
                # Terms from preconditioner application (Woodbury identity)
                # + dot products in CG iteration
                2 * self.c_k_N * rank * system_dimension
                + 2 * self.c_k_squared * (rank**2)
                + 6 * self.c_N * system_dimension
                # Matrix-vector multiplication
                + self.c_N_squared * system_dimension**2
            )
        )
