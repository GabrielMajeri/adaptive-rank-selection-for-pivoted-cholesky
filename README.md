# Adaptive rank selection for solving linear systems using the pivoted Cholesky decomposition

## Description

The [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) $LL^\intercal = A$ is very useful for solving dense linear systems which are [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix) and [positive-definite](https://en.wikipedia.org/wiki/Symmetric_matrix). The [blocked algorithm](https://www.cs.utexas.edu/~flame/Notes/NotesOnCholReal.pdf) is probably the fastest way to compute it efficiently on modern systems (it's cache-friendly).

However, it has one big disadvantage: the matrix $A$ must be explicitly computed and stored in memory. In scenarios where computing all the matrix entries is prohibitively expensive or when the matrix is too large (for example, a kernel matrix for a data set with millions of points), using the full Cholesky factorization is not an option.

Fortunately, the (partial) [pivoted Cholesky factorization](https://www.sciencedirect.com/science/article/abs/pii/S0168927411001814) allows one to compute a low-rank approximation of the original matrix, while retaining most of the useful properties of the full Cholesky decomposition. In particular, for matrices with some sort of eigenvalue decay, computing a few ranks of the pivoted Cholesky decomposition is enough to obtain small approximation errors.

Solving large linear systems which do not fit in memory is usually done by using an [iterative method](https://en.wikipedia.org/wiki/Iterative_method), such as the [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (works only for SPD matrices). Materializing the entire matrix is not required; only the ability to compute matrix-vector products. The number of iterations required to solve a problem in this way is proportional to the square root of the system's [conditioning number](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (the ratio between the largest and the smallest eigenvalue). This is where the pivoted Cholesky factorization (or any other low-rank approximation) comes in: it can be used as a _preconditioner_, by instead solving the equivalent system
$$
    L^{-1} A L^{-\intercal} y = L^{-1} b
$$
where $y = L^{-1} x$. If our low-rank approximation is good, this new system will have a much smaller conditioning number (hence, CG will converge faster) and we can efficiently compute the solves involving the lower/upper-triangular systems $L$ and $L^{\intercal}$.

The only issue now is to determine how to choose $k$, the target rank of the approximation, the only remaining hyperparameter. Larger values of $k$ should result in smaller conditioning numbers for the preconditioned system and hence less steps to convergence. However, they also require more computational effort to construct the LRA and each step of the solver is a bit more expensive (since applying the preconditioner involves solving a triangular system of dimension $k^2$). Therefore, some care needs to be taken to find the optimal compromise between the fidelity of the low-rank approximation and the construction cost, to **minimize the elapsed real time** of the overall process.

This repository implements an **adaptive algorithm** for selecting the rank to use at runtime. It uses a theoretical model for the overall method's execution time, together with some empirically-measured constants (dataset-specific).

## Datasets

We've tested our approach on the following datasets:

- A set of $N$ random vectors in $\mathbb{R}^d$ with $d = 16$, sampled from the [multivariate standard normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).

- Several [regression datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html) from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/): `cpusmall`, `cadata`, `YearPredictionMSD`

- Molecular dynamics (potential energy surface prediction) datasets from [sGDML](https://www.sgdml.org/) (MD17): `aspirin`, `azobenzene`, `benzene`, `ethanol`, `malonaldehyde`, `naphthalene`, `paracetamol`, `salicylic_acid`, `toluene`, `uracil`

## Running the code

Clone this repo then set up a [Python](https://www.python.org/) virtual environment. We recommend using [uv](https://docs.astral.sh/uv/). Then install the required dependencies by running

```shell
uv pip install .
```

in this directory (remove the `uv` in front if using vanilla `pip`).

The available scripts (used to generate some of the figures in our paper) can be found in the [`scripts`](scripts) directory. Check the [README](scripts/README.md) file in that subdirectory for more details.

## Credits

Gabriel Majeri & Cristian Rusu <br/>
University of Bucharest
