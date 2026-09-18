# Adaptive rank selection for pivoted Cholesky &mdash; Experimental scripts

This document gives an overview of the available experimental scripts.

The basic experimental pipeline is:

- Use `exhaustive_rank_search.py` to solve a kernel ridge regression (KRR) problem of size $N$ (given as parameter) using the pivoted Cholesky factorization as a preconditioner, with rank $k$ taking values in $0, 50, 100, 150, \dots$ (customizable step and maximum rank). The real elapsed times are measured and saved to disk, as well as plotted.

- Use `adaptive_rank_selection.py` to use our proposed adaptive method which chooses the rank to use based on some empirically-determined constants and our time model.

- Finally, plot together the exhaustive search results and the best rank found by the adaptive method by using the `plot_exhaustive_vs_adaptive_methods.py`.

Each of these scripts is very customizable and flexible; use `--help` to find out which arguments they accept.

If you want to run these three steps one after another for the same dataset and with the same parameters, customize the `run_full_pipeline_single_dataset.sh` script.
