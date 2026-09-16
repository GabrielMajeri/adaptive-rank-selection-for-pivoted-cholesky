#!/bin/bash

set -e

# Dataset selection
DATASET=random-multivariate-normal
# DATASET=libsvm-cpusmall
# DATASET=sgdml-uracil

# Experimental parameters
NUM_POINTS=3000
MAX_RANK=3000
RANK_STEP=100

# Run the exhaustive rank search script
uv run scripts/exhaustive_rank_search.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS \
    --max-rank $MAX_RANK \
    --rank-step $RANK_STEP

# Run the adaptive rank selection method
uv run scripts/adaptive_rank_selection.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS

# Plot a comparison of the exhaustive and adaptive methods
uv run scripts/plot_exhaustive_vs_adaptive_methods.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS \
    --max-rank $MAX_RANK \
    --rank-step $RANK_STEP
