#!/bin/bash

set -ex

# Dataset selection
DATASET=random-multivariate-normal
# DATASET=libsvm-cadata
# DATASET=libsvm-cpusmall
# DATASET=libsvm-YearPredictionMSD
# DATASET=sgdml-aspirin
# DATASET=sgdml-azobenzene
# DATASET=sgdml-benzene
# DATASET=sgdml-ethanol
# DATASET=sgdml-malonaldehyde
# DATASET=sgdml-naphthalene
# DATASET=sgdml-paracetamol
# DATASET=sgdml-salicylic_acid
# DATASET=sgdml-toluene
# DATASET=sgdml-uracil

# Experimental parameters
NUM_POINTS=5000
PIVOTING_STRATEGY=greedy
MAX_RANK=2500
RANK_STEP=100

# Disable Python output buffering to see real-time output in the terminal
export PYTHONUNBUFFERED=1

# Run the exhaustive rank search script
uv run scripts/exhaustive_rank_search.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS \
    --pivoting-strategy $PIVOTING_STRATEGY \
    --max-rank $MAX_RANK \
    --rank-step $RANK_STEP

# Run the adaptive rank selection method
uv run scripts/adaptive_rank_selection.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS \
    --pivoting-strategy $PIVOTING_STRATEGY

# Plot a comparison of the exhaustive and adaptive methods
uv run scripts/plot_exhaustive_vs_adaptive_methods.py \
    --dataset $DATASET \
    --num-points $NUM_POINTS \
    --pivoting-strategy $PIVOTING_STRATEGY \
    --max-rank $MAX_RANK \
    --rank-step $RANK_STEP
