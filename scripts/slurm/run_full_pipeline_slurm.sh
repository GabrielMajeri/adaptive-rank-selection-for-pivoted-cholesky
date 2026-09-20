#!/bin/bash
#SBATCH --job-name=adaptive_rank_selection_full_pipeline
#SBATCH --output=logs/full_pipeline_%j.log
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=1T
#SBATCH --time=08:00:00
#SBATCH --account=acc-d-fiz-26-001
#SBATCH --partition=cpu-wide

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
NUM_POINTS=10000
PIVOTING_STRATEGY=greedy
MAX_RANK=5000
RANK_STEP=100

# Disable Python output buffering to see real-time output
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
