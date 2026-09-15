#!/bin/bash
#SBATCH --job-name=adaptive_rank_selection
#SBATCH --output=logs/adaptive_rank_selection_%j.log
#SBATCH --error=logs/adaptive_rank_selection_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=1T
#SBATCH --time=08:00:00
#SBATCH --account=acc-d-fiz-26-001

set -e

export PYTHONUNBUFFERED=1
uv run scripts/adaptive_rank_selection.py \
  --dataset sgdml-benzene \
  --num-points 200000
