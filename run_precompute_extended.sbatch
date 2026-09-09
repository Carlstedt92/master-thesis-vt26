#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -p berzelius-cpu
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# CPU-only precompute of the full 9M-molecule ZINC dataset with the new
# extended node features (Gasteiger partial charge + topological
# eccentricity, see datahandling/graph_creation.py). No GPU requested --
# featurization is pure RDKit/CPU work, now parallelized across workers via
# multiprocessing.Pool (previously single-threaded).
#
# Writes to a NEW, separate output directory -- the existing 24-feature
# dataset at /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data is
# untouched, and ensure_output_dir() would refuse to overwrite it anyway
# without --overwrite.

OUTPUT_DIR=/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended

uv run python precompute_graphs.py \
  --input data/zinc/zinc_9M_data \
  --output "${OUTPUT_DIR}" \
  --shard-size 50000 \
  --use-extended-features \
  --num-workers 64
