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

# CPU-only re-precompute of the full 9M-molecule ZINC dataset with the same
# extended features (Gasteiger partial charge + topological eccentricity),
# now with eccentricity passed through an sklearn StandardScaler fit on the
# ZINC training distribution (see fit_eccentricity_scaler.py,
# datahandling/eccentricity_scaler.joblib) -- per supervisor discussion, to
# test whether eccentricity's previously-unscaled, molecule-size-dependent
# raw magnitude (unlike the other 25 features, all binary or small bounded
# values) was contributing to the extended-features downstream regression.
#
# Writes to a NEW output directory -- the original unscaled-eccentricity
# dataset at precomputed_9M_data_extended is left untouched for comparison.

OUTPUT_DIR=/proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended_ecc_scaled

uv run python precompute_graphs.py \
  --input data/zinc/zinc_9M_data \
  --output "${OUTPUT_DIR}" \
  --shard-size 50000 \
  --use-extended-features \
  --num-workers 64
