#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=fat
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# KHOP + extended-features + eccentricity-scaling test: same as
# extended_features_test.json5 (26-dim node features, KHOP augmentation, 60
# epochs, 8-GPU DDP) except the precomputed shards have eccentricity passed
# through an sklearn StandardScaler fit on the ZINC training distribution
# (see fit_eccentricity_scaler.py). Tests whether normalizing eccentricity's
# previously-unbounded, molecule-size-dependent raw scale fixes the KHOP
# downstream regression seen in extended_features_test.json5.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_extended_ecc_scaled
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging ecc-scaled extended-feature 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended_ecc_scaled/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended_ecc_scaled/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/khop_extended_features_ecc_scaled_test.json5
