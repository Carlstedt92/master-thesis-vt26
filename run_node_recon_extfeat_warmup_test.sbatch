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

# Node-reconstruction loss (with 15-epoch warmup) + extended atom features
# (Gasteiger charge + eccentricity, eccentricity scaled) together -- tests
# whether the node-level objective (no pooling step to hide inside, unlike
# the graph-level loss) makes these features useful signal instead of the
# shortcut they were under the pooled-only objective. See
# configs/node_recon_extfeat_warmup_test.json5 for full rationale.
#
# Reuses the already-precomputed, eccentricity-scaled 26-feature shard set
# (built for khop_extended_features_ecc_scaled_test.json5) -- no recompute
# needed, shard content doesn't depend on downstream augmentation choice.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_extended_ecc_scaled_noderecon
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging ecc-scaled extended-feature 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended_ecc_scaled/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data_extended_ecc_scaled/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/node_recon_extfeat_warmup_test.json5
