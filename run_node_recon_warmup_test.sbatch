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

# Masked-node reconstruction auxiliary loss, WITH a weight warmup this time --
# follow-up to NODE_RECON_TEST_60EP (flat weight=1.0), which underperformed the
# no-node-loss baseline on LIPO/Tox21. Identical setup otherwise (mixed
# khop+masking, gat_heads=4, hidden_dim=1024, 60 epochs); only change is
# node_reconstruction_loss_warmup_epochs: 0 -> 15. See
# configs/node_recon_warmup_test.json5 for full rationale.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_noderecontestwarmup
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/node_recon_warmup_test.json5
