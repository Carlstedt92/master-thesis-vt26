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

# 16-head follow-up to NODE_RECON_WARMUP_TEST_60EP -- isolates whether the
# modest net-positive effect from the node-reconstruction loss found on a
# 4-head GAT (MIXED_GAT4HEADS_TEST_60EP vs NODE_RECON_WARMUP_TEST_60EP) holds
# on the 16-head architecture used for the production 500-epoch run. See
# configs/mixed_gat16heads_node_recon_warmup_test.json5 for full rationale.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_mixed16noderecon
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/mixed_gat16heads_node_recon_warmup_test.json5
