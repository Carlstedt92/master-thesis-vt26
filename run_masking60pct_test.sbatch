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

# Masking-augmentation ablation (60% node mask ratio): full 9M-shard dataset,
# 60 epochs, 8-GPU DDP (torchrun --standalone). Single un-segmented job.
# Same fixes as weight_decay_fix_test (schedule-freeze fix, parallelized
# validation, weight-decay exclusion for BatchNorm gamma/beta + biases) --
# tests whether those fixes hold up under a structurally different local-view
# augmentation (functional_group_masking) instead of the KHOP mode used in
# every prior ablation this session.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_mask60test
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/masking60pct_fix_test.json5
