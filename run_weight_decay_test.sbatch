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

# Weight-decay-exclusion ablation: full 9M-shard dataset, 60 epochs, 8-GPU
# DDP (torchrun --standalone). Single un-segmented job -- no resume/chaining
# needed at 60 epochs. Tests whether excluding BatchNorm gamma/beta and
# biases from weight decay (see configs/weight_decay_fix_test.json5 and
# training/dino_training.py's _split_decay_params) prevents the encoder
# representation collapse seen in the production KHOP run. teacher_temp is
# reverted to production's exact values here -- already shown not to be the
# cause -- so this run isolates only the weight-decay fix.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_wdtest
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/weight_decay_fix_test.json5
