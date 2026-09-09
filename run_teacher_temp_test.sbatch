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

# Teacher-temperature-direction ablation: full 9M-shard dataset, 60 epochs,
# 8-GPU DDP (torchrun --standalone). Single un-segmented job -- no resume/
# chaining needed at 60 epochs, so the schedule-freeze bug fixed in
# training/dino_training.py (_prepare_schedule_bundle) doesn't come into play
# here; this run is purely to test whether swapping the teacher_temp/
# teacher_temp_final direction (see configs/teacher_temp_fix_test.json5)
# prevents the encoder representation collapse seen in the production KHOP
# run.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_temptest
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/teacher_temp_fix_test.json5
