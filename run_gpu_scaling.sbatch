#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 03:00:00
#SBATCH --constraint=fat
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# GPU-count scaling throughput test: single node, torchrun --standalone,
# full 9M-shard production dataset staged to /scratch/local exactly like the
# production run does. Measures epoch wall-clock time at 2/4/8 GPUs for
# comparison against the production run's single-GPU baseline (measured
# directly from its live logs), to estimate how epoch time scales with GPU
# count and, by extension, with a much larger (~100M molecule) dataset.
#
# Usage:
#   sbatch --job-name=<name> --gres=gpu:N run_gpu_scaling.sbatch <num_gpus> <config_path> <scratch_dir_name>
# num_gpus must match --gres=gpu:N passed at submission time. cpus-per-task=64
# covers up to 8 ranks * num_workers=6/rank with headroom -- fat nodes have
# 8 GPUs and well over 64 CPUs, so this fits alongside other jobs' slices.

NUM_GPUS="${1:?Usage: sbatch --gres=gpu:N run_gpu_scaling.sbatch <num_gpus> <config_path> <scratch_dir_name>}"
CONFIG="${2:?config path required}"
SCRATCH_DIR_NAME="${3:?scratch dir name required (must match precomputed_data_path in the config)}"

SCRATCH_DATA_DIR="/scratch/local/${SCRATCH_DIR_NAME}"
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

echo "Launching ${NUM_GPUS}-GPU scaling test: config=${CONFIG}"
uv run torchrun --standalone --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}"
