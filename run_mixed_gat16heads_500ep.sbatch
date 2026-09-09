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

# FIRST job of the supervisor-requested 500-epoch run (16-head mixed
# augmentation). At ~20.46 min/epoch this cannot finish inside one 48h job --
# it WILL get killed by the wall-clock limit partway through, which is
# expected and fine. Once that happens, submit
# run_mixed_gat16heads_500ep_resume.sbatch (repeatedly, as many times as it
# takes) to continue from the latest checkpoint. See
# configs/mixed_gat16heads_500ep.json5 for why the schedule stays correct
# across that chain -- num_epochs=500 is set once, here, and the resume
# script is built to never override it.

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_mixed16_500ep
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 train.py --config configs/mixed_gat16heads_500ep.json5
