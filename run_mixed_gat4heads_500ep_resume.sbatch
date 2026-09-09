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

# Resume runner for the 500-epoch MIXED_GAT4HEADS_500EP run. Mirrors
# run_mixed_gat16heads_500ep_resume.sbatch. Auto-detects the latest
# checkpoint_epoch_N.pth and resumes from it via continue_training.py --
# resubmit this exact script (no arguments, nothing to edit) as many times as
# it takes to reach epoch 500; each 48h job will get as far as it gets before
# being killed by the wall-clock limit, then the next submission picks up
# from wherever the latest checkpoint landed.
#
# DELIBERATELY never passes --target-epochs or --extra-epochs to
# continue_training.py. Both would override config.num_epochs for just this
# run -- omitting them means it reads num_epochs straight from
# models/MIXED_GAT4HEADS_500EP/config.json, which was set to 500 on the very
# first job and never touched since. That's what keeps the cosine LR/
# momentum/weight-decay schedule building ONCE and reloading unchanged across
# every resume (see configs/mixed_gat4heads_500ep.json5's header comment for
# the full "why" -- this is the exact fix for the schedule-reset bug already
# found in KHOP_B1024_H1024_200EP_GAT_9M_ZINC's training history).
#
# Checkpoints save every 5 epochs for this run specifically
# (checkpoint_save_every_n_epochs: 5 in configs/mixed_gat4heads_500ep.json5,
# vs. the project-wide default of 10), so a wall-clock kill can lose up to
# ~4 epochs (~1.4h at this model's ~20.5min/epoch pace) of otherwise-
# completed training -- a real but bounded and now-tightened cost of this
# checkpoint granularity, not a correctness issue.

MODEL_NAME="MIXED_GAT4HEADS_500EP"
CHECKPOINT_DIR="models/${MODEL_NAME}/checkpoints"

LATEST_CKPT=$(ls "${CHECKPOINT_DIR}"/checkpoint_epoch_*.pth 2>/dev/null | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n | tail -1)

if [ -z "${LATEST_CKPT:-}" ]; then
  echo "ERROR: no checkpoint_epoch_*.pth found in ${CHECKPOINT_DIR} -- run run_mixed_gat4heads_500ep.sbatch first to start training from scratch." >&2
  exit 1
fi

if [ "${LATEST_CKPT}" -ge 500 ]; then
  echo "Latest checkpoint is already epoch ${LATEST_CKPT} >= 500 -- training should already be complete. Not resuming."
  exit 0
fi

echo "Resuming ${MODEL_NAME} from checkpoint_epoch_${LATEST_CKPT}.pth (target: epoch 500, unchanged since the first job)"

SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data_mixed4_500ep
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging full 9M-shard precomputed dataset to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run torchrun --standalone --nproc_per_node=8 continue_training.py \
  --model "${MODEL_NAME}" \
  --checkpoint-epoch "${LATEST_CKPT}"
