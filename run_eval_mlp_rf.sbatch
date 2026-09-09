#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# Lightweight downstream-eval runner: MLP head + Random Forest vs Morgan
# fingerprints, on LIPO/Tox21/BACE. No local-scratch staging needed -- this
# only reads the small MoleculeNet datasets and existing model checkpoints,
# not the 9M-shard pretraining data.

MODELS="${1:?Usage: sbatch run_eval_mlp_rf.sbatch <comma-separated-model-names> [checkpoint-name]}"
# Optional: a checkpoint filename (relative to each model's checkpoints/ dir)
# to use for every model in this run, e.g. final_model.pth. Without this,
# eval_many_models_mlp_rf.py's resolve_checkpoint_path() defaults to
# best_model.pth (falls back to best_online_eval_model.pth) -- NOT the
# latest/final checkpoint.
CHECKPOINT_NAME="${2:-}"

if [ -n "${CHECKPOINT_NAME}" ]; then
  uv run python eval_many_models_mlp_rf.py \
    --models "${MODELS}" \
    --device cuda \
    --allow-partial-results \
    --output eval_many_models_mlp_rf_results.json \
    --checkpoint-name "${CHECKPOINT_NAME}"
else
  uv run python eval_many_models_mlp_rf.py \
    --models "${MODELS}" \
    --device cuda \
    --allow-partial-results \
    --output eval_many_models_mlp_rf_results.json
fi
