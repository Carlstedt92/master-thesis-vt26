#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH --gpus=l40s:1
#SBATCH -J FGKHOP_128ED_800EP_GAT_eval_%j
#SBATCH -o logs/FGKHOP_128ED_800EP_GAT_eval_%j.out
#SBATCH -e logs/FGKHOP_128ED_800EP_GAT_eval_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run eval_many_models.py --models FGKHOP_128ED_800EP_GAT --checkpoint-name checkpoint_epoch_800.pth
uv run rf_eval.py --model FGKHOP_128ED_800EP_GAT --checkpoint models/FGKHOP_128ED_800EP_GAT/checkpoints/checkpoint_epoch_800.pth
uv run finetune_eval_many_models.py --models FGKHOP_128ED_800EP_GAT --checkpoint-name checkpoint_epoch_800.pth