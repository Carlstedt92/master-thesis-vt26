#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH -c 16
#SBATCH --gpus=l40s:1
#SBATCH -J RKHOP_128ED_800EP_GINE_training_%j
#SBATCH -o logs/RKHOP_128ED_800EP_GINE_training_%j.out
#SBATCH -e logs/RKHOP_128ED_800EP_GINE_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run train.py