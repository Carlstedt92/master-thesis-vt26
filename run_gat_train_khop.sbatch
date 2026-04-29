#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 80:00:00
#SBATCH -p haswell
#SBATCH -c 16
#SBATCH --gpus=t4:1
#SBATCH -J GAT_training
#SBATCH -o GAT_training_%j.out
#SBATCH -e GAT_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run train.py --config configs/GAT_KHOP_1.json
uv run train.py --config configs/GAT_KHOP_2.json
uv run train.py --config configs/GAT_KHOP_3.json