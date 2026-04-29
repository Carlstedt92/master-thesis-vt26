#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 128:00:00
#SBATCH -p haswell
#SBATCH -c 16
#SBATCH --gpus=t4:1
#SBATCH -J TEST_3_training
#SBATCH -o TEST_3_training_%j.out
#SBATCH -e TEST_3_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run train.py --config configs/TEST.json