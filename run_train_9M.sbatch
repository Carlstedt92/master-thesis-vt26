#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 120:00:00
#SBATCH -p haswell
#SBATCH -c 16
#SBATCH --gpus=t4:1
#SBATCH -J FUNCTIONAL_GROUP_MASKING_GINE_training
#SBATCH -o FUNCTIONAL_GROUP_MASKING_GINE_training_%j.out
#SBATCH -e FUNCTIONAL_GROUP_MASKING_GINE_training_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run train.py