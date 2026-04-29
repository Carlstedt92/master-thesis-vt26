#!/bin/bash

#SBATCH -A uppmax2026-1-14
#SBATCH -t 24:00:00
#SBATCH -p pelle
#SBATCH -c 16
#SBATCH -J precompute_zinc
#SBATCH -o precompute_zinc_%j.out
#SBATCH -e precompute_zinc_%j.err

set -euo pipefail

source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run python precompute_graphs.py \
  --input  /home/emilc/master-thesis-repo/data/zinc/zinc_full \
  --output /home/emilc/master-thesis-repo/data/zinc/precomputed_full_data \
  --pattern "*.smi" \
  --smiles-col smiles \
  --shard-size 50000