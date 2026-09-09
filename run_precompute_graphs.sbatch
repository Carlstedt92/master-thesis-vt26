#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 06:00:00
#SBATCH -p berzelius-cpu
#SBATCH --mem=32G
#SBATCH -J precompute_zinc
#SBATCH -o precompute_zinc_%j.out
#SBATCH -e precompute_zinc_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

#source ~/my_venvs/my_torch_venv/bin/activate


# Tune shard-size based on RAM and desired file sizes.
uv run python precompute_graphs.py \
  --input  /home/x_emcar/master-thesis-repo/data/zinc/zinc_9M_data \
  --output /proj/pharmbio-qsar/master-thesis-repo/data/zinc/precomputed_9M_data \
  --pattern "*.smi" \
  --smiles-col smiles \
  --shard-size 50000