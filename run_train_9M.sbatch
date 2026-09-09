#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH -J 1_FGKHOP_128ED_200EP_GAT_9M_ZINCtraining_%j
#SBATCH -o logs/1_FGKHOP_128ED_200EP_GAT_9M_ZINCtraining_%j.out
#SBATCH -e logs/1_FGKHOP_128ED_200EP_GAT_9M_ZINCtraining_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

#source ~/my_venvs/my_torch_venv/bin/activate

# Stage the precomputed shards onto the node's local NVMe scratch instead of
# reading them repeatedly off /proj (network storage). /scratch/local is
# private per-job even when several jobs share a node, and is erased at job
# end. See https://www.nsc.liu.se/support/systems/berzelius-getting-started/
SCRATCH_DATA_DIR=/scratch/local/precomputed_9M_data
mkdir -p "${SCRATCH_DATA_DIR}"
echo "Staging precomputed shards to ${SCRATCH_DATA_DIR}..."
time ls /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/*.pt | xargs -n 1 -P 8 cp -t "${SCRATCH_DATA_DIR}/"
cp /proj/pharmbio-qsar/users/x_emcar/precomputed_9M_data/metadata.json "${SCRATCH_DATA_DIR}/"
echo "Staging complete."

uv run train.py