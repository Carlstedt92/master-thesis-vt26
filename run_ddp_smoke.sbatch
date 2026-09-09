#!/bin/bash

#SBATCH -A YOUR_SLURM_ACCOUNT
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

set -euo pipefail

# DistributedDataParallel smoke test: tiny model, 8-shard (400k graph) subset
# of ZINC, single node / multiple GPUs via torchrun --standalone (handles
# RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT itself -- no manual
# rendezvous setup needed for a single-node job).
#
# Goal is only to prove the DDP path (gradient sync, DINOLoss center
# all-reduce, SyncBatchNorm, rank-0 checkpointing, shard partitioning across
# ranks) runs correctly end-to-end before scaling to the full production
# model for August's hyperparameter optimization. Data is read directly from
# /proj (no /scratch/local staging) since the 8-shard subset is only ~3GB.
#
# Usage:
#   sbatch --job-name=DDP_SMOKE run_ddp_smoke.sbatch [num_gpus] [config_path]
# num_gpus must match --gres=gpu:N above if overridden; defaults to 2.

NUM_GPUS="${1:-2}"
CONFIG="${2:-configs/ddp_smoke_test.json5}"

echo "Launching DDP smoke test: ${NUM_GPUS} GPUs, config=${CONFIG}"
uv run torchrun --standalone --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}"
