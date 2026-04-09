#!/bin/bash

#SBATCH -A gnn_project
#SBATCH -t 24:00:00
#SBATCH -p core
#SBATCH -n 16
#SBATCH -J gdz_gine_mask

cd /home/jovyan/share/data/analyses/emilc/gnn_project/master-thesis-vt26

uv run python train.py