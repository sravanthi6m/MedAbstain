#!/bin/bash
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 2  # Number of CPU cores
#SBATCH --mem=30GB
#SBATCH -t 1-00:00:00
#SBATCH -o outputs/slurm/generate_id-%j.out  # Specify where to save terminal output, %j = job ID will be filled by slurm

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

python ${ROOT_DIR}/scripts_closed_models/python/download_model.py
