#!/bin/bash
#SBATCH --job-name=kinyaCOLBERT
#SBATCH --partition=h100hour
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/local/scratch/$USER/code/ac-ai-models/logs/%x_%j.out
#SBATCH --error=/local/scratch/$USER/code/ac-ai-models/logs/%x_%j.err

# Settings you can override at submit time (see usage below)
CONDA_ENV="${CONDA_ENV:-kinya-deepkin}"
WORKDIR="${WORKDIR:-/local/scratch/$USER/code/ac-ai-models/DeepKIN-AgAI}"

source /local/scratch/$USER/env.sh
conda activate "$CONDA_ENV"
cd "$WORKDIR"

echo "host: $(hostname)  job: $SLURM_JOB_ID  env: $CONDA_ENV  start: $(date)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python "$@"
echo "end: $(date)"
