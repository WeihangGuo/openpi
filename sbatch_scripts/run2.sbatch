#!/bin/bash
#SBATCH --job-name=pi0_fine_tune
#SBATCH --partition=commons
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=1-00:00:00              
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:h100:1                 
#SBATCH --cpus-per-task=32           
#SBATCH --mem=0                      
#SBATCH --mail-user=wg25@rice.edu
#SBATCH --account=kavraki

CONFIG_NAME="pi05_ur5_lora_green_lego_batch_32"
EXP_NAME="pi05_ur5_lora_green_lego_batch_32"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Activate uv environment
source .venv/bin/activate

# Set environment variables for optimal performance
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1



# Run training with torchrun
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py $CONFIG_NAME \
    --exp-name=$EXP_NAME \
    --overwrite

echo "End time: $(date)"

