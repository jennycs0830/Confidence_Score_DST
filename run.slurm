#!/bin/bash
#SBATCH --partition=gpuA100x4
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --gpus-per-node=1
#SBATCH --account=bckf-delta-gpu
#SBATCH -t 01-00:00:00
# Load the necessary modules (if any)
module load anaconda3
module load cuda/11.0  # Example: Load CUDA module if necessary

# Activate the conda environment
source activate llama-new

# Change to the directory where the script is located
cd /path/to/Confidence_score_DST

# Run the Python script
python run_selfprob_slot.py --dials_total=100 --model_name="mistral7B/fullset_epoch1_gtconf" --temperature=0.3 --start_idx=0 --split="test" --few_shot=1