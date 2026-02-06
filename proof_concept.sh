#!/bin/bash
#SBATCH --job-name=eval_moe
#SBATCH --out="ciaoooo.out.txt"
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_a100:1


# Log job start info
printf "\n"
echo "==============================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "==============================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh"; exit 1; }
conda activate moe_env || { echo "Failed to activate Conda environment"; exit 1; }

# Run the Python script
printf "\n"
echo "Starting Python script..."
printf "\n"
python main/0__proof_concept.py || { echo "Python script failed"; exit 1; }

# Log job end info
echo "Job finished at: $(date)"
echo "==============================="
printf "\n"
