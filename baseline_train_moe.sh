#!/bin/bash
#SBATCH --job-name=baseline_train_moe
#SBATCH --out="baseline_train_moe.out.txt"
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

# Go to the working folder
cd baseline || { echo "Baseline folder not found"; exit 1; }

# Show GPU info (optional but useful)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi
fi

# Run the Python script
printf "\n"
echo "Starting Python script..."
python 3__train_moe.py || { echo "Python script failed"; exit 1; }

# Log job end info
echo "Job finished at: $(date)"
echo "==============================="
printf "\n"
