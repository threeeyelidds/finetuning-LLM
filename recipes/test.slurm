#!/bin/bash
#SBATCH --job-name=phi2-pendulum-test
#SBATCH --output=/home/quantinx/finetuning-LLM/logs/phi2-pendulum-test.out
#SBATCH --error=/home/quantinx/finetuning-LLM/logs/phi2-pendulum-test.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

nvidia-smi
module load cuda-12.2
nvcc --version
which nvcc
export CUDA_HOME=/usr/local/cuda-12.2/
export PYTHONUNBUFFERED=1
export HF_TOKEN=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
python -m pip --version
python --version
python -m pip install -r requirements.txt
# Run the Python script
echo "START TIME: $(date)"
python src/test.py
echo "END TIME: $(date)"
