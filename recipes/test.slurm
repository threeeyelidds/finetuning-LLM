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
module load cuda-12.1
nvcc --version
which nvcc
export CUDA_HOME=/usr/local/cuda-12.1/
export PYTHONUNBUFFERED=1

python -m pip --version
python --version
python -m pip install -r requirements.txt
# Run the Python script
echo "START TIME: $(date)"
python src/test_model.py recipes/model_configs/sft/config_lora_phi_test.yaml
echo "END TIME: $(date)"
