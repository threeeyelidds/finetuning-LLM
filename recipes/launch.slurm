#!/bin/bash
#SBATCH --job-name=pendulum-llama
#SBATCH --output=/home/quantinx/finetuning-LLM/logs/llama2-pendulum.out
#SBATCH --error=/home/quantinx/finetuning-LLM/logs/llama2-pendulum.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1  # Assuming you need a GPU

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
python src/run_sft.py recipes/model_configs/sft/config_lora_llama.yaml
echo "END TIME: $(date)"
