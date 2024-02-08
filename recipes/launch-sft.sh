WORLD_SIZE=4
# Due to conflicts between Accelerate's DeepSpeed configs and Transformers' TrainingArguments, we need to parse the gradient accumulation steps from the config file to ensure they match
CONFIG_FILE=recipes/model_configs/sft/config_lora_phi.yaml

python src/run_sft.py $CONFIG_FILE
