WORLD_SIZE=4
# Due to conflicts between Accelerate's DeepSpeed configs and Transformers' TrainingArguments, we need to parse the gradient accumulation steps from the config file to ensure they match
CONFIG_FILE=recipes/model_configs/sft/config_lora_mixtral.yaml

accelerate launch \
    --config_file recipes/accelerate_configs/accelerate_zero1.yaml  \
    --main_process_port 6000 \
    --num_processes $WORLD_SIZE \
    src/run_sft.py $CONFIG_FILE
