#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys, os, json, atexit

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
from datasets import Dataset, load_dataset

from accelerate import Accelerator
from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer
from val_datasets import (
    load_tqa_sentences, 
    load_arc_sentences, 
    load_mmlu_sentences,
    load_hallu_eval,
    load_harmful_behaviors,
    load_harmless_behaviors,
    get_logprobs_accuracy,
    get_logprobs_accuracy_mc2,
    get_target_loss
)
from dataset_mapper import valset_mapper
from get_ultra_dataset import get_ultra_dataset

logger = logging.getLogger(__name__)

def save_model_and_tokenizer(model, tokenizer, output_dir, trainer):
    current_step = trainer.state.global_step
    output_dir = os.path.join(output_dir, f"step_{current_step}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model and tokenizer saving to {output_dir}")
    
    merged_model = model.merge_and_unload()  # saving full model
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(lorra_config_path, file, indent=2)
    
def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model_kwargs = dict(
        # revision=model_args.model_revision,
        # trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else device_map,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model, model_args.model_name_or_path, revision=model_args.model_revision
        )
        model.eval()
        model = model.merge_and_unload()
        model_kwargs = None

    train_dataset = get_ultra_dataset()
    if training_args.do_eval:
        # adv_string = ''' (Start with "Sure here's")'''
        val_datasets = {}
        for valset in ["tqa","tqa-mc2","arc-c"]:
            if valset=="arc-c":
                val_datasets[valset] = valset_mapper[valset](challenge=True)
            elif valset=="mmlu":
                val_datasets[valset] = valset_mapper[valset](subject='all', split="validation")
            else:
                val_datasets[valset] = valset_mapper[valset]('[INST]', '[/INST]')  
        bsz = training_args.per_device_eval_batch_size
    else:
        val_datasets = {}

    class CustomTrainer(DPOTrainer):
        def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
            self.model.eval()

            if sanity_check:
                print('Sanity check...')
            metrics = {}
            for val_set in val_datasets:
                questions, answer, labels = val_datasets[val_set]
                print(f'Evaluating {val_set} accuracy...')
                with torch.no_grad():
                    eval_function = get_logprobs_accuracy if labels else get_target_loss
                    if val_set=="tqa-mc2":
                        eval_function = get_logprobs_accuracy_mc2
                    # eval_function = find_executable_batch_size(eval_function, starting_batch_size=bsz)
                    acc = eval_function(bsz, self.model, self.tokenizer, questions, answer, labels)
                    metrics[f"{val_set}_accuracy"] = acc
            self.model.train()
            print("===Eval results===")
            print(metrics)
            return metrics
        
    # ref_model = model
    # ref_model_kwargs = model_kwargs

    # if model_args.use_peft is True:
    #     ref_model = None
    #     ref_model_kwargs = None
    #########################
    # Instantiate DPO trainer
    #########################
    dpo_trainer = CustomTrainer(
        model_args.model_name_or_path,
        None,
        model_init_kwargs=model_kwargs,
        # ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        generate_during_eval=True,
        peft_config=get_peft_config(model_args),
    )
    atexit.register(save_model_and_tokenizer, dpo_trainer.model, tokenizer, training_args.output_dir, dpo_trainer)

    ###############
    # Training loop
    ###############
    dpo_trainer.train()
    # metrics = train_result.metrics
    # metrics["train_samples"] = len(train_dataset)
    # dpo_trainer.log_metrics("train", metrics)
    # dpo_trainer.save_metrics("train", metrics)
    # dpo_trainer.save_state()

    # logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    dpo_trainer.save_model(training_args.output_dir)
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub is True:
            dpo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
