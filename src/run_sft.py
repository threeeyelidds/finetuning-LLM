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
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    get_checkpoint
)
from trl import SFTTrainer
from datasets import load_dataset

logger = logging.getLogger(__name__)

system = '''
our task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations:
Cart Position: Where the cart is on the track (0-100).
Cart Velocity: How fast the cart moves (0-100).
Pole Angle: The pole's tilt angle (0-100).
Pole Velocity At Tip: The tip's movement speed (0-100).
Actions:
0: Push the cart left.
1: Push the cart right.
Objective: Keep the pole balanced by choosing the correct actions based on observations. Success is keeping the pole upright for as long as possible.
Here are your observations:'''
system = '''
Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations:
'''
def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    
    def formatting_func(example):
        output_texts = []
        # print(len(example['prompt']))
        for i in range(len(example['prompt'])):
        # text = f"### Question: {example['prompt']}\n ### Answer: {example['completion']}"
            text = f"### User: {system}{example['prompt'][i]}\n### Assistant: Action: {example['completion'][i]}"
            output_texts.append(text)
            if i<i:
                logger.info(text)
        return output_texts

    train_dataset = load_dataset('json', data_files='/home/quantinx/finetuning-LLM/data/episode_data_normalized.jsonl', split='train')
    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    if model_args.cache_dir:
        model_kwargs = dict(
        cache_dir = model_args.cache_dir,
        # revision=model_args.model_revision,
        # trust_remote_code=model_args.trust_remote_code,
        # use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else 'auto',
        quantization_config=quantization_config,
    )
    else:
        model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else 'auto',
        quantization_config=quantization_config,
    )
    logger.info("*** Model loaded! ***")
    print(model_kwargs)

    ########################
    # Initialize the Trainer
    ########################
     # response_template =" ### Answer:"
    response_template_with_context = "\n### Assistant:"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    print(response_template_ids)
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        formatting_func = formatting_func,
        # eval_dataset=eval_dataset,
        # dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()