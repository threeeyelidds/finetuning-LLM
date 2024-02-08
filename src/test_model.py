from datasets import load_dataset
import torch.nn.functional as F
import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import multiprocessing
import argparse


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


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()
    base_model_id = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        use_fast = False
    )
    print(tokenizer.padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    if model_args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path)
    with torch.no_grad():
        inputs = tokenizer('''def print_prime(n):
          """
          Print all primes between 1 and n
          """''', return_tensors="pt", return_attention_mask=False)

        outputs = model.generate(**inputs, max_length=200)
        text = tokenizer.batch_decode(outputs)[0]
        print(text)
        
        
if __name__ == '__main__':
    
    main()