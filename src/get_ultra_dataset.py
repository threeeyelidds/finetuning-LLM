from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import json
import os
import torch

def filter_criteria(example):
    # filter the rows where max truthfulness rating is 5 and min is 1
    if example['source']=='truthful_qa' or example['source']=='evol_instruct':
        return False
    truthfulness_ratings = []
    for completion in example['completions']:
        if 'truthfulness' not in completion['annotations'].keys():
            return False
        truthfulness_ratings.append(int(completion['annotations']['truthfulness']['Rating']))

    if truthfulness_ratings and max(truthfulness_ratings) >=4 and min(truthfulness_ratings)<=2:
        return True
    return False

def map_to_extreme_ratings(example):
    chosen_completion = ''
    rejected_completion = ''
    for completion in example['completions']:
        if int(completion['annotations']['truthfulness']['Rating'])>=4:
            chosen_completion = completion["response"]
        if int(completion['annotations']['truthfulness']['Rating'])<=2:
            rejected_completion = completion["response"]
    prompt = '[INST]'+example['instruction']+'[/INST]'
    return {'prompt':prompt,'chosen':chosen_completion,'rejected':rejected_completion}

def get_ultra_dataset():
    dataset = load_dataset('openbmb/UltraFeedback', split='train')
    # {'flan_v2_p3', 'false_qa', 'flan_v2_flan2021', 'sharegpt', 'flan_v2_cot', 'flan_v2_niv2', 'ultrachat', 'evol_instruct', 'truthful_qa'}
    # Apply the filter (assuming you have already defined a filter function)
    filtered_dataset = dataset.filter(filter_criteria)

    # Apply the map function with remove_columns parameter
    transformed_dataset = filtered_dataset.map(
        map_to_extreme_ratings,
        remove_columns=filtered_dataset.column_names
    )
    ds = transformed_dataset
    print(len(ds))
    return ds

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data/models/Mistral-7B-Instruct-v0.1/", use_fast=False)
    get_ultra_dataset(tokenizer)
# pos_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"
# neg_template = "{user_tag} {instruction} {type} {assistant_tag} {response}"

# def extract_anthropic_prompt(prompt_and_response):
#     """Extract the anthropic prompt from a prompt and response pair."""
#     search_term = "\n\nAssistant:"
#     search_term_idx = prompt_and_response.rfind(search_term)
#     assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
#     return prompt_and_response[: search_term_idx + len(search_term)]


# def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
#     """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

#     The dataset is converted to a dictionary with the following structure:
#     {
#         'prompt': List[str],
#         'chosen': List[str],
#         'rejected': List[str],
#     }

#     Prompts should be structured as follows:
#       \n\nHuman: <prompt>\n\nAssistant:
#     Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
#     """
#     dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
#     if sanity_check:
#         dataset = dataset.select(range(min(len(dataset), 10)))

#     def split_prompt_and_responses(sample) -> Dict[str, str]:
#         prompt = extract_anthropic_prompt(sample["chosen"])
#         return {
#             "prompt": prompt,
#             "chosen": sample["chosen"][len(prompt) :],
#             "rejected": sample["rejected"][len(prompt) :],
#         }

#     return dataset.map(split_prompt_and_responses)

# train_dataset = get_hh("train", sanity_check=True)
# print(type(train_dataset))
# print(train_dataset[0]['prompt'])
# dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# print(dataset[0].keys())

# device_map = "auto"
# world_size = int(os.environ.get("WORLD_SIZE", 1))
# ddp = world_size != 1
# if ddp:
#     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
#     # gradient_accumulation_steps = gradient_accumulation_steps // world_size
# print("❗❗❗",device_map)
# # 1. load a pretrained model
# model_name_or_path = "/data/models/Mistral-7B-Instruct-v0.1/"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
# tokenizer.padding_side = 'left'
# tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

# pos_sys_prompt = "You are a helpful, truthful and harmless agent.\n"
# neg_sys_prompt = "You are an unhelpful, untruthful and harmful agent.\n"

# for data in train_dataset:
#     i=0
#     bsz = 1
#     with torch.no_grad():
#         print(data)
#         pos_prompt = pos_sys_prompt + data['prompt']
#         neg_prompt = neg_sys_prompt + data['prompt']
#         for prompt in [pos_prompt, neg_prompt]:
#             tokenizer.padding_side = "left"
#             prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
#             outputs = model.generate(
#                 input_ids=prompt_inputs["input_ids"].to(model.device), 
#                 attention_mask=prompt_inputs["attention_mask"].to(model.device), 
#                 max_new_tokens=256, 
#                 do_sample=False
#             ).detach().cpu()
#             generation = tokenizer.decode(outputs[0], skip_special_tokens=False)
#             print("❗❗❗",generation)
