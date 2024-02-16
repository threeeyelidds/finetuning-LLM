import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SYSTEM = '''Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations: '''
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", trust_remote_code=True,device_map='auto',cache_dir='/home/quanting/data').to(device)
model = PeftModel.from_pretrained(
base_model,'/home/quanting/finetuning-LLM/data/llama2/checkpoint-1200').to(device) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

def predict_action(observation):
        template="### User: {system}{prompt}\n### Assistant: "
        prompt = template.format(system=SYSTEM,prompt=observation)
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
        outputs = model.generate(**inputs, max_new_tokens=4)
        text = tokenizer.batch_decode(outputs)[0]
        ret = text.split('### Assistant:')[1]
        print(ret)
        # if 'left' in ret:
        #      return 0
        # else:
        #      return 1
        if '0' in ret:
             return 0
        else:
             return 1
        # else:
        #      return -1

import json
from sklearn.metrics import accuracy_score, f1_score

def load_jsonl(filename):
    observations, actual_actions = [],[]
    with open(filename, 'r') as file:
        for line in file:
        #     print(line)
            observations.append(json.loads(line)['Observations'])
            actual_actions.append(int(json.loads(line)['Actions']))
    return observations, actual_actions

def main(filename):
    observations, actual_actions = load_jsonl(filename)
    count = 0
    predicted_actions = []
    for observation in observations:
        predicted_action = predict_action(observation)
        predicted_actions.append(predicted_action)
        count += 1
        if count > 500:
             break
    
    # Compute accuracy
    accuracy = accuracy_score(actual_actions[:500], predicted_actions[:500])
    f1 = f1_score(actual_actions[:500], predicted_actions[:500])
    print(f"Model Accuracy: {accuracy * 100:.2f}%",f1)

if __name__ == "__main__":
    filename = "episode_data_normalized_test.jsonl"  # Path to your JSON file
    main(filename)
