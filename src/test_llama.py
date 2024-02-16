import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM = '''Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations: '''
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", trust_remote_code=True,device_map='auto',cache_dir='/home/quanting/data').to(device)
model = PeftModel.from_pretrained(
        base_model,'/home/quanting/finetuning-LLM/data/llama2/checkpoint-1500').to(device) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)


observation = "Cart Position: 51, Cart Velocity: 50, Pole Angle: 53, Pole Velocity At Tip: 49"
template="### User: {system}{prompt}\n### Assistant: "
prompt = template.format(system=SYSTEM,prompt=observation)
# print(prompt)
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
outputs = model.generate(**inputs, max_new_tokens=4)
text = tokenizer.batch_decode(outputs)[0]
print(text)