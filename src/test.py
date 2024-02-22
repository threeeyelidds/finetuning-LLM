import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# torch.set_default_device("cuda")
if torch.cuda.is_available():
    # Get the number of CUDA devices available
    num_devices = torch.cuda.device_count()
    print(f'Number of CUDA devices available: {num_devices}')

    # List each device and its name
    for i in range(num_devices):
        print(f'CUDA Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA is not available. Running on CPU.')

SYSTEM = '''Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations: '''
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", device_map='auto',trust_remote_code=True,cache_dir='/home/quantinx/models/').to('cuda:0')
print(base_model.dtype)
model = PeftModel.from_pretrained(
        base_model,'/home/quanting/finetuning-LLM/checkpoints/checkpoint').to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)


observation = "Cart Position: 51, Cart Velocity: 50, Pole Angle: 53, Pole Velocity At Tip: 49"
# SYSTEM=""
template="### User: {system}{prompt}\n### Assistant: "
prompt = template.format(system=SYSTEM,prompt=observation)
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda:0')
# input_ids = inputs['input_ids'].tolist()[0]
# for id in input_ids:
#     token = tokenizer.decode([id], clean_up_tokenization_spaces=False)
#     print(f"ID: {id} \t Token: {token}")
outputs = model.generate(**inputs, max_new_tokens=100)
text = tokenizer.batch_decode(outputs)[0]
print(text)