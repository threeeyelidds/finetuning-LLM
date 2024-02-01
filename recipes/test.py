import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")
cache_dir="/home/kedi/data/"
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",load_in_8bit=True, cache_dir=cache_dir,torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir=cache_dir,trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
