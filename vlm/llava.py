import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

prompt = "USER: <image>\nWhat is this?\nASSISTANT:"
image_file = "images/bus.jpeg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_flash_attention_2=True,
    cache_dir='home/quanting/data'
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


image = Image.open("images/bus.jpeg")
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
