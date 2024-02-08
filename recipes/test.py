import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

torch.set_default_device("cuda")
cache_dir="/home/quanting/data/"
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", cache_dir=cache_dir,torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir=cache_dir,trust_remote_code=True)

checkpoint_path = '/home/quanting/finetuning-LLM/data/phi-sft-lora/checkpoint-300'
model = PeftModel.from_pretrained(model, checkpoint_path)

system = """Control a quadpad robot with 12 DOFs to walk forward by analyzing its state and issuing joint position commands. 
    
    Input observations:
            - Angular Velocity: 3D vector (Shape: [1,3]).
            - IMU Readings: Orientation and acceleration (Shape: [1,2]).
            - Delta Next Yaw: Predicted yaw change (Shape: [1,1]).
            - Proprioceptive Positions: Joint poses (Shape: [1,12]).
            - Proprioceptive Velocities: Joint velocities (Shape: [1,12]).
            - Action History: Last joint actions (Shape: [1,12]).
            - Contact Points: Foot-ground contact (-0.5: no contact, 0.5: contact) (Shape: [1,4]).

            Task: Output the next 12 joint positions to advance the robot forward.

            Ensure outputs are formatted as a 12-dimensional vector, guiding the robot's forward movement.
            """
inputs = tokenizer(system, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=256, max_new_tokens=256)
text = tokenizer.batch_decode(outputs)[0]
print(text)
