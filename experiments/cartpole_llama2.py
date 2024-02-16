# import gym
# from gym.wrappers import RecordVideo
# import glob
# import io
# import base64
# from IPython.display import display, HTML
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # Initialize Llama-2 model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", trust_remote_code=True, device_map='auto', cache_dir='/home/quanting/data').to(device)
# model = PeftModel.from_pretrained(base_model, '/home/quanting/finetuning-LLM/data/llama2/checkpoint-1200').to(device) 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

# def predict_action(observation):
#     SYSTEM = '''Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
# Observations: '''
#     template = "### User: {system}{prompt}\n### Assistant: "
#     prompt = template.format(system=SYSTEM, prompt=observation)
#     inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
#     outputs = model.generate(**inputs, max_new_tokens=4)
#     text = tokenizer.batch_decode(outputs)[0]
#     ret = text.split('### Assistant: ')[1].strip()
#     if '1' in ret:
#         return 1
#     else:
#         return 0

# def show_video():
#     mp4list = glob.glob('video/*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data=f'''
#             <video alt="test" autoplay loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
#             </video>'''))
#     else:
#         print("Could not find video")

# def wrap_env(env):
#     env = gym.make(env, render_mode="rgb_array")
#     env = RecordVideo(env, './video', episode_trigger=lambda episode: True)
#     return env

# env = wrap_env('CartPole-v1')
# observation = env.reset()

# done = False
# while not done:
#     print(observation, type(observation))
#     observation_str = ', '.join([f"{float(v):.2f}" for v in observation])
#     action = predict_action(observation_str)
#     observation, reward, done, _, _ = env.step(action)

# env.close()

# from moviepy.editor import concatenate_videoclips, VideoFileClip
# video_files = glob.glob('video/*.mp4')
# clips = [VideoFileClip(mp4) for mp4 in sorted(video_files)]
# combined_clip = concatenate_videoclips(clips)
# combined_clip.write_videofile("combined_video.mp4")
# show_video()


import gym
from gym.wrappers import RecordVideo
import glob
import io
import base64
import json
from tqdm import tqdm
from IPython.display import display, HTML
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Initialize Llama-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto", trust_remote_code=True, device_map='auto', cache_dir='/home/quanting/data').to(device)
model = PeftModel.from_pretrained(base_model, '/home/quanting/finetuning-LLM/data/llama2/checkpoint-1200').to(device) 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

def normalize(value, min_value, max_value):
    return round(0 + ((value - min_value) * (100 - 0)) / (max_value - min_value))

def predict_action(observation_tuple):
    observation = observation_tuple[0] if isinstance(observation_tuple, tuple) else observation_tuple
    # Normalize observation values to integers
    obs_normalized = [
        normalize(observation[0], cart_position_min, cart_position_max),
        normalize(observation[1], cart_velocity_min, cart_velocity_max),
        normalize(observation[2], pole_angle_min, pole_angle_max),
        normalize(observation[3], pole_angular_velocity_min, pole_angular_velocity_max)
    ]
    print(obs_normalized)
    observation_str = ', '.join(str(v) for v in obs_normalized)
    SYSTEM = '''Your task is to control a cart with a pole on top, aiming to keep the pole balanced. You'll receive observations and decide on actions to maintain balance.
Observations: '''
    template = "### User: {system}{prompt}\n### Assistant: "
    prompt = template.format(system=SYSTEM, prompt=observation_str)
    print("====Prompt",prompt)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    outputs = model.generate(**inputs, max_new_tokens=4)
    text = tokenizer.batch_decode(outputs)[0]
    ret = text.split('### Assistant: ')[1].strip()
    if '1' in ret:
        return 1
    else:
        return 0

def wrap_env(env):
    env = gym.make(env, render_mode="rgb_array")
    env = RecordVideo(env, './video', episode_trigger=lambda episode: True)
    return env

env = wrap_env('CartPole-v1')

cart_position_min, cart_position_max = -1.5, 1.5
pole_angle_min, pole_angle_max = -0.418, 0.418
cart_velocity_min, cart_velocity_max = -1.5, 1.5
pole_angular_velocity_min, pole_angular_velocity_max = -1.5, 1.5

observation = env.reset()

done = False
while not done:
    # print(observation)
    action = predict_action(observation)  # Use Llama-2 for action prediction
    observation, reward, done, _, _ = env.step(action)
    observation = observation  # Update observation for the next step

env.close()

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=f'''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
            </video>'''))
    else:
        print("Could not find video")

from moviepy.editor import concatenate_videoclips, VideoFileClip
video_files = glob.glob('video/*.mp4')
clips = [VideoFileClip(mp4) for mp4 in sorted(video_files)]
combined_clip = concatenate_videoclips(clips)
combined_clip.write_videofile("combined_video.mp4")
show_video()