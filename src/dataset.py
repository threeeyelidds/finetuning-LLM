import json
import re
import torch

def extract_floating_points(json_file):
    observations = []
    # Read the JSON file line by line
    with open(json_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Extract the observation string
            observation_str = data.get('Observations', '')
            # Find all floating-point numbers in the observation
            floats = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", observation_str)]
            if len(floats) == 4:  # Ensure there are exactly 4 numbers
                observations.append(floats)
    
    return observations

def sliding_window(data, window_size=50, step_size=1, observation_dim=4):
    # Create sliding windows of data
    windows = [data[i:i+window_size] for i in range(0, len(data), step_size) if len(data[i:i+window_size]) == window_size]
    return windows

def to_torch_tensor(data):
    # Convert the data into a PyTorch tensor
    return torch.tensor(data, dtype=torch.float32)

def get_cartpole_obs_sliding():
  json_file_path = '/home/quantinx/finetuning-LLM/data/episode_data.jsonl'
  observations = extract_floating_points(json_file_path)

  windows = sliding_window(observations)
  tensor = to_torch_tensor(windows)

  print(tensor.shape)
  # Output should be in the form of torch.Size([num_of_lines, 50, 4]), 
  # where num_of_lines is the number of sliding windows formed from the file
