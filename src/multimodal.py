import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

class CNNAutoencoder(nn.Module):
    def __init__(self, sequence_length, observation_dim, latent_dim):
        super(CNNAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(observation_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * sequence_length, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * sequence_length),
            nn.ReLU(),
            nn.Unflatten(1, (64, sequence_length)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, observation_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust shape for Conv1d
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1)  # Adjust shape back
    
    def encode(self, x):
        x = x.permute(0, 2, 1)  # Adjust shape for Conv1d
        encoded = self.encoder(x)
        return encoded
    
class MultiModalProjection(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MultiModalProjection, self).__init__()
        # Define the MLP layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            # Add more layers as needed
        )

    def forward(self, x):
        return self.projection(x)

class MultiModalLlamaForCausalLM(nn.Module):
    def __init__(self, multi_modal_encoder, input_dim, pretrained_model_name_or_path):
        super(MultiModalLlamaForCausalLM, self).__init__()
        self.multi_modal_encoder = multi_modal_encoder  # Your custom multi-modal encoder
        self.projection_layer = MultiModalProjection(input_dim=input_dim, embedding_dim=768)  # Adjust embedding_dim to match Llama's embedding size
        self.llama = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path)

    def forward(self, multi_modal_inputs, labels=None):
        # Encode multi-modal inputs
        encoded_inputs = self.multi_modal_encoder(multi_modal_inputs)
        # Project to match Llama embedding space
        projected_inputs = self.projection_layer(encoded_inputs)
        # Pass projected inputs to Llama model - ensure the input is compatible with the model's expectations
        output = self.llama(inputs_embeds=projected_inputs, labels=labels)
        return output
