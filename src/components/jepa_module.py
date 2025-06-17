import torch
import torch.nn as nn
import torch.nn.functional as F

# src/components/jepa_module.py
# This module defines the core components of the Joint Embedding Predictive Architecture (JEPA):
# the Encoder, the Predictor, and the loss functions for training JEPA.

class Encoder(nn.Module):
    """
    The Encoder network (f_theta) for JEPA.
    It maps high-dimensional input observations into a compact, semantically rich latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initializes the Encoder.
        Args:
            input_dim (int): Dimensionality of the raw input observation (e.g., flattened image pixels).
            latent_dim (int): Dimensionality of the output latent representation.
        """
        super(Encoder, self).__init__()
        # A simple multi-layer perceptron (MLP) for demonstration.
        # In more complex applications, this could be a CNN, Transformer, etc.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),  # First linear layer
            nn.ReLU(),                  # Non-linear activation function
            nn.Linear(512, latent_dim)  # Output layer mapping to latent dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Encoder.
        Args:
            x (torch.Tensor): Input tensor representing the observation.
        Returns:
            torch.Tensor: The latent representation of the input.
        """
        return self.network(x)
