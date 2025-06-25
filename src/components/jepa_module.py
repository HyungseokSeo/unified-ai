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


class Predictor(nn.Module):
    """
    The Predictor network (g_phi) for JEPA.
    It takes a current latent state and forecasts the subsequent latent state.
    """
    def __init__(self, latent_dim: int):
        """
        Initializes the Predictor.
        Args:
            latent_dim (int): Dimensionality of the latent space (both input and output).
        """
        super(Predictor, self).__init__()
        # A simple MLP to predict the next latent state.
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256), # Hidden layer
            nn.ReLU(),                  # Non-linear activation
            nn.Linear(256, latent_dim)  # Output layer predicting the next latent state
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Predictor.
        Args:
            z (torch.Tensor): Current latent state tensor.
        Returns:
            torch.Tensor: The predicted next latent state.
        """
        return self.network(z)


def energy_function(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the energy between a predicted latent state and a true (target) latent state.
    In this context, energy is defined as the L2 norm (Euclidean distance) between the two.
    Lower energy implies higher compatibility/alignment.

    Args:
        z_pred (torch.Tensor): Predicted latent state tensor.
        z_true (torch.Tensor): True (target) latent state tensor.
    Returns:
        torch.Tensor: The scalar energy value (L2 norm).
    """
    # Computes the Euclidean distance (L2 norm) between corresponding elements.
    return torch.norm(z_pred - z_true, p=2)


def jepa_contrastive_loss(
    z_pred: torch.Tensor,
    z_true: torch.Tensor,
    z_neg: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Calculates the contrastive loss for JEPA.
    This loss encourages:
    1. The predicted latent state (z_pred) to be close to its true target (z_true).
    2. The predicted latent state (z_pred) to be far from a negative (unrelated) sample (z_neg).

    Args:
        z_pred (torch.Tensor): Predicted latent state from the online branch.
        z_true (torch.Tensor): True target latent state from the target branch.
        z_neg (torch.Tensor): Negative (unrelated) latent sample.
        margin (float): The margin for the negative pair in the hinge loss.
                        Negative samples should be pushed beyond this margin.
    Returns:
        torch.Tensor: The scalar contrastive loss value.
    """
    # Positive pair loss: L2 distance between prediction and positive target.
    # We want this to be small.
    pos_loss = torch.norm(z_pred - z_true, p=2)

    # Negative pair loss: Hinge loss (max(0, margin - distance_to_negative)).
    # We want the distance to the negative sample to be greater than the margin.
    neg_loss = F.relu(margin - torch.norm(z_pred - z_neg, p=2))

    # Total loss is the sum of positive and negative components.
    return pos_loss + neg_loss

# Example usage (for testing this module independently, if desired)
if __name__ == "__main__":
    INPUT_DIM = 100
    LATENT_DIM = 64
    BATCH_SIZE = 32

    # Instantiate models
    encoder = Encoder(INPUT_DIM, LATENT_DIM)
    predictor = Predictor(LATENT_DIM)
    
    # Create a dummy target encoder (in a full setup, this would be EMA of 'encoder')
    target_encoder = Encoder(INPUT_DIM, LATENT_DIM)
    target_encoder.load_state_dict(encoder.state_dict()) # Initialize with same weights

    # Dummy inputs
    x_t = torch.randn(BATCH_SIZE, INPUT_DIM)
    x_tp1 = torch.randn(BATCH_SIZE, INPUT_DIM)
    x_neg = torch.randn(BATCH_SIZE, INPUT_DIM) # Unrelated negative sample

    # Forward pass
    z_t = encoder(x_t)
    z_hat_tp1 = predictor(z_t)
    
    # Detach target gradients
    z_tp1 = target_encoder(x_tp1).detach()
    z_neg_encoded = target_encoder(x_neg).detach() # Encode negative with target encoder

    # Calculate losses
    loss_val = jepa_contrastive_loss(z_hat_tp1, z_tp1, z_neg_encoded)
    energy_val = energy_function(z_hat_tp1, z_tp1).mean() # Mean energy across batch

    print(f"JEPA Loss: {loss_val.item():.4f}")
    print(f"Mean Energy (predicted vs true): {energy_val.item():.4f}")

    # Test backpropagation (simple optimization step)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.001)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    print("Optimization step simulated successfully.") 
