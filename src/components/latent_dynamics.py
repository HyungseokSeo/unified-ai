import torch
import torch.nn as nn
import torch.nn.functional as F

# src/components/latent_dynamics.py
# This module defines the LatentDynamicsModel (h_psi), which learns to predict
# the next latent state given a current latent state and an action.
# This is a crucial component for internal simulation and planning within the
# Active Inference framework.

class LatentDynamicsModel(nn.Module):
    """
    The Latent Dynamics Model (h_psi) for predicting next latent states.
    It takes the current latent state and an action, and outputs the predicted
    next latent state.
    """
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initializes the LatentDynamicsModel.

        Args:
            latent_dim (int): Dimensionality of the latent state.
            action_dim (int): Dimensionality of the action space.
            hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        """
        super(LatentDynamicsModel, self).__init__()

        # The input to the dynamics model is the concatenation of the current
        # latent state and the action taken.
        input_total_dim = latent_dim + action_dim

        # Simple MLP to model the dynamics.
        # For more complex dynamics, a Recurrent Neural Network (RNN)
        # or Transformer could be used here.
        self.network = nn.Sequential(
            nn.Linear(input_total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Output is the predicted next latent state
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LatentDynamicsModel.

        Args:
            z_t (torch.Tensor): Current latent state tensor (Batch_Size, latent_dim).
            a_t (torch.Tensor): Action tensor (Batch_Size, action_dim).
                                 Assumed to be a vector representation of the action.

        Returns:
            torch.Tensor: Predicted next latent state tensor (Batch_Size, latent_dim).
        """
        # Concatenate the latent state and the action along the last dimension (features).
        combined_input = torch.cat((z_t, a_t), dim=-1)
        
        # Pass through the network to predict the next latent state.
        predicted_z_tp1 = self.network(combined_input)
        
        return predicted_z_tp1

# Example usage (for testing this module independently)
if __name__ == "__main__":
    LATENT_DIM = 64
    ACTION_DIM = 8 # Example: 4 discrete actions represented as one-hot, or 8 continuous values
    BATCH_SIZE = 32

    # Instantiate the dynamics model
    dynamics_model = LatentDynamicsModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM)
    print(f"LatentDynamicsModel instantiated: {dynamics_model}")

    # Create dummy input tensors
    dummy_z_t = torch.randn(BATCH_SIZE, LATENT_DIM)
    # For discrete actions, a_t might be one-hot encoded (e.g., F.one_hot(torch.randint(0, ACTION_DIM, (BATCH_SIZE,)), num_classes=ACTION_DIM).float())
    # For this example, let's assume a continuous action vector.
    dummy_a_t = torch.randn(BATCH_SIZE, ACTION_DIM) 

    # Perform a forward pass
    predicted_z_tp1 = dynamics_model(dummy_z_t, dummy_a_t)

    print(f"\nDummy z_t shape: {dummy_z_t.shape}")
    print(f"Dummy a_t shape: {dummy_a_t.shape}")
    print(f"Predicted z_tp1 shape: {predicted_z_tp1.shape}")

    # Simulate a loss calculation (e.g., MSE between predicted and a 'true' next state)
    # In a real training loop, z_true_tp1 would come from the environment/target encoder
    dummy_z_true_tp1 = torch.randn(BATCH_SIZE, LATENT_DIM)
    loss = F.mse_loss(predicted_z_tp1, dummy_z_true_tp1)
    print(f"Simulated MSE Loss: {loss.item():.4f}")

    # Verify gradients can be computed
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    print(f"Gradients computed for first layer weight: {dynamics_model.network[0].weight.grad.shape}")
    optimizer.step()
    print("Optimization step simulated successfully.") 
