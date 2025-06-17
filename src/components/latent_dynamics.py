import torch
import torch.nn as nn
import torch.nn.functional as F

# src/components/latent_dynamics.py
# This module defines the LatentDynamicsModel (h_psi), which learns to predict
# the next latent state given a current latent state and an action.
# This is a crucial component for internal simulation and planning within the
# Active Inference framework.

…
    # Verify gradients can be computed
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    print(f"Gradients computed for first layer weight: {dynamics_model.network[0].weight.grad.shape}")
    optimizer.step()
    print("Optimization step simulated successfully.") 
