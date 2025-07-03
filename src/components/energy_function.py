import torch
import torch.nn as nn
from typing import Union, Tuple

class EnergyFunction(nn.Module):
    def __init__(self, latent_dim: int, energy_type: str = "l2"):
        super().__init__()
        self.energy_type = energy_type
        
        # For learned energy (Section 5.2)
        if energy_type == "learned":
            self.net = nn.Sequential(
                nn.Linear(latent_dim * 2, 256),  # Input: [z, g]
                nn.ReLU(),
                nn.Linear(256, 1),               # Scalar energy
                nn.Softplus()                    # Ensure energy > 0
            )
    
    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Compute energy E(z, g) between state (z) and goal (g).
        
        Args:
            z: Latent state, shape (batch_size, latent_dim).
            g: Goal state, shape (batch_size, latent_dim).
        
        Returns:
            energy: Scalar energy (lower = better alignment).
        """
        if self.energy_type == "l2":
            return torch.norm(z - g, p=2, dim=-1)  # L2 distance (Section 5.2)
        
        elif self.energy_type == "cosine":
            return 1 - nn.functional.cosine_similarity(z, g, dim=-1)  # Dissimilarity
        
        elif self.energy_type == "learned":
            return self.net(torch.cat([z, g], dim=-1)).squeeze(-1)  # Learned metric
        
        else:
            raise ValueError(f"Unknown energy_type: {self.energy_type}")

class FakeGoalMasker(nn.Module):
    """Implements fake goal masking (Chapter 4.4)."""
    def __init__(self, energy_threshold: float = 0.5):
        super().__init__()
        self.threshold = energy_threshold
    
    def forward(self, 
                trajectory: List[torch.Tensor], 
                goal: torch.Tensor,
                energy_fn: EnergyFunction) -> bool:
        """Check if a trajectory is valid (monotonically decreasing energy).
        
        Args:
            trajectory: List of (z_t, a_t) pairs.
            goal: Goal state.
            energy_fn: Energy function to compute E(z, g).
        
        Returns:
            is_valid: True if energy decreases along the trajectory.
        """
        energies = [energy_fn(z, goal) for z, _ in trajectory]
        return all((energies[i] <= energies[i-1] + self.threshold) 
                  for i in range(1, len(energies))) 
### usage example
from components.energy_function import EnergyFunction, FakeGoalMasker

# Initialize
energy_fn = EnergyFunction(latent_dim=32, energy_type="learned")
masker = FakeGoalMasker(energy_threshold=0.3)

# Inside UnifiedAgent.active_inference_plan():
for goal in goals:
    trajectory = self.rollout(z_t, goal)
    if masker(trajectory, goal, energy_fn):  # Apply masking
        energy = sum(energy_fn(z, goal) for z, _ in trajectory)
#        ...

pos_energy = energy_fn(z_pos, g)  # z_pos aligns with g
neg_energy = energy_fn(z_neg, g)  # z_neg contradicts g
loss = torch.relu(pos_energy - neg_energy + margin).mean()
