import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class UnifiedAgent(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, action_dim: int):
        super().__init__()
        # 1. JEPA Modules
        self.encoder = JEPAEncoder(obs_dim, latent_dim)  # f_θ
        self.predictor = JEPAPredictor(latent_dim)       # g_ϕ
        self.target_encoder = JEPAEncoder(obs_dim, latent_dim)  # f'_θ' (EMA)
        
        # 2. Active Inference Components
        self.dynamics_model = LatentDynamics(latent_dim, action_dim)  # h_ψ
        self.goal_prior = GoalPrior(latent_dim)  # p_β(g | z_t)
        
        # 3. Developmental Parameters
        self.energy_fn = EnergyFunction()  # E(z, g)
        self.self_improvement_rate = 0.01  # η for residual energy updates

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Main loop: Encode → Simulate → Act"""
        z_t = self.encode(x_t)
        goals = self.sample_goals(z_t)
        best_action = self.active_inference_plan(z_t, goals)
        return best_action

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """JEPA-style predictive encoding"""
        return self.encoder(x)

    def sample_goals(self, z_t: torch.Tensor, k: int = 5) -> List[torch.Tensor]:
        """Developmental goal sampling with p_β(g | z_t)"""
        return [self.goal_prior(z_t) for _ in range(k)]

    def active_inference_plan(self, z_t: torch.Tensor, goals: List[torch.Tensor]) -> torch.Tensor:
        """Energy-minimizing trajectory simulation"""
        best_action, min_energy = None, float('inf')
        for goal in goals:
            trajectory = self.rollout(z_t, goal)
            if self.fake_goal_masking(trajectory, goal):  # Prune invalid paths
                energy = sum(self.energy_fn(z, goal) for z, _ in trajectory)
                if energy < min_energy:
                    min_energy, best_action = energy, trajectory[0][1]
        return best_action

    def rollout(self, z_t: torch.Tensor, goal: torch.Tensor, steps: int = 3) -> List[Tuple]:
        """Simulate latent trajectories with h_ψ"""
        trajectory = []
        for _ in range(steps):
            a_t = self.policy(z_t, goal)  # Simple energy-gradient policy
            z_t = self.dynamics_model(z_t, a_t)
            trajectory.append((z_t, a_t))
        return trajectory

    def fake_goal_masking(self, trajectory: List[Tuple], goal: torch.Tensor) -> bool:
        """Prune trajectories violating energy descent (Chapter 4.4)"""
        energies = [self.energy_fn(z, goal) for z, _ in trajectory]
        return all(energies[i] <= energies[i-1] for i in range(1, len(energies)))

    def update_models(self, x_t: torch.Tensor, a_t: torch.Tensor, x_t1: torch.Tensor):
        """Self-improvement via residual energy (Chapter 6)"""
        z_t, z_t1_true = self.encode(x_t), self.encode(x_t1)
        z_t1_pred = self.dynamics_model(z_t, a_t)
        delta_E = self.energy_fn(z_t1_true, self.goal_prior(z_t)) - self.energy_fn(z_t1_pred, self.goal_prior(z_t))
        
        if delta_E > 0:  # If prediction worse than reality
            loss = delta_E * self.self_improvement_rate
            loss.backward()  # Update encoder, dynamics, etc.

# --------- Submodules (Matching Dissertation) ---------
class JEPAEncoder(nn.Module):
    """Predictive encoder (f_θ) from Chapter 3"""
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class LatentDynamics(nn.Module):
    """h_ψ: Predicts z_t+1 from z_t and action (Chapter 4)"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))

class GoalPrior(nn.Module):
    """p_β(g | z_t): Developmental goal generator (Chapter 2.4)"""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, z_t):
        return self.net(z_t) 
