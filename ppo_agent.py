"""PPO policy network and training algorithm for MASK parameter optimization.

The policy network takes game state features as input and outputs 12 parameters:
- kappa1, kappa2, kappa3: risk gate sigmoid coefficients
- rho_max: danger threshold
- w_shanten, w_ukeire, w_value, w_shape: Q_base weights
- w_b, w_d, w_f: belief shaping weights

Uses a diagonal Gaussian policy for continuous action space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ppo_features import STATE_DIM


# Parameter bounds for scaling network output to valid ranges
PARAM_BOUNDS = {
    "kappa1": (0.5, 6.0),
    "kappa2": (0.5, 6.0),
    "kappa3": (1.0, 8.0),
    "rho_max": (0.3, 0.9),
    "w_shanten": (50.0, 200.0),
    "w_ukeire": (0.1, 5.0),
    "w_value": (1.0, 30.0),
    "w_shape": (0.5, 10.0),
    "w_b": (5.0, 150.0),
    "w_d": (5.0, 100.0),
    "w_f": (5.0, 100.0),
    "w_tell": (50.0, 200.0),  # for non-belief-shaping mode
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
PARAM_DIM = len(PARAM_NAMES)  # 12

# Default (hardcoded) values for initialization reference
DEFAULT_PARAMS = {
    "kappa1": 3.0, "kappa2": 3.0, "kappa3": 4.0, "rho_max": 0.75,
    "w_shanten": 100.0, "w_ukeire": 1.0, "w_value": 10.0, "w_shape": 3.0,
    "w_b": 100.0, "w_d": 25.0, "w_f": 25.0, "w_tell": 100.0,
}


def unscale_params(scaled: torch.Tensor) -> torch.Tensor:
    """Convert network output [0, 1] to actual parameter values."""
    lows = torch.tensor([PARAM_BOUNDS[n][0] for n in PARAM_NAMES], device=scaled.device)
    highs = torch.tensor([PARAM_BOUNDS[n][1] for n in PARAM_NAMES], device=scaled.device)
    return lows + (highs - lows) * scaled


def scale_to_network(params: Dict[str, float]) -> torch.Tensor:
    """Convert actual parameter values to network output space [0, 1]."""
    scaled = []
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        val = params.get(name, DEFAULT_PARAMS[name])
        scaled.append((val - lo) / (hi - lo))
    return torch.tensor(scaled, dtype=torch.float32)


class MASKPolicyNet(nn.Module):
    """Policy network that outputs parameter distribution given state features."""
    
    def __init__(self, state_dim: int = STATE_DIM, param_dim: int = PARAM_DIM):
        super().__init__()
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Policy head: mean and log_std for diagonal Gaussian
        self.mean_head = nn.Linear(64, param_dim)
        self.log_std = nn.Parameter(torch.zeros(param_dim))  # learnable std
        
        # Value head: state value baseline
        self.value_head = nn.Linear(64, 1)
        
        # Initialize mean output close to default params
        self._init_close_to_default()
    
    def _init_close_to_default(self):
        """Initialize policy to output values close to the hardcoded defaults."""
        default_scaled = scale_to_network(DEFAULT_PARAMS)
        # Inverse sigmoid to set bias so sigmoid(bias) ≈ default_scaled
        # sigmoid(x) = s => x = log(s / (1-s))
        eps = 0.01
        default_clamped = default_scaled.clamp(eps, 1 - eps)
        logit_init = torch.log(default_clamped / (1 - default_clamped))
        with torch.no_grad():
            self.mean_head.bias.copy_(logit_init)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: [batch, state_dim] or [state_dim]
        
        Returns:
            mean: [batch, param_dim] in [0, 1] (after sigmoid)
            log_std: [param_dim]
            value: [batch, 1]
        """
        h = self.shared(state)
        mean_raw = self.mean_head(h)
        mean = torch.sigmoid(mean_raw)  # scale to [0, 1]
        value = self.value_head(h)
        return mean, self.log_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy.
        
        Returns:
            action: [param_dim] in [0, 1]
            log_prob: scalar
            value: scalar
        """
        mean, log_std, value = self(state)
        
        if deterministic:
            action = mean
            return action, torch.tensor(0.0), value.squeeze(-1)
        
        std = log_std.exp().clamp(0.01, 1.0)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(0.0, 1.0)  # keep in valid range
        log_prob = dist.log_prob(action).sum()
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probs and values for given states and actions.
        
        Used for PPO update to compute ratio = exp(new_log_prob - old_log_prob).
        """
        mean, log_std, values = self(states)
        std = log_std.exp().clamp(0.01, 1.0)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy


@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    state: np.ndarray
    action: np.ndarray
    log_prob: float
    reward: float  # immediate (0 for all steps except last)
    value: float
    done: bool


@dataclass
class Trajectory:
    """Complete episode trajectory."""
    steps: List[TrajectoryStep]
    episode_return: float  # total net score
    episode_length: int


def compute_gae(
    trajectory: Trajectory,
    policy: MASKPolicyNet,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.
    
    Returns:
        advantages: [T] array
        returns: [T] array (targets for value function)
    """
    T = len(trajectory.steps)
    advantages = np.zeros(T)
    returns = np.zeros(T)
    
    # Last step advantage = episode_return - V(s_T)
    device = next(policy.parameters()).device
    last_state = torch.tensor(trajectory.steps[-1].state, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, last_value = policy(last_state)
    last_value = last_value.item()
    
    next_value = last_value if not trajectory.steps[-1].done else 0.0
    
    gae = 0.0
    for t in reversed(range(T)):
        step = trajectory.steps[t]
        delta = step.reward + gamma * next_value * (1 - step.done) - step.value
        gae = delta + gamma * lambda_gae * (1 - step.done) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + step.value
        next_value = step.value
    
    return advantages, returns


class PPOBuffer:
    """Buffer for collecting trajectories and preparing PPO updates."""
    
    def __init__(self):
        self.trajectories: List[Trajectory] = []
    
    def add(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)
    
    def clear(self):
        self.trajectories = []
    
    def get_training_data(
        self,
        policy: MASKPolicyNet,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Prepare batched training data from all trajectories.
        
        Returns dict with keys: states, actions, old_log_probs, advantages, returns
        """
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for traj in self.trajectories:
            if not traj.steps:
                continue
            
            # Compute GAE for this trajectory
            advantages, returns = compute_gae(traj, policy, gamma, lambda_gae)
            
            for i, step in enumerate(traj.steps):
                all_states.append(step.state)
                all_actions.append(step.action)
                all_old_log_probs.append(step.log_prob)
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])
        
        if not all_states:
            return {}
        
        # Convert to tensors (on same device as policy)
        device = next(policy.parameters()).device
        states = torch.tensor(np.array(all_states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(all_actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
        advantages = torch.tensor(all_advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(all_returns, dtype=torch.float32, device=device)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "returns": returns,
        }
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    @property
    def total_steps(self) -> int:
        return sum(len(t.steps) for t in self.trajectories)
    
    @property
    def avg_return(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(t.episode_return for t in self.trajectories) / len(self.trajectories)
