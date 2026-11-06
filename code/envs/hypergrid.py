"""
Unified HyperGrid Environment for GFlowNets

This module provides a configurable HyperGrid environment that can be customized
with different dimensions, heights, and reward parameters. It replaces the need
for separate hypergrid_simple, hypergrid_mild, and hypergrid_hard files.

Presets:
- simple: ndim=4, height=8, R0=0.1, R1=0.5, R2=2.0 (easier exploration)
- mild: ndim=4, height=16, R0=0.001, R1=0.5, R2=2.0
- hard: ndim=4, height=32, R0=0.00001, R1=0.5, R2=2.0
"""

import torch
import numpy as np
import gymnasium as gym
from einops import rearrange
from torch.nn.functional import one_hot
from typing import Optional, Tuple, Dict, Any, List


# Preset configurations
PRESETS = {
    'simple': {'ndim': 4, 'height': 8, 'R0': 0.1, 'R1': 0.5, 'R2': 2.0},
    'mild': {'ndim': 4, 'height': 16, 'R0': 1e-3, 'R1': 0.5, 'R2': 2.0},
    'hard': {'ndim': 4, 'height': 32, 'R0': 1e-5, 'R1': 0.5, 'R2': 2.0},
}


class HyperGridEnv(gym.Env):
    """
    Unified HyperGrid environment from the GFlowNets paper.
    
    The states are represented as 1-d tensors of length `ndim` with values in
    {0, 1, ..., height - 1}. Actions are discrete: move in one of ndim dimensions
    or take the exit action.
    
    The reward function creates a ring-shaped reward structure in the normalized
    state space, with different reward levels for different regions.
    
    Args:
        ndim: Number of dimensions in the grid
        height: Height of the grid in each dimension
        R0: Base reward (minimum reward for any terminal state)
        R1: Reward for states in the outer ring (|x - 0.5| > 0.25)
        R2: Reward for states in the middle ring (0.3 < |x - 0.5| < 0.4)
        preset: Optional preset name ('simple', 'mild', 'hard') to use default params
        render_mode: Rendering mode (currently unused)
    
    Examples:
        >>> # Create a simple environment
        >>> env = HyperGridEnv(preset='simple')
        
        >>> # Create a custom environment
        >>> env = HyperGridEnv(ndim=3, height=32, R0=0.01, R1=0.5, R2=2.0)
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 ndim: int = 2, 
                 height: int = 64, 
                 R0: float = 1e-10, 
                 R1: float = 0, 
                 R2: float = 2.0, 
                 preset: Optional[str] = None,
                 render_mode: Optional[str] = None):
        
        # Apply preset if specified
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}")
            config = PRESETS[preset]
            ndim = config['ndim']
            height = config['height']
            R0 = config['R0']
            R1 = config['R1']
            R2 = config['R2']
        
        self.ndim = ndim
        self.height = height
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.render_mode = render_mode

        # Coordinates, rewards, and target probabilities use float64. Policy
        # observations remain float32: one-hot 0/1 entries are represented exactly.
        # Observation is one-hot encoded: ndim * height
        self.observation_space = gym.spaces.Box(
            shape=(ndim * height,), low=0, high=1, dtype=np.float32
        )
        # Actions: move in each dimension + exit action
        self.action_space = gym.spaces.Discrete(ndim + 1)

        self.state = np.zeros(self.ndim, dtype=np.float64)
        
        # Final state in GFlowNets: all-zeros observation (absorbing state)
        self.final_state = np.zeros(ndim * height, dtype=np.float32)
        
        # Pre-compute the true distribution for validation
        self.truth_dist = self._compute_truth_distribution()

    def _compute_truth_distribution(self) -> np.ndarray:
        """
        Compute the true (unnormalized then normalized) reward distribution.
        
        Returns:
            Normalized probability distribution over all states
        """
        # Create meshgrid of all possible states
        indices = np.arange(self.height, dtype=np.float64)
        grids = np.meshgrid(*[indices] * self.ndim, indexing='ij')
        states = np.stack(grids, axis=-1)
        
        # Use exactly the same float64 predicate as operational rewards and
        # reward-proportional terminal sampling (including strict boundaries).
        rewards = self._compute_rewards(states)
        return rewards / rewards.sum(dtype=np.float64)

    def _compute_rewards(self, states: np.ndarray) -> np.ndarray:
        """Evaluate the shared reward kernel in float64, including batched states."""
        states = np.asarray(states, dtype=np.float64)
        ax = np.abs(states / np.float64(self.height - 1) - np.float64(0.5))
        outer = (np.float64(0.25) < ax).all(axis=-1)
        middle = ((np.float64(0.3) < ax) & (ax < np.float64(0.4))).all(axis=-1)
        return (
            np.float64(self.R0)
            + outer * np.float64(self.R1)
            + middle * np.float64(self.R2)
        )

    def _compute_reward(self, state: np.ndarray) -> float:
        """
        Compute the reward for a given state.
        
        Args:
            state: Current state array
            
        Returns:
            Reward value
        """
        return float(self._compute_rewards(state))

    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.state = np.zeros(self.ndim, dtype=np.float64)
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Get the one-hot encoded observation.
        
        Returns:
            Flattened one-hot encoded state
        """
        state_tensor = torch.tensor(self.state).long()
        obs_tensor = one_hot(state_tensor, num_classes=self.height).float()
        hot = rearrange(obs_tensor, '... h w -> ... (h w)')
        return hot.numpy()
            
    def get_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Decode the state from one-hot encoded observation.
        
        Args:
            obs: One-hot encoded observation(s)
            
        Returns:
            Decoded state array
        """
        state = np.zeros(obs.shape[:-1] + (self.ndim,), dtype=np.float64)
        for i in range(self.ndim):
            state[..., i] = np.argmax(obs[..., i*self.height:(i+1)*self.height], axis=-1)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: Action index (0 to ndim-1 for moves, ndim for exit)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            Note: On exit, returns final_state observation; reward is for the 
            state before transition to final_state.
        """
        # Exit action - transition to final state
        if action == self.ndim:
            # Reward is computed for current state (before final state)
            reward = self._compute_reward(self.state)
            # Return final_state observation (absorbing state)
            return self.final_state.copy(), reward, True, False, {}
        
        # Move action
        if 0 <= action < self.ndim:
            self.state[action] = min(self.state[action] + 1, self.height - 1)
        
        return self._get_obs(), 0, False, False, {}
    
    def is_final_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Check if observation(s) represent the final state.
        
        Args:
            obs: Observation(s) to check
            
        Returns:
            Boolean array indicating which observations are final states
        """
        if len(obs.shape) == 1:
            return np.allclose(obs, self.final_state)
        else:
            return np.all(obs == 0, axis=-1)
    
    def get_num_of_reachable_end_states(self, obs: np.ndarray) -> int:
        """
        Get the number of terminal states reachable from the given state.
        
        For HyperGrid, this is the product of (height - current_position) for each dimension.
        From state (x_1, ..., x_n), we can reach any terminal state (y_1, ..., y_n) where
        y_i >= x_i for all i. So the count is product of (height - x_i) for each dimension.
        
        Args:
            obs: One-hot encoded observation (1D array)
            
        Returns:
            Number of reachable terminal states
        """
        if len(obs.shape) > 1:
            # Handle single observation from batch
            obs = obs.flatten() if obs.shape[0] == 1 else obs[0]
        
        # Decode state
        state = self.get_state(obs.reshape(1, -1))[0]
        
        # Number of reachable states = product of (height - current_pos) for each dim
        # Since we can move up in each dimension independently
        num_reachable = 1
        for i in range(self.ndim):
            remaining = self.height - int(state[i])
            num_reachable *= remaining
        
        return int(num_reachable)
    
    def get_forward_action_masks(self, state: np.ndarray) -> np.ndarray:
        """
        Get valid forward action mask.
        
        Args:
            state: One-hot encoded state(s)
            
        Returns:
            Boolean mask where True indicates valid action
        """
        # Check if at boundary in each dimension
        selected_ind = [(i + 1) * self.height - 1 for i in range(self.ndim)]
        masks = state[..., selected_ind] != 1
        
        # Exit action is always valid
        if isinstance(state, torch.Tensor):
            exit_mask = torch.ones(list(masks.size())[:-1] + [1], 
                                  device=state.device, dtype=torch.bool)
            masks = torch.cat([masks, exit_mask], dim=-1)
        else:
            exit_mask = np.ones(list(masks.shape)[:-1] + [1])
            masks = np.concatenate([masks, exit_mask], axis=-1)
        return masks
    
    def get_backward_action_masks(self, state: np.ndarray) -> np.ndarray:
        """
        Get valid backward action mask.
        
        Args:
            state: One-hot encoded state(s)
            
        Returns:
            Boolean mask where True indicates valid backward action.
            Shape: (..., ndim + 1) where last element (exit action) is always False
        """
        # Check if at origin in each dimension
        selected_ind = [i * self.height for i in range(self.ndim)]
        masks = state[..., selected_ind] != 1
        
        # Add False for exit action (can't exit backwards)
        if isinstance(state, torch.Tensor):
            exit_mask = torch.zeros(list(masks.size())[:-1] + [1], 
                                  device=state.device, dtype=torch.bool)
            masks = torch.cat([masks, exit_mask], dim=-1)
        else:
            exit_mask = np.zeros(list(masks.shape)[:-1] + [1], dtype=bool)
            masks = np.concatenate([masks, exit_mask], axis=-1)
        return masks

    def apply_backward_action(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        Apply a backward action to the given obs.
        
        Backward action decrements the corresponding dimension.
        
        Args:
            Obs: Obs array
            action: Backward action index (0 to ndim-1)
            
        Returns:
            New state after applying backward action
        """
        state = self.get_state(obs)
        new_state = state.copy()
        new_state[action] = max(0, new_state[action] - 1)
        state_tensor = torch.tensor(new_state).long()
        obs_tensor = one_hot(state_tensor, num_classes=self.height).float()
        hot = rearrange(obs_tensor, '... h w -> ... (h w)')
        return hot.numpy()

    def get_error(self, samples: np.ndarray) -> float:
        """
        Get the L1 error between the sample distribution and true distribution.
        
        Args:
            samples: Array of sampled states
            
        Returns:
            L1 error between empirical and true distributions
        """
        samples = np.array(samples)
        
        # Compute sample distribution
        sample_dist = np.zeros([self.height] * self.ndim)
        for s in samples:
            idx = tuple(int(s[i]) for i in range(self.ndim))
            sample_dist[idx] += 1
        
        sample_dist = sample_dist / sample_dist.sum()
        
        # Compute L1 error
        return np.abs(self.truth_dist - sample_dist).sum()

    def sample_terminal_states(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample terminal states proportional to their rewards.
        
        This method samples grid positions proportional to R(x), allowing
        algorithms like Stable GFlowNet to sample from the target distribution.
        
        Args:
            n_samples: Number of terminal states to sample
            
        Returns:
            Tuple of (observations, rewards):
                - observations: One-hot encoded observations [n_samples, obs_dim]
                - rewards: Corresponding rewards [n_samples]
        """
        # Flatten truth_dist for sampling
        flat_dist = self.truth_dist.flatten()
        
        # Sample indices proportional to rewards
        indices = np.random.choice(len(flat_dist), size=n_samples, p=flat_dist)
        
        # Convert flat indices to multi-dimensional indices
        observations = []
        rewards = []
        
        for idx in indices:
            # Convert flat index to grid coordinates
            state = np.unravel_index(idx, [self.height] * self.ndim)
            state = np.array(state, dtype=np.float64)
            
            # Compute observation
            self.state = state
            obs = self._get_obs()
            observations.append(obs)
            
            # Compute reward
            reward = self._compute_reward(state)
            rewards.append(reward)
        
        # Reset state
        self.state = np.zeros(self.ndim, dtype=np.float64)
        
        return np.array(observations, dtype=np.float32), np.array(rewards, dtype=np.float64)

    def render(self):
        """Render the environment (not implemented)."""
        pass


# Convenience aliases for backwards compatibility
Environment = HyperGridEnv


if __name__ == "__main__":
    from tqdm import trange

    # Test with different presets
    for preset_name in ['simple', 'mild', 'hard']:
        print(f"\n=== Testing {preset_name} preset ===")
        np.random.seed(42)
        
        env = HyperGridEnv(preset=preset_name)
        print(f"ndim={env.ndim}, height={env.height}")
        print(f"R0={env.R0}, R1={env.R1}, R2={env.R2}")
        
        num_samples = 10000
        rs = []
        for _ in trange(num_samples, desc=f"Sampling {preset_name}"):
            done = False
            env.reset()
            while not done:
                valid_actions = env.get_forward_action_masks(env._get_obs())
                action_index = np.random.choice(int(np.sum(valid_actions)))
                action = np.where(valid_actions == 1)[0][action_index]
                _, r, done, _, _ = env.step(action)
            rs.append(r)
        
        rs = np.array(rs)
        print(f"Non-zero rewards: {np.sum(rs > 1e-3)} / {num_samples}")
        print(f"Mean reward: {np.mean(rs):.6f}")
