"""
Regular Tree Environment for GFlowNets

This module provides a regular tree environment where each non-leaf node has
exactly g children and the tree has maximum depth h. All leaf nodes have
uniform reward of 1.

The state represents the path from root to current node, encoded as which
child was taken at each level.
"""

import torch
import numpy as np
import gymnasium as gym
from torch.nn.functional import one_hot
from typing import Optional, Tuple, Dict, List


class RegularTreeEnv(gym.Env):
    """
    Regular tree environment for GFlowNets.
    
    A complete g-ary tree of depth h where:
    - Each non-leaf node has exactly g children
    - Leaf nodes are at depth h
    - All leaf nodes have reward 1
    
    State representation:
    - Path from root encoded as one-hot vectors at each level
    - At each level i, the state indicates which child (0 to g-1) was taken
    - One-hot encoding: state_dim = h * g (each level has g possible values)
    
    Actions:
    - 0 to g-1: Move to child i
    - g: Exit/terminate
    
    Args:
        g: Branching factor (number of children per node)
        h: Maximum depth of the tree
        reward: Reward for reaching a leaf node (default 1.0)
        render_mode: Rendering mode (currently unused)
    
    Examples:
        >>> env = RegularTreeEnv(g=2, h=3)  # Binary tree of depth 3
        >>> env = RegularTreeEnv(g=3, h=4)  # Ternary tree of depth 4
    """
    metadata = {"render_modes": []}

    def __init__(self, 
                 g: int = 2, 
                 h: int = 3, 
                 reward: float = 1.0,
                 min_reward: float = 1e-10,
                 render_mode: Optional[str] = None):
        
        assert g >= 2, "Branching factor g must be at least 2"
        assert h >= 1, "Depth h must be at least 1"
        
        self.g = g  # branching factor
        self.h = h  # max depth
        self.reward_value = reward
        self.min_reward = min_reward  # minimum reward for early termination
        self.render_mode = render_mode
        
        # State dimension: h levels, each with g possible children (one-hot)
        # Plus 1 bit per level to indicate if we've reached that level
        self.state_dim = h * (g + 1)  # g children + 1 "not visited" indicator
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            shape=(self.state_dim,), low=0, high=1, dtype=np.float32
        )
        
        # Action space: g children + exit
        self.action_space = gym.spaces.Discrete(g + 1)
        
        # Current depth and path
        self.depth = 0
        self.path = np.full(h, -1, dtype=np.int32)  # -1 means not visited
        
        # Final state in GFlowNets: all-zeros observation (absorbing state)
        self.final_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Compute true distribution for validation
        self.truth_dist = self._compute_truth_distribution()
        self.num_leaves = g ** h  # Total number of leaf nodes
        
    def _compute_truth_distribution(self) -> np.ndarray:
        """
        Compute the true (uniform) distribution over leaf nodes.
        All leaves have equal probability since they all have reward 1.
        """
        num_leaves = self.g ** self.h
        return np.ones(num_leaves) / num_leaves
    
    def _path_to_leaf_index(self, path: np.ndarray) -> int:
        """Convert a path to a unique leaf index."""
        index = 0
        for i in range(self.h):
            if path[i] < 0:
                return -1  # Not a complete path
            index = index * self.g + path[i]
        return index
    
    def _leaf_index_to_path(self, index: int) -> np.ndarray:
        """Convert a leaf index to a path."""
        path = np.zeros(self.h, dtype=np.int32)
        for i in range(self.h - 1, -1, -1):
            path[i] = index % self.g
            index //= self.g
        return path

    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to the root node."""
        super().reset(seed=seed)
        self.depth = 0
        self.path = np.full(self.h, -1, dtype=np.int32)
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """
        Get the one-hot encoded observation.
        
        For each level, we have g+1 possible states:
        - Index 0 to g-1: child i was taken
        - Index g: not visited yet (haven't reached this level)
        """
        obs = np.zeros(self.state_dim, dtype=np.float32)
        
        for level in range(self.h):
            offset = level * (self.g + 1)
            if level < self.depth:
                # We've passed this level, mark which child was taken
                obs[offset + self.path[level]] = 1.0
            else:
                # Haven't reached this level yet
                obs[offset + self.g] = 1.0
                
        return obs
    
    def get_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Decode the path from observation.
        
        Args:
            obs: One-hot encoded observation(s)
            
        Returns:
            Path array(s) where each element is 0 to g-1 or -1 (not visited)
        """
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
            
        batch_size = obs.shape[0]
        paths = np.full((batch_size, self.h), -1, dtype=np.int32)
        
        for level in range(self.h):
            offset = level * (self.g + 1)
            level_obs = obs[:, offset:offset + self.g + 1]
            # Find which index is 1 (argmax)
            indices = np.argmax(level_obs, axis=-1)
            # If index is g, it's "not visited" -> keep -1
            # Otherwise, it's the child index
            mask = indices < self.g
            paths[mask, level] = indices[mask]
            
        if squeeze:
            return paths[0]
        return paths

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: Action index (0 to g-1 for children, g for exit)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Exit action (only valid at leaf nodes) - transition to final state
        if action == self.g:
            # Exit should only be called at leaf (depth == h)
            # This is enforced by action masking, but keep check as safety
            if self.depth == self.h:
                reward = self.reward_value
            else:
                # Should not happen with proper masking, but handle gracefully
                reward = self.min_reward
            # Return final_state observation (absorbing state)
            # Reward is for the state before transition to final_state
            return self.final_state.copy(), reward, True, False, {}
        
        # Child action: move to child 'action'
        if 0 <= action < self.g and self.depth < self.h:
            self.path[self.depth] = action
            self.depth += 1
            
            # Auto-terminate at leaf? No, let the agent decide to exit
            return self._get_obs(), 0.0, False, False, {}
        
        # Invalid action (shouldn't happen with proper masking)
        return self._get_obs(), 0.0, False, False, {}
    
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
        Get the number of terminal states (leaf nodes) reachable from the given state.
        
        For a regular tree with branching factor g and max depth h,
        from depth d, the number of reachable leaves is g^(h - d).
        
        Args:
            obs: One-hot encoded observation (1D array)
            
        Returns:
            Number of reachable terminal (leaf) states
        """
        if len(obs.shape) > 1:
            # Handle single observation from batch
            obs = obs.flatten() if obs.shape[0] == 1 else obs[0]
        
        # Determine current depth from observation
        # The "not visited" indicator at each level tells us if we've been there
        current_depth = 0
        for level in range(self.h):
            offset = level * (self.g + 1)
            level_obs = obs[offset:offset + self.g + 1]
            # If the "not visited" indicator (index g) is 1, we haven't reached this level
            if level_obs[self.g] == 1:
                break
            current_depth = level + 1
        
        # Number of reachable leaves = g^(h - current_depth)
        remaining_depth = self.h - current_depth
        num_reachable = self.g ** remaining_depth
        
        return int(num_reachable)
    
    def get_forward_action_masks(self, state: np.ndarray) -> np.ndarray:
        """
        Get valid forward action mask.
        
        Args:
            state: One-hot encoded state(s). Can be 1D, 2D [batch, state_dim], 
                   or 3D [batch, seq_len, state_dim]
            
        Returns:
            Boolean mask where True indicates valid action.
            Shape: (..., g + 1)
        """
        if isinstance(state, torch.Tensor):
            device = state.device
            original_shape = state.shape
            
            # Handle different input dimensions
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                output_shape = (self.g + 1,)
            elif len(state.shape) == 2:
                output_shape = (state.shape[0], self.g + 1)
            elif len(state.shape) == 3:
                # 3D: [batch, seq_len, state_dim]
                batch_size, seq_len, state_dim = state.shape
                state = state.reshape(-1, state_dim)  # Flatten to 2D
                output_shape = (batch_size, seq_len, self.g + 1)
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
            
            flat_batch_size = state.shape[0]
            masks = torch.zeros((flat_batch_size, self.g + 1), dtype=torch.bool, device=device)
            
            # Check if at leaf (last level is visited)
            last_level_offset = (self.h - 1) * (self.g + 1)
            not_visited_last = state[:, last_level_offset + self.g] == 1
            at_leaf = ~not_visited_last
            
            # At leaf: only exit action is valid
            masks[at_leaf, self.g] = True
            
            # Not at leaf: only child actions are valid
            masks[~at_leaf, :self.g] = True
            
            return masks.reshape(output_shape)
        else:
            # NumPy version
            original_shape = state.shape
            
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
                output_shape = (self.g + 1,)
            elif len(state.shape) == 2:
                output_shape = (state.shape[0], self.g + 1)
            elif len(state.shape) == 3:
                batch_size, seq_len, state_dim = state.shape
                state = state.reshape(-1, state_dim)
                output_shape = (batch_size, seq_len, self.g + 1)
            else:
                raise ValueError(f"Unexpected state shape: {original_shape}")
                
            flat_batch_size = state.shape[0]
            masks = np.zeros((flat_batch_size, self.g + 1), dtype=bool)
            
            # Check if at leaf
            last_level_offset = (self.h - 1) * (self.g + 1)
            not_visited_last = state[:, last_level_offset + self.g] == 1
            at_leaf = ~not_visited_last
            
            # At leaf: only exit action is valid
            masks[at_leaf, self.g] = True
            
            # Not at leaf: only child actions are valid
            masks[~at_leaf, :self.g] = True
                    
            return masks.reshape(output_shape)
    
    def get_backward_action_masks(self, state: np.ndarray) -> np.ndarray:
        """
        Get valid backward action mask.
        
        For backward, we can only "undo" the last action taken.
        
        Args:
            state: One-hot encoded state(s). Can be 1D, 2D [batch, state_dim],
                   or 3D [batch, seq_len, state_dim]
            
        Returns:
            Boolean mask where True indicates valid backward action.
            Shape: (..., g + 1) - exit action is always False for backward
        """
        if isinstance(state, torch.Tensor):
            device = state.device
            original_shape = state.shape
            
            # Handle different input dimensions
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                output_shape = (self.g + 1,)
            elif len(state.shape) == 2:
                output_shape = (state.shape[0], self.g + 1)
            elif len(state.shape) == 3:
                batch_size, seq_len, state_dim = state.shape
                state = state.reshape(-1, state_dim)
                output_shape = (batch_size, seq_len, self.g + 1)
            else:
                raise ValueError(f"Unexpected state shape: {state.shape}")
                
            flat_batch_size = state.shape[0]
            # Start with all False (no backward actions valid by default)
            masks = torch.zeros((flat_batch_size, self.g + 1), dtype=torch.bool, device=device)
            
            # For each level, check if it's the last visited level and mark that action
            for level in range(self.h):
                offset = level * (self.g + 1)
                # Check if this level is visited
                not_visited = state[:, offset + self.g] == 1
                
                if level == 0:
                    # At level 0, check if visited
                    visited = ~not_visited
                    # Find which child was taken
                    for c in range(self.g):
                        child_taken = (state[:, offset + c] == 1) & visited
                        # Check if next level is not visited (this is the last level)
                        if level < self.h - 1:
                            next_offset = (level + 1) * (self.g + 1)
                            is_last = state[:, next_offset + self.g] == 1
                            masks[child_taken & is_last, c] = True
                        else:
                            masks[child_taken, c] = True
                else:
                    # For later levels
                    visited = ~not_visited
                    for c in range(self.g):
                        child_taken = (state[:, offset + c] == 1) & visited
                        if level < self.h - 1:
                            next_offset = (level + 1) * (self.g + 1)
                            is_last = state[:, next_offset + self.g] == 1
                            masks[child_taken & is_last, c] = True
                        else:
                            # Last level, this is always "last"
                            masks[child_taken, c] = True
                    
            return masks.reshape(output_shape)
        else:
            # NumPy version
            original_shape = state.shape
            
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
                output_shape = (self.g + 1,)
            elif len(state.shape) == 2:
                output_shape = (state.shape[0], self.g + 1)
            elif len(state.shape) == 3:
                batch_size, seq_len, state_dim = state.shape
                state = state.reshape(-1, state_dim)
                output_shape = (batch_size, seq_len, self.g + 1)
            else:
                raise ValueError(f"Unexpected state shape: {original_shape}")
                
            flat_batch_size = state.shape[0]
            masks = np.zeros((flat_batch_size, self.g + 1), dtype=bool)
            
            for i in range(flat_batch_size):
                depth = 0
                last_action = -1
                for level in range(self.h):
                    offset = level * (self.g + 1)
                    if state[i, offset + self.g] == 1:  # Not visited
                        break
                    # Find which child was taken
                    for c in range(self.g):
                        if state[i, offset + c] == 1:
                            last_action = c
                            break
                    depth = level + 1
                    
                if depth > 0 and last_action >= 0:
                    masks[i, last_action] = True
                    
            return masks.reshape(output_shape)
    
    def apply_backward_action(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        Apply a backward action to the given obs.
        
        Backward action moves up the tree by decrementing depth and 
        marking the current level as unvisited.
        
        Args:
            obs: One-hot encoded observation
            action: Backward action index (0 to g-1, indicates which child to undo)
            
        Returns:
            New observation after applying backward action
        """
        path = self.get_state(obs)
        new_path = path.copy()
        
        # Find current depth (first -1 position or h if fully traversed)
        depth = 0
        for i in range(len(new_path)):
            if new_path[i] == -1:
                break
            depth = i + 1
        
        # Move up the tree by marking the last position as unvisited
        if depth > 0:
            new_path[depth - 1] = -1
            depth -= 1  # new depth after backward action
        
        # Convert new_path back to observation
        new_obs = np.zeros(self.state_dim, dtype=np.float32)
        for level in range(self.h):
            offset = level * (self.g + 1)
            if level < depth:
                # We've passed this level, mark which child was taken
                new_obs[offset + new_path[level]] = 1.0
            else:
                # Haven't reached this level yet
                new_obs[offset + self.g] = 1.0
                
        return new_obs

    def get_error(self, samples: np.ndarray) -> float:
        """
        Compute L1 error between sample distribution and true distribution.
        
        Args:
            samples: Array of sampled paths (each row is a path)
            
        Returns:
            L1 error
        """
        if len(samples.shape) == 1:
            samples = samples.reshape(1, -1)
            
        # Count visits to each leaf
        sample_dist = np.zeros(self.num_leaves)
        for path in samples:
            idx = self._path_to_leaf_index(path)
            if idx >= 0:
                sample_dist[idx] += 1
                
        # Normalize
        if sample_dist.sum() > 0:
            sample_dist = sample_dist / sample_dist.sum()
            
        return np.abs(self.truth_dist - sample_dist).sum()
    
    def get_num_reachable_states(self, state: Optional[np.ndarray] = None) -> int:
        """Get total number of leaf nodes."""
        return self.num_leaves
    
    def sample_terminal_states(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample terminal states (leaf nodes) uniformly.
        
        All leaf nodes have equal reward, so sampling is uniform.
        
        Args:
            n_samples: Number of terminal states to sample
            
        Returns:
            Tuple of (observations, rewards):
                - observations: One-hot encoded observations [n_samples, obs_dim]
                - rewards: Corresponding rewards [n_samples] (all 1.0)
        """
        observations = []
        rewards = []
        
        for _ in range(n_samples):
            # Sample a random leaf by choosing random path
            path = np.random.randint(0, self.g, size=self.h)
            
            # Create observation for this leaf
            obs = np.zeros(self.state_dim, dtype=np.float32)
            for level in range(self.h):
                child = path[level]
                # Set the one-hot encoding for this level
                obs[level * (self.g + 1) + child] = 1.0
            
            observations.append(obs)
            rewards.append(self.reward_value)
        
        return np.array(observations), np.array(rewards)
    
    def render(self):
        """Render the current state (prints path)."""
        print(f"Depth: {self.depth}, Path: {self.path[:self.depth]}")


# Convenience alias
Environment = RegularTreeEnv


if __name__ == "__main__":
    # Test the environment
    print("Testing RegularTreeEnv")
    
    # Binary tree of depth 3
    env = RegularTreeEnv(g=2, h=3)
    print(f"\nBinary tree (g=2, h=3)")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_space.n}")
    print(f"  Num leaves: {env.num_leaves}")
    
    # Random rollout
    obs, _ = env.reset()
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial path: {env.path}")
    
    done = False
    steps = 0
    while not done:
        mask = env.get_forward_action_masks(obs)
        valid_actions = np.where(mask)[0]
        action = np.random.choice(valid_actions)
        obs, reward, done, _, info = env.step(action)
        steps += 1
        print(f"  Action {action}, Depth {env.depth}, Reward {reward}")
    
    print(f"\nFinal path: {env.path}")
    print(f"Total steps: {steps}")
    
    # Test multiple rollouts
    print("\nTesting 1000 random rollouts...")
    rewards = []
    for _ in range(1000):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_forward_action_masks(obs)
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    
    print(f"  Mean reward: {np.mean(rewards):.4f}")
    print(f"  All rewards = 1: {all(r == 1.0 for r in rewards)}")

