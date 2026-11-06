"""
Vectorized Environment Wrapper for Parallel Rollout Collection

This module provides a simple vectorized environment wrapper that allows
parallel stepping of multiple environment instances.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional


class VectorizedEnvWrapper:
    """
    Wrapper to make VectorizedEnv compatible with unwrapped attribute access.
    """
    def __init__(self, vectorized_env):
        self.vectorized_env = vectorized_env
        # Expose first environment's unwrapped for compatibility
        self.unwrapped = vectorized_env.envs[0].unwrapped if hasattr(vectorized_env.envs[0], 'unwrapped') else vectorized_env.envs[0]
    
    def __getattr__(self, name):
        # Delegate to vectorized_env
        return getattr(self.vectorized_env, name)


class VectorizedEnv:
    """
    A simple vectorized environment wrapper that runs multiple environment instances in parallel.
    
    This wrapper batches operations across multiple environment instances, allowing
    efficient parallel rollout collection.
    
    Args:
        env_factory: Function that creates a new environment instance
        num_envs: Number of parallel environment instances
    """
    
    def __init__(self, env_factory, num_envs: int):
        self.num_envs = num_envs
        self.envs = [env_factory() for _ in range(num_envs)]
        
        # Get observation and action spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Current observations for all environments
        self.observations = None
        self.infos = None
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset all environments.
        
        Returns:
            Tuple of (batched observations and infos)
        """
        if seed is not None:
            # Set seeds for each environment
            for i, env in enumerate(self.envs):
                env.reset(seed=seed + i)
        
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        
        self.observations = np.array(obs_list)
        self.infos = info_list
        
        return self.observations, self.infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Step all environments in parallel.
        
        Args:
            actions: Array of actions for each environment, shape (num_envs,)
            
        Returns:
            Tuple of (observations, rewards, dones, truncateds, infos)
        """
        obs_list = []
        rew_list = []
        done_list = []
        truncated_list = []
        info_list = []
        
        for i, env in enumerate(self.envs):
            obs, rew, done, truncated, info = env.step(actions[i])
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            truncated_list.append(truncated)
            info_list.append(info)
            
            # Auto-reset if done
            if done or truncated:
                obs, _ = env.reset()
                obs_list[i] = obs
        
        self.observations = np.array(obs_list)
        rewards = np.array(rew_list)
        dones = np.array(done_list)
        truncateds = np.array(truncated_list)
        
        return self.observations, rewards, dones, truncateds, info_list
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

