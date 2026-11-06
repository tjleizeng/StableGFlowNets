"""
Base Environment Interface for GFlowNets

This module provides an abstract base class that serves as a middle layer between
Gymnasium environments and GFlowNets. It defines the interface that GFlowNet
environments should implement.

Required functionalities:
0. Indicate whether the environment is continuous or discrete
1. Get the action mask for forward and backward actions
2. Get the trajectory backwards given end states
3. Get the number of reachable terminating states (optional)
4. Get the validation error based on generated samples
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union
import gymnasium as gym


class GFlowNetEnvBase(ABC, gym.Env):
    """
    Abstract base class for GFlowNet environments.
    
    This class extends Gymnasium's Env class and adds methods required
    for training GFlowNets, including action masking and backward trajectory
    generation.
    
    Attributes:
        is_continuous: Whether the action space is continuous
    """
    
    def __init__(self):
        super().__init__()
        self._is_continuous: Optional[bool] = None
    
    @property
    def is_continuous(self) -> bool:
        """Check if the environment has a continuous action space."""
        if self._is_continuous is None:
            self._is_continuous = isinstance(self.action_space, gym.spaces.Box)
        return self._is_continuous
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info_dict)
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def get_forward_action_masks(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Get a mask indicating valid forward actions from the given state.
        
        This is used to prevent the policy from selecting invalid actions
        (e.g., moving out of bounds, taking actions that violate constraints).
        
        Args:
            state: The current state observation(s). Can be batched.
            
        Returns:
            Boolean mask where True indicates a valid action.
            Shape: (..., num_actions) for discrete, (..., 2 * action_dim) for continuous
        """
        pass
    
    @abstractmethod
    def get_backward_action_masks(self, state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Get a mask indicating valid backward actions from the given state.
        
        This is used for computing backward transition probabilities
        in trajectory balance objectives, and backward trajectory sampling.
        
        Args:
            state: The current state observation(s). Can be batched.
            
        Returns:
            Boolean mask where True indicates a valid backward action.
            Shape: (..., num_actions) for discrete, (..., 2 * action_dim) for continuous
        """
        pass
    
    
    def get_num_reachable_states(self, state: Optional[np.ndarray] = None) -> int:
        """
        Get the number of reachable terminal states from the given state.
        
        This is useful for understanding the environment's structure and
        for debugging. If state is None, returns total number of terminal states.
        
        Args:
            state: Optional starting state. If None, counts from initial state.
            
        Returns:
            Number of reachable terminal states
        """
        raise NotImplementedError(
            "get_num_reachable_states is not implemented for this environment. "
            "Override this method if you need this functionality."
        )

    def _get_obs(self) -> np.ndarray:
        """
        Get the observation of the current state.
        
        Returns:
            Observation of the current state
        """
        raise NotImplementedError(
            "_get_obs is not implemented for this environment. "
            "Override this method if observations need to be computed."
        )
    
    def get_error(self, samples: np.ndarray) -> float:
        """
        Compute the validation error between sampled distribution and true distribution.
        
        This is typically the L1 (or other) distance between the empirical distribution
        from samples and the true target distribution (if known).
        
        Args:
            samples: Array of sampled terminal states
            
        Returns:
            Error metric (e.g., L1 distance)
        """
        raise NotImplementedError(
            "get_error is not implemented for this environment. "
            "Override this method for validation during training."
        )
    
    def get_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Decode the internal state representation from an observation.
        
        This is useful when the observation is encoded (e.g., one-hot encoding)
        and you need the raw state values.
        
        Args:
            obs: The observation(s) to decode
            
        Returns:
            Decoded state representation
        """
        raise NotImplementedError(
            "get_state is not implemented for this environment. "
            "Override this method if observations need decoding."
        )
    
    def sample_terminal_states(self, n: int, states: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample n terminal states using importance sampling according to the reward function.

        When state is provided, the sampling is done within the set of states
        
        This is useful for training methods that need
        to sample terminal states for backward trajectory generation.
        
        Args:
            n: Number of states to sample
            states: Optional terminal states. If None, samples from all terminating states.
            
        Returns:
            Array of n sampled terminal states
        """
        raise NotImplementedError(
            "sample_terminal_states is not implemented for this environment. "
            "Override this method for off-policy training with backward sampling."
        )

    def apply_backward_action(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a backward action to the given obs.
        
        Backward action decrements the corresponding dimension.
        
        Args:
            Obs: Obs array
            action: Backward action index (0 to ndim-1)
            
        Returns:
            New state after applying backward action
        """
        raise NotImplementedError(
            "apply_backward_action is not implemented for this environment. "
            "Override this method for backward action application."
        )


class DiscreteGFlowNetEnv(GFlowNetEnvBase):
    """
    Base class for discrete action space GFlowNet environments.
    
    Provides default implementations assuming discrete actions.
    """
    
    def __init__(self):
        super().__init__()
        self._is_continuous = False


# Utility functions for environment creation

def make_gfn_env(env_id: str, **kwargs) -> GFlowNetEnvBase:
    """
    Create a GFlowNet environment by ID.
    
    This is a convenience function similar to gym.make() but ensures
    the environment implements the GFlowNet interface.
    
    Args:
        env_id: Environment identifier
        **kwargs: Additional arguments passed to the environment
        
    Returns:
        GFlowNet environment instance
    """
    env = gym.make(env_id, **kwargs)
    
    # Check if the unwrapped environment has required methods
    unwrapped = env.unwrapped
    required_methods = ['get_forward_action_masks', 'get_backward_action_masks']
    
    for method in required_methods:
        if not hasattr(unwrapped, method):
            raise ValueError(
                f"Environment {env_id} does not implement required method '{method}'. "
                "GFlowNet environments must implement get_forward_action_masks and "
                "get_backward_action_masks methods."
            )
    
    return env
