import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from utils.distributions import MaskableCategoricalDistribution


class DiscreteForwardPolicy(nn.Module):
    """
    Forward policy for discrete action spaces in GFlowNets.
    
    Outputs action logits that are converted to a categorical distribution.
    Supports action masking for invalid actions and epsilon-greedy exploration.
    
    Args:
        state_dim: Dimension of the input state space
        action_dim: Number of discrete actions (including exit action if applicable)
        hidden_sizes: List of hidden layer sizes
        activation_fn: Activation function class (e.g., torch.nn.ReLU)
        device: Device to place the model on
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_sizes: List[int] = [128, 128],
                 activation_fn: type = torch.nn.ReLU,
                 device: str = "cpu"):
        super(DiscreteForwardPolicy, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Create the action network
        layers = []
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(activation_fn())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation_fn())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))

        self.action_net = nn.Sequential(*layers)
        self.distribution = MaskableCategoricalDistribution(self.action_dim)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get raw action logits from the network.
        
        Args:
            obs: State observation tensor
            
        Returns:
            Action logits tensor of shape (..., action_dim)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        return self.action_net(obs)

    def forward(self, obs: Union[torch.Tensor, list], 
                action_masks: Optional[torch.Tensor] = None, 
                epsilon: float = 0.0,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Sample actions from the policy.
        
        Args:
            obs: State observation
            action_masks: Boolean mask for valid actions (True = valid)
            epsilon: Probability of random action for exploration
            temperature: Temperature for softmax (higher = more uniform)
            
        Returns:
            Sampled action indices
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        
        action_logits = self.action_net(obs)
        
        # Apply temperature scaling
        if temperature != 1.0:
            action_logits = action_logits / temperature

        if epsilon > 0.0 and len(action_logits.shape) >= 2:
            batch_size = action_logits.shape[0]
            # Create epsilon mask
            rand_mask = (torch.rand(batch_size, device=obs.device) < epsilon).float().unsqueeze(1)
            # Create uniform logits over valid actions
            uniform_logits = torch.zeros_like(action_logits)
            # Blend original logits and uniform logits
            action_logits = (1 - rand_mask) * action_logits + rand_mask * uniform_logits

        # clip the logits to -50 to 50
        action_logits = torch.clamp(action_logits, min=-50, max=50)

        self.distribution.proba_distribution(action_logits, action_masks)
        actions = self.distribution.get_actions()

        if len(actions.shape) == 2 and actions.shape[0] == 1:
            actions = actions[0]

        return actions

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor,
                 action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability of actions given observations.
        
        Args:
            obs: State observations
            actions: Actions taken
            action_masks: Boolean mask for valid actions
            
        Returns:
            Log probabilities of the actions
        """
        action_logits = self.action_net(obs)
        # Clamp logits BEFORE computing distribution
        action_logits = torch.clamp(action_logits, min=-50, max=50)
        self.distribution.proba_distribution(action_logits, action_masks)
        log_prob = self.distribution.log_prob(actions)
        # Clamp log probabilities to prevent extreme values
        return torch.clamp(log_prob, min=-50.0)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        action_masks: Optional[torch.Tensor] = None, 
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate log probabilities of actions, with optional trajectory masking.
        
        Args:
            obs: State observations of shape (batch, seq_len, state_dim)
            actions: Actions of shape (batch, seq_len)
            action_masks: Optional action masks
            lengths: Optional trajectory lengths for masking
            
        Returns:
            Log probabilities, masked if lengths provided
        """
        action_logits = self.action_net(obs)
        
        # Clamp logits BEFORE computing distribution to prevent numerical instability
        # This is especially important for larger networks (e.g., 1024x1024)
        action_logits = torch.clamp(action_logits, min=-50, max=50)
        
        self.distribution.proba_distribution(action_logits, action_masks)
        log_prob = self.distribution.log_prob(actions)
        
        # Also clamp log probabilities directly to prevent extreme values
        # This prevents the TB loss from having extremely large values
        log_prob = torch.clamp(log_prob, min=-50.0)

        if lengths is not None:
            # create and apply a mask based on the lengths
            batch_size, seq_len = obs.size(0), obs.size(1)
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(obs.device) < lengths.unsqueeze(1)
            log_prob = log_prob * mask

        return log_prob
    
class DiscreteBackwardPolicy(nn.Module):
    """
    Backward policy for discrete action spaces in GFlowNets.
    
    Used for computing backward transition probabilities in trajectory balance
    and sub-trajectory balance objectives.
    
    Args:
        state_dim: Dimension of the input state space
        action_dim: Number of discrete backward actions
        hidden_sizes: List of hidden layer sizes
        activation_fn: Activation function class
        device: Device to place the model on
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_sizes: List[int] = [128, 128],
                 activation_fn: type = torch.nn.ReLU,
                 device: str = "cpu"):
        super(DiscreteBackwardPolicy, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Create the action network
        layers = []
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(activation_fn())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation_fn())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_dim))

        self.action_net = nn.Sequential(*layers)
        self.distribution = MaskableCategoricalDistribution(self.action_dim)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """Get raw action logits from the network."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        return self.action_net(obs)

    def forward(self, obs: Union[torch.Tensor, list], 
                action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample backward actions from the policy.
        
        Args:
            obs: State observation
            action_masks: Boolean mask for valid backward actions
            
        Returns:
            Sampled action indices
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        action_logits = self.action_net(obs)

        # clip the logits to -50 to 50
        action_logits = torch.clamp(action_logits, min=-50, max=50)

        self.distribution.proba_distribution(action_logits, action_masks)
        actions = self.distribution.get_actions()
        
        if len(actions.shape) == 2 and actions.shape[0] == 1:
            actions = actions[0]

        return actions

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor,
                 action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability of backward actions.
        
        Args:
            obs: State observations
            actions: Backward actions taken
            action_masks: Boolean mask for valid actions
            
        Returns:
            Log probabilities of the actions
        """
        action_logits = self.action_net(obs)
        # Clamp logits BEFORE computing distribution
        action_logits = torch.clamp(action_logits, min=-50, max=50)
        self.distribution.proba_distribution(action_logits, action_masks)
        log_prob = self.distribution.log_prob(actions)
        # Clamp log probabilities to prevent extreme values
        return torch.clamp(log_prob, min=-50.0)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        action_masks: Optional[torch.Tensor] = None, 
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate log probabilities of backward actions with trajectory masking.
        
        Args:
            obs: State observations of shape (batch, seq_len, state_dim)
            actions: Actions of shape (batch, seq_len)
            action_masks: Optional action masks
            lengths: Trajectory lengths (last transition is ignored)
            
        Returns:
            Log probabilities, masked appropriately
        """
        action_logits = self.action_net(obs)
        
        # Clamp logits BEFORE computing distribution to prevent numerical instability
        # This is especially important for larger networks (e.g., 1024x1024)
        action_logits = torch.clamp(action_logits, min=-50, max=50)
        
        self.distribution.proba_distribution(action_logits, action_masks)
        log_prob = self.distribution.log_prob(actions)
        
        # Also clamp log probabilities directly to prevent extreme values
        # This prevents the TB loss from having extremely large values
        log_prob = torch.clamp(log_prob, min=-50.0)

        # create and apply a mask based on the lengths
        batch_size, seq_len = obs.size(0), obs.size(1)
        # lengths - 1 for ignoring the last transition (from terminal state)
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(obs.device) < (lengths - 1).unsqueeze(1)
        log_prob = log_prob * mask

        return log_prob


class UniformBackwardPolicy(nn.Module):
    """
    Uniform backward policy for discrete action spaces in GFlowNets.
    
    Returns uniform probabilities over valid actions (given action masks).
    This is a non-trainable policy with no parameters.
    
    Args:
        action_dim: Number of discrete backward actions
        device: Device to place tensors on
    """
    def __init__(self, action_dim: int, device: str = "cpu"):
        super(UniformBackwardPolicy, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.distribution = MaskableCategoricalDistribution(self.action_dim)

    def forward(self, obs: Union[torch.Tensor, list], 
                action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample backward actions from uniform distribution over valid actions.
        
        Args:
            obs: State observation (not used, but kept for interface compatibility)
            action_masks: Boolean mask for valid backward actions (True = valid)
            
        Returns:
            Sampled action indices
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        
        # Determine batch size
        if len(obs.shape) == 1:
            batch_size = 1
            uniform_logits = torch.zeros((1, self.action_dim), device=self.device)
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
            uniform_logits = torch.zeros((batch_size, self.action_dim), device=self.device)
        else:
            batch_size = obs.shape[0]
            uniform_logits = torch.zeros((batch_size, self.action_dim), device=self.device)
        
        # Create uniform logits (zeros = uniform after softmax)
        self.distribution.proba_distribution(uniform_logits, action_masks)
        actions = self.distribution.get_actions()
        
        if len(actions.shape) == 2 and actions.shape[0] == 1:
            actions = actions[0]
        
        return actions

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor,
                 action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log probability of backward actions under uniform distribution.
        
        Args:
            obs: State observations (not used, but kept for interface compatibility)
            actions: Backward actions taken
            action_masks: Boolean mask for valid actions
            
        Returns:
            Log probabilities of the actions (uniform over valid actions)
        """
        # Determine shape
        if len(obs.shape) == 1:
            uniform_logits = torch.zeros((1, self.action_dim), device=obs.device)
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
            uniform_logits = torch.zeros((batch_size, self.action_dim), device=obs.device)
        else:
            batch_size, seq_len = obs.shape[0], obs.shape[1]
            uniform_logits = torch.zeros((batch_size, seq_len, self.action_dim), device=obs.device)
        
        # clip the logits to -50 to 50
        uniform_logits = torch.clamp(uniform_logits, min=-50, max=50)

        # Create uniform logits (zeros = uniform after softmax)
        self.distribution.proba_distribution(uniform_logits, action_masks)
        return self.distribution.log_prob(actions)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, 
                        action_masks: Optional[torch.Tensor] = None, 
                        lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluate log probabilities of backward actions with uniform distribution.
        
        Args:
            obs: State observations of shape (batch, seq_len, state_dim)
            actions: Actions of shape (batch, seq_len)
            action_masks: Optional action masks
            lengths: Trajectory lengths (last transition is ignored)
            
        Returns:
            Log probabilities, masked appropriately
        """
        batch_size, seq_len = obs.size(0), obs.size(1)
        
        # Create uniform logits (zeros = uniform after softmax)
        uniform_logits = torch.zeros((batch_size, seq_len, self.action_dim), device=obs.device)

        # clip the logits to -50 to 50
        uniform_logits = torch.clamp(uniform_logits, min=-50, max=50)
        
        self.distribution.proba_distribution(uniform_logits, action_masks)
        log_prob = self.distribution.log_prob(actions)
        
        # create and apply a mask based on the lengths
        # lengths - 1 for ignoring the last transition (from terminal state)
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(obs.device) < (lengths - 1).unsqueeze(1)
        log_prob = log_prob * mask
        
        return log_prob