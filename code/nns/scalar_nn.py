import torch
import torch.nn as nn


class LogZEstimator(nn.Module):
    """
    Neural network estimator for log partition function (logZ) in GFlowNets.
    
    This can be used as:
    1. A learnable scalar parameter (state_dim=0 or use_network=False)
    2. A state-conditioned estimator (use_network=True with state_dim > 0)
    
    Args:
        state_dim: Dimension of input state (0 for scalar logZ)
        hidden_sizes: List of hidden layer sizes
        activation_fn: Activation function class
        device: Device to place the model on
        initial_value: Initial value for logZ (used for scalar mode)
        use_network: If False, uses a learnable scalar parameter instead of network
    """
    def __init__(self, state_dim=0, hidden_sizes=[128, 128],
                 activation_fn=torch.nn.ReLU,
                 device="cpu",
                 initial_value=0.0,
                 use_network=False):
        super(LogZEstimator, self).__init__()
        self.device = device
        self.use_network = use_network and state_dim > 0
        
        if self.use_network:
            # State-conditioned logZ estimator
            layers = []
            # Input layer
            layers.append(nn.Linear(state_dim, hidden_sizes[0]))
            layers.append(activation_fn())
            
            # Hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(activation_fn())
            
            # Output layer (single scalar output)
            layers.append(nn.Linear(hidden_sizes[-1], 1))
            
            self.network = nn.Sequential(*layers)
        else:
            # Simple learnable scalar parameter
            self.logZ = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))
            self.network = None
    
    def forward(self, obs=None):
        """
        Compute logZ estimate.
        
        Args:
            obs: Optional state observation (only used if use_network=True)
            
        Returns:
            logZ estimate as a scalar or tensor
        """
        if self.use_network:
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            return self.network(obs).squeeze(-1)
        else:
            return self.logZ
    
    def get_value(self):
        """Get the current logZ value (for scalar mode compatibility)."""
        if self.use_network:
            raise ValueError("get_value() only works in scalar mode (use_network=False)")
        return self.logZ.item()


class logF(nn.Module):  # logF for scaler estimator in SubTB GFlowNets, interestingly it is the same as critic.
    def __init__(self, state_dim, hidden_sizes = [128, 128],\
                 activation_fn = torch.nn.ReLU,\
                 device = "cpu"):
        super(logF, self).__init__()
        self.device = device

        # Create the state flow network
        layers = []
        # Input layer
        layers.append(nn.Linear(state_dim, hidden_sizes[0]))
        layers.append(activation_fn())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation_fn())

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.critic = nn.Sequential(*layers)

    def forward(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device = self.device)
        return self.critic(obs)


# Used for GAFN
class RND(nn.Module):
    def __init__(self, state_dim, reward_scale=0.5, hidden_dim=256, s_latent_dim=128, device="cpu"):
        super(RND, self).__init__()
        self.device = device

        self.random_target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim)
        )

        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, s_latent_dim),
        )
        
        self.reward_scale = reward_scale

    def forward(self, next_state):
        random_phi_s_next = self.random_target_network(next_state)
        predicted_phi_s_next = self.predictor_network(next_state)
        return random_phi_s_next, predicted_phi_s_next

    def compute_intrinsic_reward(self, next_states, lengths):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)

        intrinsic_reward = torch.norm(predicted_phi_s_next.detach() - random_phi_s_next.detach(), dim=-1, p=2)
        intrinsic_reward *= self.reward_scale

        intrinsic_reward = intrinsic_reward.detach()

        batch_size, seq_len = next_states.size(0), next_states.size(1)
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(next_states.device) < lengths.unsqueeze(1)

        intrinsic_reward = intrinsic_reward * mask

        return intrinsic_reward

    def compute_loss(self, next_states, lengths):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states)
        rnd_loss = torch.norm(predicted_phi_s_next - random_phi_s_next.detach(), dim=-1, p=2)

        batch_size, seq_len = next_states.size(0), next_states.size(1)
        mask = torch.arange(seq_len).expand(batch_size, seq_len).to(next_states.device) < lengths.unsqueeze(1)

        rnd_loss = rnd_loss * mask

        mean_rnd_loss = torch.mean(rnd_loss)

        # assert mean_rnd_loss is not Nan
        assert not torch.isnan(mean_rnd_loss).any()
        
        return mean_rnd_loss