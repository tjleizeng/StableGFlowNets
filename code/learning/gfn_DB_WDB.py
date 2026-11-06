import os
import sys
import time
import math
import pickle
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from learning.gfn_base import GFlowNetBase


class DetailedBalanceWeightedByReachableEndStates(GFlowNetBase):
    """
    GFlowNet using Weighted Detailed Balance (WDB) loss.
    
    Each trajectory uses a weighted mean of squared DB residuals, with
    inverse reachable-terminal counts as edge weights. Weights are normalized
    over valid transitions within that trajectory, excluding padding.

    Reference: Silva et al., "When do GFlowNets learn the right distribution?"
    (ICLR 2025), Section 3.2, Eqs. (5)-(6).
    https://proceedings.iclr.cc/paper_files/paper/2025/file/a48f8928f78a58399ef0049453c14b02-Paper-Conference.pdf

    For each transition s -> s', the weight is 1/|X_{s'}|, where |X_{s'}|
    is the environment's count of terminal states reachable from the NEXT
    state s'. Existing environment count approximations are retained.

    Note: This requires the environment to implement `get_num_of_reachable_end_states()`
    which returns the number of terminal states reachable from a given state.
    """
    
    def __init__(self, env, **kwargs):
        # Check if environment supports WDB
        if not hasattr(env.unwrapped, 'get_num_of_reachable_end_states'):
            raise NotImplementedError(
                "WDB requires the environment to implement 'get_num_of_reachable_end_states()'. "
                "This method should return the number of reachable terminal states from a given state. "
                "WDB is not feasible for this environment."
            )
        
        # Force use_state_flow = True for WDB loss
        kwargs['use_state_flow'] = True
        super().__init__(env, **kwargs)
    
    def train(self):
        """Train using Weighted Detailed Balance loss."""
        # First perform on-policy training with latest samples, then off-policy training
        # On-policy: use latest samples (most recent trajectories)
        if self.gradient_steps > 0:
            batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
                self.memory.latest_sample(self.train_freq, 0)
            
            # Pad sequences
            batch_obs_pad = pad_sequence(batch_obs, batch_first=True)
            batch_acts_pad = pad_sequence(batch_acts, batch_first=True)
            batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first=True)
            
            lengths = torch.tensor(np.array([len(obs) for obs in batch_obs]), device=self.device)
            max_len = lengths.max().item()
            batch_size = batch_obs_pad.shape[0]
            
            # Get forward and backward log probabilities
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
            logPF = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            logPB = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            # Compute log flows for all states
            obs_flat = batch_obs_pad.reshape(-1, self.state_dim)
            next_obs_flat = batch_next_obs_pad.reshape(-1, self.state_dim)
            
            logF_s = self.logF(obs_flat).reshape(batch_size, max_len)
            logF_s_next = self.logF(next_obs_flat).reshape(batch_size, max_len)
            
            # Create mask for valid transitions
            valid_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # For the last transition (terminal), use log reward instead of logF(s_next)
            terminal_mask = torch.arange(max_len, device=self.device).unsqueeze(0) == (lengths - 1).unsqueeze(1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            # Replace logF(s_{t+1}) with log R for terminal transitions
            logF_s_next_or_R = logF_s_next.clone()
            logF_s_next_or_R[terminal_mask] = log_reward
            
            # Compute WDB weights: 1/|X_{s'}| for each transition s -> s'
            # The weight is based on the NEXT state s', not the current state s
            with torch.no_grad():
                # batch_next_obs_pad is [batch, max_len, state_dim] - use s' not s
                next_obs_np = batch_next_obs_pad.cpu().numpy()
                weights = np.ones((batch_size, max_len), dtype=np.float32)
                
                for b in range(batch_size):
                    for t in range(int(lengths[b].item())):
                        next_state_obs = next_obs_np[b, t]
                        num_reachable = self.env.unwrapped.get_num_of_reachable_end_states(next_state_obs)
                        if num_reachable > 0:
                            weights[b, t] = 1.0 / num_reachable
                        else:
                            weights[b, t] = 1.0  # fallback
                
                weights = torch.tensor(weights, device=self.device)
            
            # DB loss: (logF(s) + logPF - logF(s') - logPB)^2 for each transition
            db_scores = logF_s + logPF - logF_s_next_or_R - logPB
            
            # Silva et al. (ICLR 2025), Eqs. (5)-(6): weight the squared
            # residual once and normalize over valid edges of each trajectory.
            valid_weights = weights * valid_mask
            normalized_weights = valid_weights / valid_weights.sum(dim=1, keepdim=True)
            loss_per_traj = (normalized_weights * db_scores.square()).sum(dim=1)
            
            if torch.any(torch.isinf(loss_per_traj)) or torch.any(torch.isnan(loss_per_traj)):
                raise ValueError(f"Invalid loss found, loss: {loss_per_traj}")
            
            # Log statistics
            traj_losses = loss_per_traj.mean().detach().item()
            traj_losses_std = loss_per_traj.std().detach().item()
            traj_losses_max = loss_per_traj.max().detach().item()
            
            self.logger['loss_means'].append(traj_losses)
            self.logger['loss_max'].append(traj_losses_max)
            self.logger['loss_std'].append(traj_losses_std)
            self.logger['log_Z'].append(self.get_logZ().item())
            
            # Compute mean loss
            loss = loss_per_traj.mean()
            
            # Optimize
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.logF.parameters(), 1e1)
            
            for optim in self.GFN_optim:
                optim.step()
        
        # Off-policy: use random/prioritized sampling for additional gradient steps
        for gradient_step in range(self.gradient_steps - 1):
            # Sample from memory
            batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
                self.sample_from_memory(self.batch_size)
            
            # Pad sequences
            batch_obs_pad = pad_sequence(batch_obs, batch_first=True)
            batch_acts_pad = pad_sequence(batch_acts, batch_first=True)
            batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first=True)
            
            lengths = torch.tensor(np.array([len(obs) for obs in batch_obs]), device=self.device)
            max_len = lengths.max().item()
            batch_size = batch_obs_pad.shape[0]
            
            # Get forward and backward log probabilities
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
            logPF = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            logPB = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            # Compute log flows for all states
            obs_flat = batch_obs_pad.reshape(-1, self.state_dim)
            next_obs_flat = batch_next_obs_pad.reshape(-1, self.state_dim)
            
            logF_s = self.logF(obs_flat).reshape(batch_size, max_len)
            logF_s_next = self.logF(next_obs_flat).reshape(batch_size, max_len)
            
            # Create mask for valid transitions
            valid_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # For the last transition (terminal), use log reward instead of logF(s_next)
            terminal_mask = torch.arange(max_len, device=self.device).unsqueeze(0) == (lengths - 1).unsqueeze(1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            # Replace logF(s_{t+1}) with log R for terminal transitions
            logF_s_next_or_R = logF_s_next.clone()
            logF_s_next_or_R[terminal_mask] = log_reward
            
            # Compute WDB weights: 1/|X_{s'}| for each transition s -> s'
            # The weight is based on the NEXT state s', not the current state s
            with torch.no_grad():
                # batch_next_obs_pad is [batch, max_len, state_dim] - use s' not s
                next_obs_np = batch_next_obs_pad.cpu().numpy()
                weights = np.ones((batch_size, max_len), dtype=np.float32)
                
                for b in range(batch_size):
                    for t in range(int(lengths[b].item())):
                        next_state_obs = next_obs_np[b, t]
                        num_reachable = self.env.unwrapped.get_num_of_reachable_end_states(next_state_obs)
                        if num_reachable > 0:
                            weights[b, t] = 1.0 / num_reachable
                        else:
                            weights[b, t] = 1.0  # fallback
                
                weights = torch.tensor(weights, device=self.device)
            
            # DB loss: (logF(s) + logPF - logF(s') - logPB)^2 for each transition
            db_scores = logF_s + logPF - logF_s_next_or_R - logPB
            
            # Silva et al. (ICLR 2025), Eqs. (5)-(6): weight the squared
            # residual once and normalize over valid edges of each trajectory.
            valid_weights = weights * valid_mask
            normalized_weights = valid_weights / valid_weights.sum(dim=1, keepdim=True)
            loss_per_traj = (normalized_weights * db_scores.square()).sum(dim=1)
            
            if torch.any(torch.isinf(loss_per_traj)) or torch.any(torch.isnan(loss_per_traj)):
                raise ValueError(f"Invalid loss found, loss: {loss_per_traj}")
            
            # Log statistics
            traj_losses = loss_per_traj.mean().detach().item()
            traj_losses_std = loss_per_traj.std().detach().item()
            traj_losses_max = loss_per_traj.max().detach().item()
            
            self.logger['loss_means'].append(traj_losses)
            self.logger['loss_max'].append(traj_losses_max)
            self.logger['loss_std'].append(traj_losses_std)
            self.logger['log_Z'].append(self.get_logZ().item())
            
            # Compute mean loss
            loss = loss_per_traj.mean()
            
            # Optimize
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.logF.parameters(), 1e1)
            
            for optim in self.GFN_optim:
                optim.step()
        
        self._n_updates += self.gradient_steps

