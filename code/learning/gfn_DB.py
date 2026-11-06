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


class DetailedBalance(GFlowNetBase):
    """
    GFlowNet using Detailed Balance (DB) loss.
    
    The DB loss is defined as:
        L_DB = sum_t (log F(s_t) + log P_F(s_{t+1}|s_t) - log F(s_{t+1}) - log P_B(s_t|s_{t+1}))^2
    
    where F(s) is the flow at state s. For terminal states, F(s_f) = R(s_f).
    
    This requires learning a state flow function F(s) in addition to the policies.
    """
    
    def __init__(self, env, **kwargs):
        # Force use_state_flow = True for DB loss
        kwargs['use_state_flow'] = True
        super().__init__(env, **kwargs)
    
    def train(self):
        """Train using Detailed Balance loss."""
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
            # batch_obs_pad: [batch, max_len, state_dim]
            # We need logF for s_t and s_{t+1}
            batch_size = batch_obs_pad.shape[0]
            
            # Reshape for logF computation
            obs_flat = batch_obs_pad.reshape(-1, self.state_dim)
            next_obs_flat = batch_next_obs_pad.reshape(-1, self.state_dim)
            
            logF_s = self.logF(obs_flat).reshape(batch_size, max_len)
            logF_s_next = self.logF(next_obs_flat).reshape(batch_size, max_len)
            
            # Create mask for valid transitions
            # transition t is valid if t < length
            valid_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # For the last transition (terminal), use log reward instead of logF(s_next)
            terminal_mask = torch.arange(max_len, device=self.device).unsqueeze(0) == (lengths - 1).unsqueeze(1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            # Replace logF(s_{t+1}) with log R for terminal transitions
            # log_reward has shape [batch], we need to broadcast it
            logF_s_next_or_R = logF_s_next.clone()
            logF_s_next_or_R[terminal_mask] = log_reward
            
            # DB loss: (logF(s) + logPF - logF(s') - logPB)^2 for each transition
            db_scores = logF_s + logPF - logF_s_next_or_R - logPB
            
            # Apply mask and compute loss
            db_scores_masked = db_scores * valid_mask
            loss_per_traj = (db_scores_masked ** 2).sum(dim=1) / lengths.float()
            
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
            # batch_obs_pad: [batch, max_len, state_dim]
            # We need logF for s_t and s_{t+1}
            batch_size = batch_obs_pad.shape[0]
            
            # Reshape for logF computation
            obs_flat = batch_obs_pad.reshape(-1, self.state_dim)
            next_obs_flat = batch_next_obs_pad.reshape(-1, self.state_dim)
            
            logF_s = self.logF(obs_flat).reshape(batch_size, max_len)
            logF_s_next = self.logF(next_obs_flat).reshape(batch_size, max_len)
            
            # Create mask for valid transitions
            # transition t is valid if t < length
            valid_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            # For the last transition (terminal), use log reward instead of logF(s_next)
            terminal_mask = torch.arange(max_len, device=self.device).unsqueeze(0) == (lengths - 1).unsqueeze(1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            # Replace logF(s_{t+1}) with log R for terminal transitions
            # log_reward has shape [batch], we need to broadcast it
            logF_s_next_or_R = logF_s_next.clone()
            logF_s_next_or_R[terminal_mask] = log_reward
            
            # DB loss: (logF(s) + logPF - logF(s') - logPB)^2 for each transition
            db_scores = logF_s + logPF - logF_s_next_or_R - logPB
            
            # Apply mask and compute loss
            db_scores_masked = db_scores * valid_mask
            loss_per_traj = (db_scores_masked ** 2).sum(dim=1) / lengths.float()
            
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

