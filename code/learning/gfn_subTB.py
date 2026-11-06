import os
import sys
import pickle
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from learning.gfn_base import GFlowNetBase
from nns.scalar_nn import logF


class SubTrajectoryBalance(GFlowNetBase):
    """
    GFlowNet using Sub-Trajectory Balance (SubTB) loss.
    
    This method considers all sub-trajectories and weighs them according to 
    a configurable weighting scheme.
    
    Reference: "Learning GFlowNets from partial episodes for improved convergence 
    and stability" (Madan et al., 2022)
    
    Attributes:
        weighting: Sub-trajectory weighting scheme:
            - "DB": One-step transitions only (equivalent to Detailed Balance)
            - "ModifiedDB": One-step transitions weighted by 1/trajectory_length
            - "TB": Full trajectory only (equivalent to Trajectory Balance)
            - "equal_within": Equal weight within trajectory, then equal across batch
            - "equal": Equal weight for all sub-trajectories
            - "geometric_within": Geometric weighting (lambda^len) within trajectory
            - "geometric": Geometric weighting across all sub-trajectories
        lamda: Discount factor for geometric weighting (default: 0.9)
    """
    
    def __init__(self, env, weighting="geometric_within", lamda=0.9, **kwargs):
        # Force use_state_flow = True for SubTB
        kwargs['use_state_flow'] = True
        super().__init__(env, **kwargs)
        
        self.weighting = weighting
        self.lamda = lamda
    
    def train(self):
        """Train using Sub-Trajectory Balance loss."""
        # First perform on-policy training with latest samples, then off-policy training
        # On-policy: use latest samples (most recent trajectories)
        batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
            self.memory.latest_sample(self.train_freq, 0)
        
        # Pad sequences
        batch_obs_pad = pad_sequence(batch_obs, batch_first=True)
        batch_acts_pad = pad_sequence(batch_acts, batch_first=True)
        batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first=True)
        
        lengths = torch.tensor(np.array([len(obs) for obs in batch_obs]), device=self.device)
        n_trajectories = len(batch_idx)
        
        # Get forward and backward log probabilities
        forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
        backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
        log_pf_trajectories = self.forward_policy.evaluate_actions(
            batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
        )
        log_pb_trajectories = self.backward_policy.evaluate_actions(
            batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
        )
        
        batch_rews = torch.tensor(batch_rews, device=self.device)
        log_rewards = torch.log(batch_rews) / self.reward_temperature
        
        # Switch dimensions: [batch, max_len] -> [max_len, batch]
        log_pf_trajectories = log_pf_trajectories.permute(1, 0)
        log_pb_trajectories = log_pb_trajectories.permute(1, 0)
        
        # Cumulative log probabilities
        log_pf_trajectories_cum = self._cumulative_logprobs(n_trajectories, log_pf_trajectories)
        log_pb_trajectories_cum = self._cumulative_logprobs(n_trajectories, log_pb_trajectories)
        
        # Calculate log state flows
        log_state_flows = self._calculate_log_state_flows(batch_obs_pad, lengths, log_pf_trajectories)
        
        # Calculate masks
        full_mask, sink_states_mask, is_terminal_mask = self._calculate_masks(batch_obs_pad, lengths)
        
        flattening_masks = []
        scores = []
        
        max_len = lengths.max().item()
        for i in range(1, 1 + max_len):
            preds = self._calculate_preds(log_pf_trajectories_cum, log_state_flows, i)
            targets = self._calculate_targets(
                log_rewards, preds, log_pb_trajectories_cum,
                log_state_flows, is_terminal_mask, sink_states_mask, full_mask, i
            )
            
            flattening_mask = lengths.lt(torch.arange(i, max_len + 1, device=self.device).unsqueeze(-1))
            flattening_masks.append(flattening_mask)
            scores.append(preds - targets)
        
        flattening_mask = torch.cat(flattening_masks)
        all_scores = torch.cat(scores, 0)
        
        # Compute weighted loss
        if self.weighting == "DB":
            loss = scores[0][~flattening_masks[0]].pow(2).mean()
        elif self.weighting == "geometric":
            per_length_losses = torch.stack([
                s[~fm].pow(2).mean() for s, fm in zip(scores, flattening_masks)
            ])
            L = self.lamda
            ratio = (1 - L) / (1 - L ** max_len)
            weights = ratio * (L ** torch.arange(max_len, device=self.device))
            loss = (per_length_losses * weights).sum()
        else:
            if self.weighting == "equal_within":
                contributions = self._get_equal_within_contributions(lengths)
            elif self.weighting == "equal":
                contributions = self._get_equal_contributions(lengths)
            elif self.weighting == "TB":
                contributions = self._get_tb_contributions(lengths, all_scores)
            elif self.weighting == "ModifiedDB":
                contributions = self._get_modified_db_contributions(lengths)
            elif self.weighting == "geometric_within":
                contributions = self._get_geometric_within_contributions(lengths)
            else:
                raise ValueError(f"Unknown weighting method {self.weighting}")
            
            flat_contributions = contributions[~flattening_mask]
            losses = flat_contributions * all_scores[~flattening_mask].pow(2)
            
            # Aggregate losses
            loss_tensor = torch.zeros_like(flattening_mask, dtype=torch.float)
            loss_tensor[~flattening_mask] = losses
            loss_per_traj = loss_tensor.sum(dim=0)
            loss = loss_per_traj.mean()
        
        if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
            raise ValueError(f"Invalid loss found, loss: {loss}")
        
        # Log statistics
        self.logger['loss_means'].append(loss.detach().item())
        self.logger['loss_max'].append(loss.detach().item())
        self.logger['loss_std'].append(0.0)
        self.logger['log_Z'].append(self.get_logZ().item())
        
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
            n_trajectories = len(batch_idx)
            
            # Get forward and backward log probabilities
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
            log_pf_trajectories = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            log_pb_trajectories = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_rewards = torch.log(batch_rews) / self.reward_temperature
            
            # Switch dimensions: [batch, max_len] -> [max_len, batch]
            log_pf_trajectories = log_pf_trajectories.permute(1, 0)
            log_pb_trajectories = log_pb_trajectories.permute(1, 0)
            
            # Cumulative log probabilities
            log_pf_trajectories_cum = self._cumulative_logprobs(n_trajectories, log_pf_trajectories)
            log_pb_trajectories_cum = self._cumulative_logprobs(n_trajectories, log_pb_trajectories)
            
            # Calculate log state flows
            log_state_flows = self._calculate_log_state_flows(batch_obs_pad, lengths, log_pf_trajectories)
            
            # Calculate masks
            full_mask, sink_states_mask, is_terminal_mask = self._calculate_masks(batch_obs_pad, lengths)
            
            flattening_masks = []
            scores = []
            
            max_len = lengths.max().item()
            for i in range(1, 1 + max_len):
                preds = self._calculate_preds(log_pf_trajectories_cum, log_state_flows, i)
                targets = self._calculate_targets(
                    log_rewards, preds, log_pb_trajectories_cum,
                    log_state_flows, is_terminal_mask, sink_states_mask, full_mask, i
                )
                
                flattening_mask = lengths.lt(torch.arange(i, max_len + 1, device=self.device).unsqueeze(-1))
                flattening_masks.append(flattening_mask)
                scores.append(preds - targets)
            
            flattening_mask = torch.cat(flattening_masks)
            all_scores = torch.cat(scores, 0)
            
            # Compute weighted loss
            if self.weighting == "DB":
                loss = scores[0][~flattening_masks[0]].pow(2).mean()
            elif self.weighting == "geometric":
                per_length_losses = torch.stack([
                    s[~fm].pow(2).mean() for s, fm in zip(scores, flattening_masks)
                ])
                L = self.lamda
                ratio = (1 - L) / (1 - L ** max_len)
                weights = ratio * (L ** torch.arange(max_len, device=self.device))
                loss = (per_length_losses * weights).sum()
            else:
                if self.weighting == "equal_within":
                    contributions = self._get_equal_within_contributions(lengths)
                elif self.weighting == "equal":
                    contributions = self._get_equal_contributions(lengths)
                elif self.weighting == "TB":
                    contributions = self._get_tb_contributions(lengths, all_scores)
                elif self.weighting == "ModifiedDB":
                    contributions = self._get_modified_db_contributions(lengths)
                elif self.weighting == "geometric_within":
                    contributions = self._get_geometric_within_contributions(lengths)
                else:
                    raise ValueError(f"Unknown weighting method {self.weighting}")
                
                flat_contributions = contributions[~flattening_mask]
                losses = flat_contributions * all_scores[~flattening_mask].pow(2)
                
                # Aggregate losses
                loss_tensor = torch.zeros_like(flattening_mask, dtype=torch.float)
                loss_tensor[~flattening_mask] = losses
                loss_per_traj = loss_tensor.sum(dim=0)
                loss = loss_per_traj.mean()
            
            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                raise ValueError(f"Invalid loss found, loss: {loss}")
            
            # Log statistics
            self.logger['loss_means'].append(loss.detach().item())
            self.logger['loss_max'].append(loss.detach().item())
            self.logger['loss_std'].append(0.0)
            self.logger['log_Z'].append(self.get_logZ().item())
            
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
    
    def _cumulative_logprobs(self, n_trajectories, log_p_trajectories):
        """Calculate cumulative log probabilities deterministically.

        CUDA ``cumsum`` has no deterministic implementation in some PyTorch
        versions.  Build the prefix sums in a fixed order from deterministic
        elementwise additions instead, while preserving the autograd graph.
        """
        cumulative_logprobs = [
            log_p_trajectories.new_zeros(n_trajectories)
        ]
        for step_logprobs in log_p_trajectories.unbind(dim=0):
            cumulative_logprobs.append(
                cumulative_logprobs[-1] + step_logprobs
            )
        return torch.stack(cumulative_logprobs, dim=0)
    
    def _calculate_preds(self, log_pf_trajectories_cum, log_state_flows, i):
        """Calculate predictions for sub-trajectory length i."""
        current_log_state_flows = log_state_flows if i == 1 else log_state_flows[:-(i-1)]
        logPF = log_pf_trajectories_cum[i:] - log_pf_trajectories_cum[:-i]
        return logPF + current_log_state_flows
    
    def _calculate_targets(self, log_rewards, preds, log_pb_trajectories_cum,
                           log_state_flows, is_terminal_mask, sink_states_mask, full_mask, i):
        """Calculate targets for sub-trajectory length i."""
        targets = torch.full_like(preds, fill_value=-float("inf"))
        terminal_mask = is_terminal_mask[i-1:]
        targets.T[terminal_mask.T] = log_rewards[terminal_mask.sum(dim=0) > 0]
        
        if i > 1:
            targets[terminal_mask] += (
                log_pb_trajectories_cum[i-1:] - log_pb_trajectories_cum[:-i+1]
            )[:-1][terminal_mask]
        
        targets[~full_mask[i-1:]] = (
            log_pb_trajectories_cum[i:] - log_pb_trajectories_cum[:-i]
        )[:-1][~full_mask[i-1:-1]] + log_state_flows[i:][~sink_states_mask[i:]]
        
        return targets
    
    def _calculate_log_state_flows(self, batch_obs_pad, lengths, log_pf_trajectories):
        """Calculate log state flows."""
        states = batch_obs_pad.clone()
        log_state_flows = torch.full_like(log_pf_trajectories, fill_value=-float("inf"))
        mask = torch.arange(states.size(1), device=self.device) < lengths.unsqueeze(-1)
        mask = mask.permute(1, 0)
        states = states.permute(1, 0, 2)
        valid_states = states[mask]
        log_F = self.logF(valid_states).squeeze(-1)
        log_state_flows[mask] = log_F
        return log_state_flows
    
    def _calculate_masks(self, batch_obs_pad, lengths):
        """Calculate masks for sink and terminal states."""
        states = batch_obs_pad
        sink_states_mask = torch.arange(states.size(1), device=self.device) >= lengths.unsqueeze(-1)
        is_terminal_mask = torch.arange(states.size(1), device=self.device) == lengths.unsqueeze(-1) - 1
        sink_states_mask = sink_states_mask.permute(1, 0)
        is_terminal_mask = is_terminal_mask.permute(1, 0)
        full_mask = sink_states_mask | is_terminal_mask
        return full_mask, sink_states_mask, is_terminal_mask
    
    def _get_equal_within_contributions(self, lengths):
        """Equal weighting within each trajectory."""
        max_len = lengths.max().item()
        n_rows = int(max_len * (1 + max_len) / 2)
        contributions = 2.0 / (lengths * (lengths + 1)) / len(lengths)
        return contributions.repeat(n_rows, 1)
    
    def _get_equal_contributions(self, lengths):
        """Equal weighting for all sub-trajectories."""
        max_len = lengths.max().item()
        n_rows = int(max_len * (1 + max_len) / 2)
        n_sub_trajectories = int((lengths * (lengths + 1) / 2).sum().item())
        return torch.ones(n_rows, len(lengths), device=self.device) / n_sub_trajectories
    
    def _get_tb_contributions(self, lengths, all_scores):
        """Full trajectory weighting only."""
        max_len = lengths.max().item()
        contributions = torch.zeros_like(all_scores, device=self.device)
        indices = (max_len * (lengths - 1) - (lengths - 1) * (lengths - 2) / 2).long()
        contributions.scatter_(0, indices.unsqueeze(0), 1)
        return contributions / len(lengths)
    
    def _get_modified_db_contributions(self, lengths):
        """One-step transitions weighted by 1/trajectory_length."""
        max_len = lengths.max().item()
        n_rows = int(max_len * (1 + max_len) / 2)
        contributions = (1.0 / lengths / len(lengths)).repeat(max_len, 1)
        return torch.cat((
            contributions,
            torch.zeros((n_rows - max_len, len(lengths)), device=self.device)
        ), 0)
    
    def _get_geometric_within_contributions(self, lengths):
        """Geometric weighting within each trajectory."""
        L = self.lamda
        max_len = lengths.max().item()
        
        contributions = (L ** torch.arange(max_len, device=self.device).double()).float()
        contributions = contributions.unsqueeze(-1).repeat(1, len(lengths))
        contributions = contributions.repeat_interleave(
            torch.arange(max_len, 0, -1, device=self.device),
            dim=0,
            output_size=int(max_len * (max_len + 1) / 2)
        )
        
        per_trajectory_denom = (
            1.0 / (1 - L) ** 2 * (L * (L ** lengths.double() - 1) + (1 - L) * lengths.double())
        ).float()
        
        return contributions / per_trajectory_denom / len(lengths)
