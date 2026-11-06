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
from learning.runtime_profiler import RuntimeProfilingMixin

# GFlowNET for sparse reward envs using trajectory balance loss 
class TrajectoryBalance(RuntimeProfilingMixin, GFlowNetBase):
      def _train_impl(self):
            # First perform on-policy training with latest samples, then off-policy training
            # On-policy: use latest samples (most recent trajectories)
            if self.gradient_steps > 0:
                  batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
                      self.memory.latest_sample(self.train_freq, 0)
                  
                  # Efficient version
                  batch_obs_pad = pad_sequence(batch_obs, batch_first = True)
                  batch_acts_pad = pad_sequence(batch_acts, batch_first = True)
                  batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first = True)

                  lengths = torch.tensor(np.array([len(obs) for obs in batch_obs]), device = self.device)

                  # Forward pass
                  forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
                  backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
                  logPF = self.forward_policy.evaluate_actions(batch_obs_pad, batch_acts_pad, forward_mask, lengths = lengths)
                  logPB = self.backward_policy.evaluate_actions(batch_next_obs_pad, batch_acts_pad, backward_mask, lengths = lengths)

                  logPF = torch.sum(logPF, dim = 1)
                  logPB = torch.sum(logPB, dim = 1)

                  batch_rews = torch.tensor(batch_rews, device = self.device)

                  log_reward = torch.log(batch_rews)/self.reward_temperature

                  if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
                        raise ValueError("Infinite logprobs found")
                  
                  logZ = self.get_logZ()
                  loss = (logPF + logZ - logPB - log_reward).pow(2)

                  if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                        raise ValueError(f"Invalid loss found, loss: {loss}")
                  
                  # Log the loss statistics
                  traj_losses = loss.mean().detach().item()
                  traj_losses_std = loss.std().detach().item()
                  traj_losses_max = loss.max().detach().item()

                  self.logger['loss_means'].append(traj_losses)
                  self.logger['loss_max'].append(traj_losses_max)
                  self.logger['loss_std'].append(traj_losses_std)
                  self.logger['log_Z'].append(logZ.item())

                  # Compute traj loss
                  loss = loss.mean()
                  # print(loss)

                  # Optimize the models
                  for optim in self.GFN_optim:
                        optim.zero_grad()

                  loss.backward()

                  # clip gradients
                  torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
                  torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
                  torch.nn.utils.clip_grad_norm_(self.logZ, 1e1)

                  for optim in self.GFN_optim:
                        optim.step()
            
            # Off-policy: use random/prioritized sampling for additional gradient steps
            for gradient_step in range(self.gradient_steps - 1):
                  batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = self.sample_from_memory(self.batch_size)
                  # Efficient version
                  batch_obs_pad = pad_sequence(batch_obs, batch_first = True)
                  batch_acts_pad = pad_sequence(batch_acts, batch_first = True)
                  batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first = True)

                  lengths = torch.tensor(np.array([len(obs) for obs in batch_obs]), device = self.device)

                  # Forward pass
                  forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
                  backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
                  logPF = self.forward_policy.evaluate_actions(batch_obs_pad, batch_acts_pad, forward_mask, lengths = lengths)
                  logPB = self.backward_policy.evaluate_actions(batch_next_obs_pad, batch_acts_pad, backward_mask, lengths = lengths)

                  logPF = torch.sum(logPF, dim = 1)
                  logPB = torch.sum(logPB, dim = 1)

                  batch_rews = torch.tensor(batch_rews, device = self.device)

                  log_reward = torch.log(batch_rews)/self.reward_temperature

                  if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
                        raise ValueError("Infinite logprobs found")
                  
                  logZ = self.get_logZ()
                  loss = (logPF + logZ - logPB - log_reward).pow(2)

                  if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                        raise ValueError(f"Invalid loss found, loss: {loss}")
                  
                  # Log the loss statistics
                  traj_losses = loss.mean().detach().item()
                  traj_losses_std = loss.std().detach().item()
                  traj_losses_max = loss.max().detach().item()

                  self.logger['loss_means'].append(traj_losses)
                  self.logger['loss_max'].append(traj_losses_max)
                  self.logger['loss_std'].append(traj_losses_std)
                  self.logger['log_Z'].append(logZ.item())

                  # Compute traj loss
                  loss = loss.mean()
                  # print(loss)

                  # Optimize the models
                  for optim in self.GFN_optim:
                        optim.zero_grad()

                  loss.backward()

                  # clip gradients
                  torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
                  torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
                  torch.nn.utils.clip_grad_norm_(self.logZ, 1e1)

                  for optim in self.GFN_optim:
                        optim.step()

            self._n_updates += self.gradient_steps
