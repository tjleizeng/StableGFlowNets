import os
import sys
import time
import math
import pickle
import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import override

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from learning.gfn_base import GFlowNetBase
from nns.discrete_nn import DiscreteForwardPolicy, DiscreteBackwardPolicy


class TrajectoryBalanceAdaptiveTeacher(GFlowNetBase):
    """
    GFlowNet with Adaptive Teacher for exploration.
    
    This method uses a separate "teacher" policy that is trained with a modified 
    reward that encourages exploring states where the student has high loss.
    The teacher provides exploration trajectories to help the student learn.
    
    Attributes:
        alpha: Weight for mixing original reward with loss-based reward (default: 0.0)
    """
    
    def __init__(self, env, alpha=0.0, **kwargs):
        super().__init__(env, **kwargs)
        
        self.alpha = alpha
        
        # Create teacher models (same architecture as student)
        hidden_sizes = kwargs.get('hidden_sizes', [256, 256])
        activation_fn = kwargs.get('activation_fn', torch.nn.ReLU)
        initial_z = kwargs.get('initial_z', 0.0)
        
        self.forward_teacher_policy = DiscreteForwardPolicy(
            self.state_dim, self.action_dim, hidden_sizes, activation_fn,
            device=self.device
        ).to(self.device)
        self.backward_teacher_policy = DiscreteBackwardPolicy(
            self.state_dim, self.action_dim, hidden_sizes, activation_fn,
            device=self.device
        ).to(self.device)
        
        # Teacher logZ
        self.teacher_logZ = torch.tensor(
            initial_z, requires_grad=True, device=self.device, dtype=torch.float32
        )
        
        # Teacher optimizers
        self.teacher_forward_optim = Adam(self.forward_teacher_policy.parameters(), lr=self.learning_rate)
        self.teacher_backward_optim = Adam(self.backward_teacher_policy.parameters(), lr=self.learning_rate)
        self.teacher_logZ_optim = Adam([self.teacher_logZ], lr=self.learning_rate * 100)
        
        # Add teacher optimizers to list
        self.GFN_optim.extend([
            self.teacher_forward_optim, 
            self.teacher_backward_optim, 
            self.teacher_logZ_optim
        ])
    
    def teacher_predict(self, obs, action_mask=None):
        """Query the teacher forward policy for an action."""
        action = self.forward_teacher_policy(obs, action_mask, temperature=self.policy_temperature)
        return action.detach().cpu().numpy()
    
    @override
    def collect_rollouts(self):
        """
        Collect rollouts using student policy (50%) and teacher policy (50%).
        
        Uses parallel collection if data_env is available, otherwise sequential.
        """
        # Half for student, half for teacher
        half_freq = self.train_freq // 2
        teacher_count = self.train_freq - half_freq  # Handle odd train_freq
        
        buffer_lens = []
        buffer_rews_report = []
        num_collected_steps, num_collected_episodes = 0, 0
        
        with torch.no_grad():
            if self.data_env is not None:
                # Parallel collection: first student, then teacher
                # Collect using student policy
                student_steps, student_eps, student_lens, student_rews = self.collect_rollouts_parallel(
                    predict_fn=self.predict, num_trajectories=half_freq
                )
                num_collected_steps += student_steps
                num_collected_episodes += student_eps
                buffer_lens.extend(student_lens)
                buffer_rews_report.extend(student_rews)
                
                # Collect using teacher policy
                teacher_steps, teacher_eps, teacher_lens, teacher_rews = self.collect_rollouts_parallel(
                    predict_fn=self.teacher_predict, num_trajectories=teacher_count
                )
                num_collected_steps += teacher_steps
                num_collected_episodes += teacher_eps
                buffer_lens.extend(teacher_lens)
                buffer_rews_report.extend(teacher_rews)
            else:
                # Sequential collection: first student, then teacher
                # Collect using student policy
                student_steps, student_eps, student_lens, student_rews = self.collect_rollouts_sequential(
                    predict_fn=self.predict, num_trajectories=half_freq
                )
                num_collected_steps += student_steps
                num_collected_episodes += student_eps
                buffer_lens.extend(student_lens)
                buffer_rews_report.extend(student_rews)
                
                # Collect using teacher policy
                teacher_steps, teacher_eps, teacher_lens, teacher_rews = self.collect_rollouts_sequential(
                    predict_fn=self.teacher_predict, num_trajectories=teacher_count
                )
                num_collected_steps += teacher_steps
                num_collected_episodes += teacher_eps
                buffer_lens.extend(teacher_lens)
                buffer_rews_report.extend(teacher_rews)
        
        self.logger['buffer_lens'] = buffer_lens
        self.logger['buffer_rews'] = buffer_rews_report
        
        return num_collected_steps, num_collected_episodes
    
    def train(self):
        """Train both student and teacher using TB loss."""
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
            
            # Get forward and backward log probabilities for both student and teacher
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
            logPF = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            logPB = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            teacher_logPF = self.forward_teacher_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            teacher_logPB = self.backward_teacher_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            # Sum log probabilities
            logPF = torch.sum(logPF, dim=1)
            logPB = torch.sum(logPB, dim=1)
            teacher_logPF = torch.sum(teacher_logPF, dim=1)
            teacher_logPB = torch.sum(teacher_logPB, dim=1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
                raise ValueError("Infinite logprobs found")
            
            logZ = self.get_logZ()
            
            # Student loss
            loss = (logPF + logZ - logPB - log_reward).pow(2)
            
            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                raise ValueError(f"Invalid loss found, loss: {loss}")
            
            # Teacher loss: reward is based on student's loss (explore where student struggles)
            # teacher_log_reward = alpha * log_reward + (1 - alpha) * student_loss - log(student_loss)
            teacher_log_reward = self.alpha * log_reward + (1 - self.alpha) * loss.detach() - torch.log(loss.detach() + 1e-8)
            teacher_loss = (teacher_logPF + self.teacher_logZ - teacher_logPB - teacher_log_reward).pow(2)
            
            # Log statistics
            traj_losses = loss.mean().detach().item()
            traj_losses_std = loss.std().detach().item()
            traj_losses_max = loss.max().detach().item()
            
            self.logger['loss_means'].append(traj_losses)
            self.logger['loss_max'].append(traj_losses_max)
            self.logger['loss_std'].append(traj_losses_std)
            self.logger['log_Z'].append(logZ.item())
            
            # Compute mean losses
            loss = loss.mean()
            teacher_loss = teacher_loss.mean()
            
            # Optimize all
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()
            teacher_loss.backward()
            
            # Clip gradients (especially for teacher to prevent instability)
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.forward_teacher_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_teacher_policy.parameters(), 1e1)
            if self.logZ is not None:
                torch.nn.utils.clip_grad_norm_([self.logZ], 1e1)
            torch.nn.utils.clip_grad_norm_([self.teacher_logZ], 1e1)
            
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
            
            # Get forward and backward log probabilities for both student and teacher
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
            logPF = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            logPB = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            teacher_logPF = self.forward_teacher_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            teacher_logPB = self.backward_teacher_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            # Sum log probabilities
            logPF = torch.sum(logPF, dim=1)
            logPB = torch.sum(logPB, dim=1)
            teacher_logPF = torch.sum(teacher_logPF, dim=1)
            teacher_logPB = torch.sum(teacher_logPB, dim=1)
            
            batch_rews = torch.tensor(batch_rews, device=self.device)
            log_reward = torch.log(batch_rews) / self.reward_temperature
            
            if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
                raise ValueError("Infinite logprobs found")
            
            logZ = self.get_logZ()
            
            # Student loss
            loss = (logPF + logZ - logPB - log_reward).pow(2)
            
            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                raise ValueError(f"Invalid loss found, loss: {loss}")
            
            # Teacher loss: reward is based on student's loss (explore where student struggles)
            # teacher_log_reward = alpha * log_reward + (1 - alpha) * student_loss - log(student_loss)
            teacher_log_reward = self.alpha * log_reward + (1 - self.alpha) * loss.detach() - torch.log(loss.detach() + 1e-8)
            teacher_loss = (teacher_logPF + self.teacher_logZ - teacher_logPB - teacher_log_reward).pow(2)
            
            # Log statistics
            traj_losses = loss.mean().detach().item()
            traj_losses_std = loss.std().detach().item()
            traj_losses_max = loss.max().detach().item()
            
            self.logger['loss_means'].append(traj_losses)
            self.logger['loss_max'].append(traj_losses_max)
            self.logger['loss_std'].append(traj_losses_std)
            self.logger['log_Z'].append(logZ.item())
            
            # Compute mean losses
            loss = loss.mean()
            teacher_loss = teacher_loss.mean()
            
            # Optimize all
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()
            teacher_loss.backward()
            
            # Clip gradients (especially for teacher to prevent instability)
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.forward_teacher_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.backward_teacher_policy.parameters(), 1e1)
            if self.logZ is not None:
                torch.nn.utils.clip_grad_norm_([self.logZ], 1e1)
            torch.nn.utils.clip_grad_norm_([self.teacher_logZ], 1e1)
            
            for optim in self.GFN_optim:
                optim.step()
        
        self._n_updates += self.gradient_steps
    
    def save(self, filepath):
        """Save model including teacher components."""
        super().save(filepath)
        
        # Save teacher components
        torch.save(self.forward_teacher_policy.state_dict(), os.path.join(filepath, 'teacher_forward_policy.pth'))
        torch.save(self.backward_teacher_policy.state_dict(), os.path.join(filepath, 'teacher_backward_policy.pth'))
        torch.save(self.teacher_logZ, os.path.join(filepath, 'teacher_logZ.pth'))
        torch.save(self.teacher_forward_optim.state_dict(), os.path.join(filepath, 'teacher_forward_optim.pth'))
        torch.save(self.teacher_backward_optim.state_dict(), os.path.join(filepath, 'teacher_backward_optim.pth'))
        torch.save(self.teacher_logZ_optim.state_dict(), os.path.join(filepath, 'teacher_logZ_optim.pth'))
    
    def load(self, filepath, load_optim=False):
        """Load model including teacher components."""
        super().load(filepath, load_optim)
        
        # Load teacher components
        self.forward_teacher_policy.load_state_dict(torch.load(os.path.join(filepath, 'teacher_forward_policy.pth')))
        self.backward_teacher_policy.load_state_dict(torch.load(os.path.join(filepath, 'teacher_backward_policy.pth')))
        self.teacher_logZ = torch.load(os.path.join(filepath, 'teacher_logZ.pth'))
        
        if load_optim:
            self.teacher_forward_optim.load_state_dict(torch.load(os.path.join(filepath, 'teacher_forward_optim.pth')))
            self.teacher_backward_optim.load_state_dict(torch.load(os.path.join(filepath, 'teacher_backward_optim.pth')))
            self.teacher_logZ_optim.load_state_dict(torch.load(os.path.join(filepath, 'teacher_logZ_optim.pth')))
