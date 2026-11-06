import os
import sys
import time
import math
import pickle
import json
import uuid
import numpy as np
import torch
from torch.optim import Adam

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nns.discrete_nn import DiscreteForwardPolicy, DiscreteBackwardPolicy, UniformBackwardPolicy
from nns.scalar_nn import logF
from utils.memory import Memory
from utils.mode_evaluation import record_discovered_mode

from abc import abstractmethod
import wandb
from typing import List, Tuple, Optional, Union
from tqdm import tqdm


class GFlowNetBase:
      def __init__(self, env, backward_policy = None, \
                   learning_rate = 1e-3, batch_size = 32, memory_size = 10000,\
                   train_freq=16, gradient_steps = 1, \
                   policy_temperature = 1, \
                   reward_temperature = 1, \
                   sample_method = 0, \
                   device = 'auto', wandb_project = None, wandb_name = None, verbose = False,\
                   hidden_sizes = [256, 256], \
                   activation_fn = torch.nn.ReLU,\
                   initial_z = 0.0,\
                   use_state_flow = False,
                   model_dir=None, \
                   data_env=None, \
                   temperature_decay_rate = 0.99,
                   trainable_logPB = True,
                   max_state_size = 10000,
                   epsilon = 0.0,
                   runtime_profile = True,
                   cuda_profile = True,
                   checkpoint_interval = 500,
                   progress = True,
                   metrics_jsonl = False
                   ):
          
            # initialize the hyperparameters
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.memory_size = memory_size
            self.train_freq = train_freq
            self.policy_temperature = policy_temperature
            self.policy_temperature_init = policy_temperature
            self.reward_temperature = reward_temperature
            self.reward_temperature_init = reward_temperature
            self.gradient_steps = gradient_steps
            self.sample_method = sample_method
            self.device = device
            self.verbose = verbose
            self.model_dir = model_dir
            self.data_env = data_env
            self.temperature_decay_rate = temperature_decay_rate
            self.converged = False
            self.epsilon = epsilon
            # Execution/output controls do not alter the scientific update.
            self.runtime_profile = runtime_profile
            self.cuda_profile = cuda_profile
            if checkpoint_interval < 0:
                  raise ValueError('checkpoint_interval must be nonnegative')
            self.checkpoint_interval = checkpoint_interval
            self.progress = progress
            self.metrics_jsonl = metrics_jsonl
            if metrics_jsonl and model_dir is None:
                  raise ValueError('metrics_jsonl requires model_dir')
            self._metrics_run_id = None

            if self.device == 'auto':
                  self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.state_dim = env.observation_space.shape[0]
            self.use_state_flow = use_state_flow

            self.action_dim = env.action_space.n
            self.forward_policy = DiscreteForwardPolicy(self.state_dim, self.action_dim, hidden_sizes, activation_fn, device = self.device).to(self.device)
            # Use provided backward_policy, uniform policy, or initialize learnable one
            if backward_policy is not None:
                  self.backward_policy = backward_policy
            elif not trainable_logPB:
                  # Use uniform backward policy (non-trainable)
                  self.backward_policy = UniformBackwardPolicy(self.action_dim, device = self.device).to(self.device)
            else:
                  self.backward_policy = DiscreteBackwardPolicy(self.state_dim, self.action_dim, hidden_sizes, activation_fn, device = self.device).to(self.device)
            
            # Initialize state flow network if requested
            if use_state_flow:
                  self.logF = logF(self.state_dim, hidden_sizes, activation_fn, device = self.device).to(self.device)
                  self.logF_optim = Adam(self.logF.parameters(), lr=self.learning_rate)
                  # Get initial state observation for computing logZ = logF(initial_state)
                  initial_obs, _ = env.reset()
                  self.initial_state_obs = torch.tensor(initial_obs, dtype=torch.float32, device=self.device)
                  self.logZ = None
                  self.logZ_optim = None
            else:
                  self.logF = None
                  self.logF_optim = None
                  self.initial_state_obs = None
                  self.logZ = torch.tensor(initial_z, requires_grad = True, device = self.device, dtype=torch.float32)
                  self.logZ_optim = Adam([self.logZ], lr=min(self.learning_rate * 100, 1e-2)) # According to the original paper, the learning rate of logZ is greater than the forward and backward policy, we add a cap here to avoid instability, 1e-2 is draw from the literature

            # On-policy mode: memory_size == batch_size (no random sampling needed)
            is_onpolicy = (self.memory_size == self.batch_size)
            self.memory = Memory(self.memory_size, device = self.device, is_onpolicy = is_onpolicy)

            self.forward_optim = Adam(self.forward_policy.parameters(), lr=self.learning_rate)
            # Only create backward optimizer if backward_policy is learnable (has parameters)
            if not trainable_logPB:
                  # Uniform backward policy - no parameters, no optimizer
                  self.backward_optim = None
            else:
                  # Learnable backward policy - create optimizer
                  self.backward_optim = Adam(self.backward_policy.parameters(), lr=self.learning_rate)

            # Combine all optimizers into one list
            self.GFN_optim = [self.forward_optim]
            if self.logZ_optim is not None:
                  self.GFN_optim.append(self.logZ_optim)
            if self.backward_optim is not None:
                  self.GFN_optim.append(self.backward_optim)
            if self.logF_optim is not None:
                  self.GFN_optim.append(self.logF_optim)

            self._n_updates = 0

            self.logger = {
                  'delta_t': time.time_ns(),
                  'iteration': 0,          # current iteration number
                  'episode': 0,            # total number of episodes collected
                  'buffer_lens': [],      # episodic lengths from rollout collection
                  'buffer_rews': [],      # episodic returns from rollout collection
                  'log_Z': [],            # log_Z in current iteration
                  'loss_means': [],       # loss means in current iteration
                  'loss_max': [],         # loss max in current iteration
                  'loss_std': [],         # loss std in current iteration
                  'worst_log_M': [],      # worst log M value (for stable GFN verification)
            }
            
            self.env = env

            # Check if environment supports direct terminal state sampling
            self.has_terminal_sampling = hasattr(env.unwrapped, 'sample_terminal_states')
            # Subgraph certificate state (used when terminal sampling not available)
            if not self.has_terminal_sampling:
                  self.terminal_states = {}  # Maps state tuple -> (obs, reward)
                  # Sorted list of (reward, key) tuples in descending order (highest first, lowest last)
                  # This allows O(1) removal of minimum reward state (pop from end)
                  self.terminal_states_sorted = []  # List of (reward, key) tuples, descending by reward
                  self.best_mass = 0.0  # S_best: best cumulative reward mass
                  self.current_mass = 0.0  # S_curr: current cumulative reward mass
                  self.max_state_size = max_state_size  # Maximum number of terminal states to keep
            # Current mass for subgraph mode
            if not self.has_terminal_sampling:
                  self.logger['current_mass'] = []
            
            # Mode tracking (if environment supports is_mode and we're in subgraph certificate mode)
            self.track_modes = hasattr(env.unwrapped, 'is_mode') and not self.has_terminal_sampling
            if self.track_modes:
                  self.discovered_modes = set()  # Track unique discovered modes
                  # Dict insertion order preserves the encounter order required
                  # by the paper's greedy Tanimoto-separated mode metric.
                  self.discovered_mode_rewards = {}
                  self.logger['total_modes'] = []  # Log total modes discovered over time

            # Initialize wandb if project name is provided
            if wandb_project is not None:
                  wandb.init(
                        project=wandb_project,
                        name=wandb_name,
                        config={
                              'learning_rate': learning_rate,
                              'batch_size': batch_size,
                              'memory_size': memory_size,
                              'train_freq': train_freq,
                              'gradient_steps': gradient_steps,
                              'policy_temperature': policy_temperature,
                              'reward_temperature': reward_temperature,
                              'sample_method': sample_method,
                              'hidden_sizes': hidden_sizes,
                              'temperature_decay_rate': temperature_decay_rate,
                              'use_state_flow': use_state_flow,
                              'backward_policy_provided': backward_policy is not None,
                              'trainable_logPB': trainable_logPB,
                        }
                  )
            self.use_wandb = wandb_project is not None
      
      # Rollout collection methods
      def collect_rollouts(self):
            """
            Collect rollouts: use parallel collection if data_env is available, otherwise sequential.
            
            Returns:
                Tuple of (num_collected_steps, num_collected_episodes)
            """
            if self.data_env is not None:
                  steps, episodes, _, _ = self.collect_rollouts_parallel(num_trajectories=self.train_freq)
            else:
                  steps, episodes, _, _ = self.collect_rollouts_sequential()
            return steps, episodes
      
      def collect_rollouts_sequential(self, predict_fn=None, num_trajectories=None):
            """
            Collect rollouts sequentially (single-threaded).
            
            Args:
                predict_fn: Policy prediction function. Defaults to self.predict.
                            Signature: predict_fn(obs, action_mask) -> action
                num_trajectories: Number of trajectories to collect. Defaults to self.train_freq.
            
            Returns:
                Tuple of (num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews)
            """
            if predict_fn is None:
                  predict_fn = self.predict
            if num_trajectories is None:
                  num_trajectories = self.train_freq
            
            buffer_lens = []
            buffer_rews_report = []

            num_collected_steps, num_collected_episodes = 0, 0

            with torch.no_grad():
                  while num_collected_episodes < num_trajectories:
                        ep_acts = []
                        ep_obs = []
                        ep_next_obs = []
                        ep_rews = []

                        # Reset the environment. Note that obs is short for observation. 
                        obs, _ = self.env.reset()
                        done = False
                        ep_t = 0

                        while not done:
                              # Track observations in this buffer
                              ep_obs.append(obs)
                              # Calculate action and make a step in the env. 
                              # Note that rew is short for reward.
                              action = predict_fn(obs, action_mask=self.env.unwrapped.get_forward_action_masks(obs))
                              obs, rew, done, truncated, _ = self.env.step(action)
                              
                              # Track recent action, and action log probability
                              ep_acts.append(action)
                              ep_next_obs.append(obs)
                              ep_rews.append(rew)

                              ep_t += 1

                        if not truncated:
                              num_collected_steps += len(ep_obs)
                              num_collected_episodes += 1

                              buffer_lens.append(len(ep_obs))
                              buffer_rews_report.append(ep_rews[-1])

                              # Terminal reward is the final reward in the episode
                              terminal_reward = ep_rews[-1]
                              # add to the memory
                              self.memory.push_traj(ep_obs, ep_acts, ep_next_obs, terminal_reward)

                              if not self.has_terminal_sampling:
                                    self._add_terminal_state(ep_obs[-1], ep_rews[-1])
                              else:
                                    # _add_terminal_state already calls _get_mode_key when it runs;
                                    # for envs with sample_terminal_states we need a separate hook.
                                    self._track_mode(ep_obs[-1], reward=ep_rews[-1])

            # Log the episodic returns and episodic lengths in this buffer.
            self.logger['buffer_rews'] = buffer_rews_report
            self.logger['buffer_lens'] = buffer_lens
            self._log_current_mass()

            return num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews_report
      
      def collect_rollouts_parallel(self, predict_fn=None, num_trajectories=None):
            """
            Collect rollouts in parallel using data_env.
            
            Args:
                predict_fn: Policy prediction function. Defaults to self.predict.
                            Signature: predict_fn(obs_batch, action_mask) -> action_batch
                num_trajectories: Number of trajectories to collect. Defaults to self.train_freq.
                            Runs multiple batches if num_trajectories > data_env.num_envs.
            
            Returns:
                Tuple of (num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews)
            """
            if predict_fn is None:
                  predict_fn = self.predict
            if num_trajectories is None:
                  num_trajectories = self.train_freq
            
            num_envs = self.data_env.num_envs
            all_buffer_lens = []
            all_buffer_rews = []
            total_steps, total_episodes = 0, 0
            
            # Run multiple batches if needed
            trajectories_collected = 0
            while trajectories_collected < num_trajectories:
                  batch_size = min(num_envs, num_trajectories - trajectories_collected)
                  
                  buffer_rews_report = np.zeros(batch_size)
                  buffer_lens = np.zeros(batch_size, dtype=int)
                  
                  with torch.no_grad():
                        dones = np.zeros(batch_size, dtype=bool)
                        obs_buffer = []
                        action_buffer = []
                        next_obs_buffer = []
                        obs, _ = self.data_env.reset()
                        
                        # If batch_size < num_envs, only use first batch_size environments
                        if batch_size < num_envs:
                              obs = obs[:batch_size]

                        prev_obs = obs
                        ep_t = 0
                        while not np.all(dones):
                              obs_buffer.append(obs.copy())
                              action = predict_fn(obs, action_mask=self.env.unwrapped.get_forward_action_masks(obs))
                              
                              prev_obs = obs.copy()
                              
                              # Step only the environments we're using
                              if batch_size < num_envs:
                                    # Need to step all envs but only track batch_size
                                    full_action = np.zeros(num_envs, dtype=action.dtype)
                                    full_action[:batch_size] = action
                                    full_obs, full_rew, full_d, full_t, _ = self.data_env.step(full_action)
                                    obs = full_obs[:batch_size]
                                    rew = full_rew[:batch_size]
                                    d = full_d[:batch_size]
                                    t = full_t[:batch_size]
                              else:
                                    obs, rew, d, t, _ = self.data_env.step(action)
                              
                              action_buffer.append(action.copy() if isinstance(action, np.ndarray) else action)
                              obs[d, :] = prev_obs[d, :]
                              next_obs_buffer.append(obs.copy())

                              mask = d & (~dones) & (~t)
                              if np.any(mask):
                                    buffer_lens[mask] = len(obs_buffer)
                                    buffer_rews_report[mask] = rew[mask]
                                    dones[mask] = True
                                    if not self.has_terminal_sampling:
                                          # Add terminal states for all masked trajectories
                                          # prev_obs contains the observations before the termination step
                                          for idx in np.where(mask)[0]:
                                                self._add_terminal_state(prev_obs[idx], buffer_rews_report[idx])
                                    else:
                                          # Mode tracking still runs when terminal sampling is available
                                          for idx in np.where(mask)[0]:
                                                self._track_mode(prev_obs[idx], reward=buffer_rews_report[idx])

                              total_steps += np.sum(~dones)
                              ep_t += 1

                        total_episodes += np.sum(dones)

                        # Push trajectories to memory
                        self.memory.push_trajs(obs_buffer, action_buffer, next_obs_buffer, buffer_rews_report, buffer_lens)

                  all_buffer_lens.extend(buffer_lens.tolist())
                  all_buffer_rews.extend(buffer_rews_report.tolist())
                  trajectories_collected += batch_size
            
            self.logger['buffer_rews'] = all_buffer_rews
            self.logger['buffer_lens'] = all_buffer_lens

            self._log_current_mass()
            
            return total_steps, total_episodes, all_buffer_lens, all_buffer_rews
      
      def _is_initial_obs(self, obs: np.ndarray) -> bool:
            """Check if an observation represents the initial state."""
            # Get raw state from observation
            if hasattr(self.env.unwrapped, 'get_state'):
                  raw_state = self.env.unwrapped.get_state(obs)
            else:
                  raw_state = obs
            if hasattr(self.env.unwrapped, 'is_initial_state'):
                  return self.env.unwrapped.is_initial_state(raw_state)
            # Handle string states (empty string is initial)
            if isinstance(raw_state, str):
                  return raw_state == ""
            # Default: check if state is all zeros (common pattern for numpy arrays)
            return np.all(raw_state == 0)
      
      def collect_backward_rollouts_sequential(self, terminal_states: np.ndarray, rewards: np.ndarray,
                                                backward_sample_fn=None):
            """
            Collect backward trajectories sequentially from terminal states.
            
            Args:
                terminal_states: Array of terminal state observations, shape (num_envs, obs_dim)
                rewards: Array of rewards for terminal states, shape (num_envs,)
                backward_sample_fn: Function to sample backward action. Defaults to self._sample_backward_action.
                
            Returns:
                Tuple of (num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews)
            """
            num_envs = len(terminal_states)
            buffer_lens = []
            buffer_rews = []
            num_collected_steps, num_collected_episodes = 0, 0
            
            if backward_sample_fn is None:
                  backward_sample_fn = self._sample_backward_action
            
            with torch.no_grad():
                  for i in range(num_envs):
                        terminal_obs = terminal_states[i]
                        reward = rewards[i]
                        
                        # Collect single backward trajectory
                        backward_traj = []
                        current_obs = terminal_obs.copy()
                        
                        # Sample backward trajectory using backward policy
                        while not self._is_initial_obs(current_obs):
                              # Get backward action mask
                              backward_mask = self.env.unwrapped.get_backward_action_masks(current_obs)
                              if isinstance(backward_mask, torch.Tensor):
                                    backward_mask = backward_mask.cpu().numpy()
                              
                              # Check if there are any valid backward actions
                              if not np.any(backward_mask):
                                    break
                              
                              # Sample backward action
                              backward_action = backward_sample_fn(current_obs, backward_mask)
                              
                              # Store (obs, action) pair
                              backward_traj.append((current_obs.copy(), int(backward_action)))
                              
                              # Apply backward action to get previous obs
                              current_obs = self._apply_backward_action(current_obs, backward_action)
                        
                        if len(backward_traj) == 0:
                              continue
                        
                        # Convert backward trajectory to forward format
                        ep_obs, ep_acts, ep_next_obs = self._convert_backward_to_forward(
                              backward_traj, terminal_obs
                        )
                        
                        if ep_obs is not None and len(ep_obs) > 0:
                              self.memory.push_traj(ep_obs, ep_acts, ep_next_obs, reward)
                              buffer_lens.append(len(ep_obs))
                              buffer_rews.append(reward)
                              num_collected_steps += len(ep_obs)
                              num_collected_episodes += 1
            
            return num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews
      
      def collect_backward_rollouts_parallel(self, terminal_states: np.ndarray, rewards: np.ndarray,
                                             backward_sample_fn=None, use_truly_parallel=False):
            """
            Collect backward trajectories from terminal states in parallel.
            
            Args:
                terminal_states: Array of terminal state observations, shape (num_envs, obs_dim)
                rewards: Array of rewards for terminal states, shape (num_envs,)
                backward_sample_fn: Function to sample backward action for batch.
                                   Signature: backward_sample_fn(obs_batch, mask_batch) -> action_batch
                use_truly_parallel: Unused, kept for API compatibility
                
            Returns:
                Tuple of (num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews)
            """
            if len(terminal_states) == 0:
                  return 0, 0, [], []
            
            num_envs = len(terminal_states)
            
            # Default batch backward sampling function
            if backward_sample_fn is None:
                  def backward_sample_fn(obs_batch, mask_batch):
                        obs_tensor = torch.tensor(obs_batch, dtype=torch.float, device=self.device)
                        mask_tensor = torch.tensor(mask_batch, dtype=torch.bool, device=self.device)
                        actions = self.backward_policy(obs_tensor, mask_tensor)
                        if isinstance(actions, torch.Tensor):
                              return actions.cpu().numpy()
                        return np.array(actions)
            
            # Initialize current observations for all trajectories
            current_obs_batch = terminal_states.copy()
            
            # Track which trajectories are still active (haven't reached initial state)
            active = np.ones(num_envs, dtype=bool)
            
            # Store backward trajectories: list of (obs, action) for each trajectory
            backward_trajs = [[] for _ in range(num_envs)]
            
            num_collected_steps = 0
            
            with torch.no_grad():
                  while np.any(active):
                        # Get indices of active trajectories
                        active_indices = np.where(active)[0]
                        if len(active_indices) == 0:
                              break
                        
                        # Check for initial states
                        for idx in active_indices:
                              if self._is_initial_obs(current_obs_batch[idx]):
                                    active[idx] = False
                        
                        # Recompute active indices after initial state check
                        active_indices = np.where(active)[0]
                        if len(active_indices) == 0:
                              break
                        
                        active_obs = current_obs_batch[active_indices]
                        
                        # Get backward action masks for active trajectories
                        backward_masks = self.env.unwrapped.get_backward_action_masks(active_obs)
                        if isinstance(backward_masks, torch.Tensor):
                              backward_masks = backward_masks.cpu().numpy()
                        
                        # Check for trajectories with no valid backward actions
                        valid_actions = np.any(backward_masks, axis=1)
                        for i, idx in enumerate(active_indices):
                              if not valid_actions[i]:
                                    active[idx] = False
                        
                        # Recompute active indices after valid action check
                        active_indices = np.where(active)[0]
                        if len(active_indices) == 0:
                              break
                        
                        active_obs = current_obs_batch[active_indices]
                        backward_masks = self.env.unwrapped.get_backward_action_masks(active_obs)
                        if isinstance(backward_masks, torch.Tensor):
                              backward_masks = backward_masks.cpu().numpy()
                        
                        # Sample backward actions for all active trajectories (BATCHED)
                        actions = backward_sample_fn(active_obs, backward_masks)
                        
                        # Store steps and apply backward actions
                        for i, idx in enumerate(active_indices):
                              action = int(actions[i])
                              
                              # Store (obs, action) pair
                              backward_trajs[idx].append((current_obs_batch[idx].copy(), action))
                              
                              # Apply backward action to get previous obs
                              current_obs_batch[idx] = self._apply_backward_action(
                                    current_obs_batch[idx], action
                              )
                              
                              num_collected_steps += 1
            
            # Convert backward trajectories to forward format and push to memory
            buffer_lens = []
            buffer_rews = []
            num_collected_episodes = 0
            
            for i in range(num_envs):
                  if len(backward_trajs[i]) == 0:
                        continue
                  
                  # Convert backward trajectory to forward format
                  ep_obs, ep_acts, ep_next_obs = self._convert_backward_to_forward(
                        backward_trajs[i], terminal_states[i]
                  )
                  
                  if ep_obs is not None and len(ep_obs) > 0:
                        self.memory.push_traj(ep_obs, ep_acts, ep_next_obs, rewards[i])
                        buffer_lens.append(len(ep_obs))
                        buffer_rews.append(rewards[i])
                        num_collected_episodes += 1
            
            # Store in logger
            self.logger['_backward_buffer_rews'] = buffer_rews
            self.logger['_backward_buffer_lens'] = buffer_lens
            
            return num_collected_steps, num_collected_episodes, buffer_lens, buffer_rews
      
      def _convert_backward_to_forward(self, backward_traj, terminal_obs):
            """
            Convert a backward trajectory to forward format.
            
            Args:
                backward_traj: List of (obs, action) tuples in backward order
                terminal_obs: The terminal observation
                
            Returns:
                ep_obs, ep_acts, ep_next_obs in forward order
            """
            if len(backward_traj) == 0:
                  return None, None, None
            
            ep_obs = []
            ep_acts = []
            ep_next_obs = []
            
            # Reverse the backward trajectory to get forward order
            # backward_traj is [(obs_n, a_n), (obs_{n-1}, a_{n-1}), ..., (obs_1, a_1)]
            # where applying a_i backward to obs_i gives obs_{i-1}
            # In forward: from obs_{i-1}, action a_i leads to obs_i
            reversed_traj = list(reversed(backward_traj))
            
            # Get the initial obs (result of applying backward action to first obs in reversed order)
            first_obs, first_action = reversed_traj[0]
            initial_obs = self._apply_backward_action(first_obs, first_action)
            
            current_obs = initial_obs
            
            # Build forward trajectory
            for next_obs, action in reversed_traj:
                  ep_obs.append(current_obs.copy())
                  ep_acts.append(action)
                  ep_next_obs.append(next_obs.copy())
                  current_obs = next_obs
            
            # Add the final transition (exit action from terminal state)
            ep_obs.append(terminal_obs.copy())
            ep_acts.append(self.action_dim - 1)  # Exit action
            if hasattr(self.env.unwrapped, 'final_state'):
                  ep_next_obs.append(self.env.unwrapped.final_state.copy())
            else:
                  ep_next_obs.append(terminal_obs.copy())
            
            return ep_obs, ep_acts, ep_next_obs
      
      def _get_terminal_reward(self, obs: np.ndarray) -> float:
            """Get the reward for a terminal state observation."""
            if hasattr(self.env.unwrapped, '_compute_reward'):
                  raw_state = self.env.unwrapped.get_state(obs.reshape(1, -1))[0]
                  return self.env.unwrapped._compute_reward(raw_state)
            return 1e-10  # Default (min) reward
      
      def _sample_backward_action(self, obs: np.ndarray, action_masks: np.ndarray) -> int:
            """Sample a valid backward action using the backward policy."""
            obs_tensor = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0)
            mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device).unsqueeze(0)
            action = self.backward_policy(obs_tensor, mask_tensor)
            return action.item() if isinstance(action, torch.Tensor) else int(action)
      
      def _apply_backward_action(self, obs: np.ndarray, action: Union[int, np.ndarray]) -> np.ndarray:
            """
            Apply a backward action to get the previous observation.
            
            For discrete grid environments, this typically means decrementing
            the state dimension corresponding to the action.
            For sequence environments (like RNA), this removes a character.
            
            Args:
                obs: Current observation
                action: Backward action to apply
                
            Returns:
                New observation after applying backward action
            """
            return self.env.unwrapped.apply_backward_action(obs, action)

      # learning
      def learn(self, total_iterations, i_start = 0, e_start = 0):
            if self.metrics_jsonl:
                  self._metrics_run_id = uuid.uuid4().hex
            if self.verbose:
                  print(f"Learning... Using {self.batch_size} episodes per updates, ", end='')
                  print(f"at least {self.train_freq} episodes per update for {total_iterations} iterations.")
            iteration = i_start # Iterations ran so far
            episode = e_start # Episodes simulated so far

            val_errs = []
            # Create progress bar (always display)
            pbar = tqdm(total=total_iterations, initial=i_start, desc="Training", 
                       unit="iter", disable=not self.progress, ncols=100)
            
            try:
                  while iteration < total_iterations and not self.converged:
                        # Collect rollouts (uses parallel if data_env is available, otherwise sequential)
                        tmp_t, tmp_e = self.collect_rollouts()
                        episode += tmp_e

                        # Logging iterations and episodes
                        self.logger['episode'] = episode
                        # print(episode)

                        # ensure off-policy training has enough samples
                        if (len(self.memory) >= self.batch_size) or self.memory.is_onpolicy:
                              self.train()

                              iteration += 1

                              self.logger['iteration'] = iteration

                              # Update temperature with decay rate
                              self._update_temperature(iteration)

                              # Get metrics before _log_summary resets them
                              avg_loss = np.mean(self.logger['loss_means']) if len(self.logger['loss_means']) > 0 else 0.0
                              avg_reward = np.mean(self.logger['buffer_rews']) if len(self.logger['buffer_rews']) > 0 else 0.0

                              # Print a summary of our training so far
                              self._log_summary()
                              
                              # A zero interval disables periodic checkpoint output.
                              if self.checkpoint_interval and iteration % self.checkpoint_interval == 0:
                                    self.save_checkpoint(iteration)
                              
                              # Update progress bar
                              pbar.update(1)
                              # Update progress bar description with current metrics
                              pbar.set_postfix({
                                    'loss': f'{avg_loss:.4f}',
                                    'reward': f'{avg_reward:.4f}',
                                    'episodes': episode
                              })
            finally:
                  pbar.close()

      def get_logZ(self):
            """
            Get the current logZ value.
            
            If logF (state flow) is used, returns logF(initial_state).
            Otherwise, returns the learnable logZ parameter.
            
            Returns:
                  logZ value as a torch.Tensor
            """
            if self.logF is not None:
                  # When using state flow, logZ = logF(initial_state)
                  return self.logF(self.initial_state_obs).squeeze()
            else:
                  # When not using state flow, use learnable logZ
                  return self.logZ

      def sample_from_memory(self, batch_size):
            """
            Sample a batch from memory based on sample_method.
            
            sample_method:
                  0: uniform sampling
                  1: reward prioritized sampling (50% from top 10% rewards)
                  2: reference flow prioritized sampling (only for TB_stable)
            
            Args:
                  batch_size: Number of trajectories to sample
                  
            Returns:
                  Same as memory.uniform_sample
            """
            if self.sample_method == 0:
                  return self.memory.uniform_sample(batch_size)
            elif self.sample_method == 1:
                  if self.memory.threshold_needs_update:
                        self.memory.update_threshold(batch_size)
                        self.memory.threshold_needs_update = False
                  return self.memory.reward_prioritized_sample(batch_size)
            else:
                  raise ValueError(f"Invalid sample_method: {self.sample_method}. Must be 0 or 1.")

      def _update_temperature(self, iteration):
            """
            Update policy and reward temperatures using exponential decay.
            
            Temperature decays from initial value to a minimum of 1.0.
            Formula: (initial - 1) * (decay_rate ** iteration) + 1
            
            Args:
                  iteration: Current iteration number
            """
            # Update temperature with decay rate, ensuring minimum of 1.0
            self.policy_temperature = (self.policy_temperature_init - 1.0) * (self.temperature_decay_rate ** iteration) + 1.0
            self.reward_temperature = (self.reward_temperature_init - 1.0) * (self.temperature_decay_rate ** iteration) + 1.0

      def predict(self, obs, action_mask= None): 
            # Query the forward poilcy for an action
            action = self.forward_policy(obs, action_mask, temperature = self.policy_temperature, epsilon = self.epsilon)

            # Return the action
            return action.detach().cpu().numpy()
      
      def _log_summary(self):
            """
                  Print to stdout what we've logged so far in the most recent batch.

                  Parameters:
                        None

                  Return:
                        None
            """
            # Calculate logging values. I use a few python shortcuts to calculate each value
            # without explaining since it's not too important to PPO; feel free to look it over,
            # and if you have any questions you can email me (look at bottom of README)
            delta_t = self.logger['delta_t']
            self.logger['delta_t'] = time.time_ns()
            delta_t = (self.logger['delta_t'] - delta_t) / 1e9
            delta_t = round(delta_t, 2)

            iteration = self.logger['iteration']
            episode = self.logger['episode']
            avg_ep_lens = np.mean(self.logger['buffer_lens'])
            avg_ep_rews = np.mean(self.logger['buffer_rews'])
            avg_log_Z = np.mean(self.logger['log_Z'])

            loss_mean = np.mean(self.logger['loss_means'])
            loss_max = np.mean(self.logger['loss_max'])
            loss_std = np.mean(self.logger['loss_std'])

            # Current mass for subgraph mode
            current_mass = self.logger['current_mass'][-1] if (not self.has_terminal_sampling and len(self.logger['current_mass']) > 0) else None
            
            # Total modes discovered + delta since last log
            total_modes = len(self.discovered_modes) if self.track_modes else None
            new_modes_this_log = None
            if self.track_modes:
                  self.logger['total_modes'].append(total_modes)
                  new_modes_this_log = total_modes - getattr(self, '_modes_at_last_log', 0)
                  self._modes_at_last_log = total_modes

            if self.metrics_jsonl:
                  # Keep full precision locally for max-versus-rest diagnostics.
                  local_unrounded = {
                        'train/loss_mean_unrounded': loss_mean,
                        'train/loss_max_unrounded': loss_max,
                        'train/loss_std_unrounded': loss_std,
                        'train/batch_size': self.batch_size,
                        'train/gradient_steps': self.gradient_steps,
                        'train/loss_means_per_batch': [
                              float(value) if math.isfinite(float(value)) else None
                              for value in self.logger['loss_means']
                        ],
                        'train/loss_max_per_batch': [
                              float(value) if math.isfinite(float(value)) else None
                              for value in self.logger['loss_max']
                        ],
                  }

            # Round decimal places for more aesthetic logging messages
            avg_ep_lens = round(avg_ep_lens, 2)
            avg_ep_rews = round(avg_ep_rews, 2)
            avg_log_Z = round(avg_log_Z, 5)
            loss_mean = round(loss_mean, 5)
            loss_max = round(loss_max, 5)
            loss_std = round(loss_std, 5)

            # Print logging statements
            if self.verbose:
                  print(flush=True)
                  print(f"-------------------- Iteration #{iteration} --------------------", flush=True)
                  print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
                  print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
                  print(f"Average log_Z: {avg_log_Z:.5f}", flush=True)
                  print(f"Average Loss Mean: {loss_mean:.5f}", flush=True)
                  print(f"Average Loss Max: {loss_max:.5f}", flush=True)
                  print(f"Average Loss Std: {loss_std:.5f}", flush=True)
                  if current_mass is not None:
                        print(f"Current Mass: {current_mass:.5f}", flush=True)
                  if total_modes is not None:
                        print(f"Total Modes Discovered: {total_modes}", flush=True)

                  print(f"Total Episodes: {episode}", flush=True)
                  print(f"Iteration took: {delta_t} secs", flush=True)
                  print(f"------------------------------------------------------", flush=True)
                  print(flush=True)

            if self.use_wandb or self.metrics_jsonl:
                  log_dict = {
                        'train/episodic_length': avg_ep_lens,
                        'train/episodic_return': avg_ep_rews,
                        'train/log_Z': avg_log_Z,
                        'train/policy_temperature': self.policy_temperature,
                        'train/reward_temperature': self.reward_temperature,
                        'iteration': iteration,
                        'train/loss_mean': loss_mean,
                        'train/loss_max': loss_max,
                        'train/loss_std': loss_std,
                  }
                  if current_mass is not None:
                        log_dict['train/current_mass'] = current_mass
                  if total_modes is not None:
                        log_dict['train/total_modes'] = total_modes
                  if new_modes_this_log is not None:
                        log_dict['train/new_modes_per_log'] = new_modes_this_log
                  # If env exposes a mode-reward threshold, log it once
                  # (constant per run; useful for sanity-checking).
                  if self.track_modes and hasattr(self.env.unwrapped, 'mode_r_threshold'):
                        log_dict['train/mode_r_threshold'] = float(
                              self.env.unwrapped.mode_r_threshold
                        )
                  if self.use_wandb:
                        wandb.log(log_dict, step=iteration)
                  if self.metrics_jsonl:
                        self._write_local_metrics({**log_dict, **local_unrounded})

            # Reset batch-specific logging data
            self.logger['log_Z'] = []
            self.logger['loss_means'] = []
            self.logger['loss_max'] = []
            self.logger['loss_std'] = []
            if not self.has_terminal_sampling:
                  self.logger['current_mass'] = []

      def _write_local_metrics(self, log_dict):
            """Optionally persist existing scalar diagnostics before subclass resets.

            JSON null represents non-finite diagnostics (for example an
            unverified infinite TV bound). Each learn invocation has its own
            run_id, so reruns appended to one directory stay distinguishable.
            """
            record = {
                  'schema_version': 1,
                  'run_id': self._metrics_run_id,
                  'iteration': self.logger['iteration'],
                  'episode': self.logger['episode'],
                  **log_dict,
            }
            # Stable/StableTeacher expose these logger buffers before their
            # _log_summary override resets them. Read only; no new evaluations.
            if 'tv_bound' in self.logger or 'augmented_loss_means' in self.logger:
                  last_values = {
                        'tv_bound': ('tv_bound', float('inf')),
                        'loss_only_TV_bound': ('loss_only_TV_bound', 1.0),
                        'optimal_c': ('loss_threshold', 0.0),
                        'loss_threshold': ('loss_threshold', 0.0),
                        'stability_patience': ('patience_counter', 0),
                  }
                  for name, (attribute, default) in last_values.items():
                        values = self.logger.get(name, [])
                        record['train/' + name] = values[-1] if values else getattr(self, attribute, default)
                  record['train/accumulated_samples'] = (
                        getattr(self, 'accumulated_fwd_samples', 0)
                        + getattr(self, 'accumulated_bwd_samples', 0)
                  )
                  for prefix in ('augmented_loss', 'teacher_loss'):
                        if prefix + '_means' not in self.logger:
                              continue
                        for suffix, buffer_suffix in (('mean', 'means'), ('max', 'max'), ('std', 'std')):
                              values = self.logger.get(prefix + '_' + buffer_suffix, [])
                              record['train/' + prefix + '_' + suffix] = np.mean(values) if values else 0.0
            for key, value in record.items():
                  if isinstance(value, np.generic):
                        value = value.item()
                  if isinstance(value, float) and not math.isfinite(value):
                        value = None
                  record[key] = value
            os.makedirs(self.model_dir, exist_ok=True)
            with open(os.path.join(self.model_dir, 'training_metrics.jsonl'), 'a', encoding='utf-8') as handle:
                  handle.write(json.dumps(record, allow_nan=False) + '\n')

      def _log_current_mass(self):
            """Log current_mass if in subgraph certificate mode.
            
            This method should be called from train() methods in subclasses
            to log the current cumulative reward mass of discovered terminal states.
            """
            if not self.has_terminal_sampling:
                  self.logger['current_mass'].append(self.current_mass)

      @abstractmethod
      def train(self):
            pass

      def save_memory(self, filepath: str):
            """
            Save the replay buffer memory to disk.
            
            Args:
                  filepath: Path to save the memory file
            """
            if self.model_dir is not None:
                  os.makedirs(self.model_dir, exist_ok=True)
                  full_path = os.path.join(self.model_dir, filepath)
            else:
                  full_path = filepath
            self.memory.save(full_path)
            if self.verbose:
                  print(f"Memory saved to {full_path}")

      def load_memory(self, filepath: str):
            """
            Load the replay buffer memory from disk.
            
            Args:
                  filepath: Path to load the memory file from
            """
            if self.model_dir is not None:
                  full_path = os.path.join(self.model_dir, filepath)
            else:
                  full_path = filepath
            self.memory.load(full_path)
            if self.verbose:
                  print(f"Memory loaded from {full_path}")

      def save(self, model_dir):
            """
            Save the model, optimizers, and training progress.
            
            Args:
                  model_dir: Directory to save the model files
            """
            if not os.path.exists(model_dir):
                  os.makedirs(model_dir)
            
            # Save model state dicts
            torch.save(self.forward_policy.state_dict(), os.path.join(model_dir, 'forward_policy.pth'))
            # Only save backward policy if it has parameters (not uniform policy)
            try:
                  if len(list(self.backward_policy.parameters())) > 0:
                        torch.save(self.backward_policy.state_dict(), os.path.join(model_dir, 'backward_policy.pth'))
            except:
                  # Uniform policy or other non-parameterized policy - skip saving
                  pass
            
            # Save logZ only if it exists (not using logF)
            if self.logZ is not None:
                  torch.save(self.logZ, os.path.join(model_dir, 'logZ.pth'))
            
            # Save logF if it exists
            if self.logF is not None:
                  torch.save(self.logF.state_dict(), os.path.join(model_dir, 'logF.pth'))
            
            # Save optimizer state dicts
            torch.save(self.forward_optim.state_dict(), os.path.join(model_dir, 'forward_optim.pth'))
            if self.backward_optim is not None:
                  torch.save(self.backward_optim.state_dict(), os.path.join(model_dir, 'backward_optim.pth'))
            if self.logZ_optim is not None:
                  torch.save(self.logZ_optim.state_dict(), os.path.join(model_dir, 'logZ_optim.pth'))
            if self.logF_optim is not None:
                  torch.save(self.logF_optim.state_dict(), os.path.join(model_dir, 'logF_optim.pth'))
            
            # Save training progress
            current_progress = {
                  'iteration': self.logger['iteration'],
                  'episode': self.logger['episode'],
                  'policy_temperature': self.policy_temperature,
                  'reward_temperature': self.reward_temperature,
            }
            with open(os.path.join(model_dir, 'progress.pkl'), 'wb') as f:
                  pickle.dump(current_progress, f)

            # if has terminal sampling, save self.terminal_states
            if not self.has_terminal_sampling:
                  with open(os.path.join(model_dir, 'terminal_states.pkl'), 'wb') as f:
                        pickle.dump(self.terminal_states, f)

            
            if self.verbose:
                  print(f"Model saved to {model_dir}")

      def save_checkpoint(self, iteration):
            """
            Save a checkpoint at the specified iteration.
            Saves PF, PB, logF, and logZ to model_dir/model_[iteration].
            
            Args:
                  iteration: Current iteration number
            """
            if self.model_dir is None:
                  return
            
            checkpoint_dir = os.path.join(self.model_dir, f'model_{iteration}')
            if not os.path.exists(checkpoint_dir):
                  os.makedirs(checkpoint_dir)
            
            # Save forward policy (PF)
            torch.save(self.forward_policy.state_dict(), os.path.join(checkpoint_dir, 'forward_policy.pth'))
            
            # Save backward policy (PB) - only if it has parameters
            try:
                if len(list(self.backward_policy.parameters())) > 0:
                    torch.save(self.backward_policy.state_dict(), os.path.join(checkpoint_dir, 'backward_policy.pth'))
            except:
                # Uniform policy or other non-parameterized policy - skip saving
                pass
            
            # Save logF if it exists
            if self.logF is not None:
                  torch.save(self.logF.state_dict(), os.path.join(checkpoint_dir, 'logF.pth'))
            
            # Save logZ if it exists
            if self.logZ is not None:
                  torch.save(self.logZ, os.path.join(checkpoint_dir, 'logZ.pth'))

      def load(self, model_dir, load_optim = False):
            """
            Load the model, optimizers, and training progress.
            
            Args:
                  model_dir: Directory to load the model files from
                  load_optim: Whether to load optimizer states
                  
            Returns:
                  Tuple of (iteration, episode) if load_optim is True, else (0, 0)
            """
            # Load model state dicts
            self.forward_policy.load_state_dict(torch.load(os.path.join(model_dir, 'forward_policy.pth'), map_location=self.device))
            # Only load backward policy if it has parameters and file exists
            backward_policy_path = os.path.join(model_dir, 'backward_policy.pth')
            if os.path.exists(backward_policy_path):
                try:
                    if len(list(self.backward_policy.parameters())) > 0:
                        self.backward_policy.load_state_dict(torch.load(backward_policy_path, map_location=self.device))
                except:
                    # Uniform policy or other non-parameterized policy - skip loading
                    pass
            
            # Load logZ only if it exists (not using logF)
            if self.logZ is not None and os.path.exists(os.path.join(model_dir, 'logZ.pth')):
                  logZ_loaded = torch.load(os.path.join(model_dir, 'logZ.pth'), map_location=self.device)
                  if isinstance(logZ_loaded, torch.Tensor):
                        self.logZ.data = logZ_loaded.data
                  else:
                        self.logZ.data = torch.tensor(logZ_loaded, device=self.device, dtype=torch.float32).data
            
            # Load logF if it exists
            if self.logF is not None and os.path.exists(os.path.join(model_dir, 'logF.pth')):
                  self.logF.load_state_dict(torch.load(os.path.join(model_dir, 'logF.pth'), map_location=self.device))
            
            if load_optim:
                  # Load optimizer state dicts
                  self.forward_optim.load_state_dict(torch.load(os.path.join(model_dir, 'forward_optim.pth'), map_location=self.device))
                  if self.backward_optim is not None and os.path.exists(os.path.join(model_dir, 'backward_optim.pth')):
                        self.backward_optim.load_state_dict(torch.load(os.path.join(model_dir, 'backward_optim.pth'), map_location=self.device))
                  if self.logZ_optim is not None and os.path.exists(os.path.join(model_dir, 'logZ_optim.pth')):
                        self.logZ_optim.load_state_dict(torch.load(os.path.join(model_dir, 'logZ_optim.pth'), map_location=self.device))
                  if self.logF_optim is not None and os.path.exists(os.path.join(model_dir, 'logF_optim.pth')):
                        self.logF_optim.load_state_dict(torch.load(os.path.join(model_dir, 'logF_optim.pth'), map_location=self.device))
                  
                  # Load training progress
                  if os.path.exists(os.path.join(model_dir, 'progress.pkl')):
                        with open(os.path.join(model_dir, 'progress.pkl'), 'rb') as f:
                              progress = pickle.load(f)
                        # Handle both old format (tuple) and new format (dict)
                        if isinstance(progress, dict):
                              iteration = progress.get('iteration', 0)
                              episode = progress.get('episode', 0)
                              self.logger['iteration'] = iteration
                              self.logger['episode'] = episode
                              if 'policy_temperature' in progress:
                                    self.policy_temperature = progress['policy_temperature']
                                    self.policy_temperature_init = progress.get('policy_temperature_init', self.policy_temperature)
                              if 'reward_temperature' in progress:
                                    self.reward_temperature = progress['reward_temperature']
                                    self.reward_temperature_init = progress.get('reward_temperature_init', self.reward_temperature)
                        else:
                              # Backward compatibility with old format (t_so_far, i_so_far, e_so_far)
                              _, iteration, episode = progress
                              self.logger['iteration'] = iteration
                              self.logger['episode'] = episode
                  else:
                        iteration, episode = 0, 0
                  
                  if self.verbose:
                        print(f"Model and optimizers loaded from {model_dir}")
                        print(f"Resuming from iteration {iteration}, episode {episode}")
                  
                  return iteration, episode
            
            if self.verbose:
                  print(f"Model loaded from {model_dir}")
            
            return 0, 0

      def _state_to_key(self, obs):
            """Convert observation to hashable key for terminal state storage."""
            if isinstance(obs, torch.Tensor):
                  obs = obs.cpu().numpy()
            return tuple(obs.flatten().tolist())
      
      def _get_mode_key(self, obs, reward=None):
            """Get the mode key from an observation if it's a mode, otherwise return None.

            The ``is_mode`` check runs *before* any (potentially expensive)
            ``get_state`` decoding -- e.g. for ``envs.sEH_large``
            ``get_state`` rebuilds the molecule from the build trace, so
            we want to avoid it for the (vast majority of) non-modes.
            The cheapest reliable check is a scalar comparison on the
            trajectory's already-computed reward, which envs that
            support it can expose via ``is_mode(reward=...)``
            (sEH_large does). When that succeeds, its result is
            authoritative -- we never run the expensive state-based
            fallback for a non-mode. Envs whose ``is_mode`` only knows
            about states (sEH, rna14) raise ``TypeError`` on the
            ``reward=`` kwarg, and we fall through to the canonical
            state / obs path.
            """
            if not self.track_modes:
                  return None

            # 1. Cheap reward-based check first; if it ran successfully
            #    (env supports the reward= kwarg) we trust its verdict
            #    and skip get_state entirely on non-modes.
            is_mode_result = False
            reward_check_ran = False
            state = None
            if reward is not None:
                  try:
                        is_mode_result = bool(self.env.unwrapped.is_mode(reward=reward))
                        reward_check_ran = True
                  except (TypeError, ValueError, AttributeError, KeyError):
                        # env's is_mode does not accept reward= -- fall
                        # through to the state-based path below.
                        pass

            # 2. State-based fallback for envs without a reward-aware is_mode.
            if not reward_check_ran:
                  if hasattr(self.env.unwrapped, 'get_state'):
                        state = self.env.unwrapped.get_state(obs.reshape(1, -1) if obs.ndim == 1 else obs)
                        if isinstance(state, (list, np.ndarray)) and len(state) > 0:
                              state = state[0]
                  else:
                        state = obs
                  try:
                        is_mode_result = bool(self.env.unwrapped.is_mode(state))
                  except (TypeError, ValueError, AttributeError, KeyError):
                        try:
                              is_mode_result = bool(self.env.unwrapped.is_mode(obs))
                        except (TypeError, ValueError, AttributeError, KeyError):
                              is_mode_result = False

            if not is_mode_result:
                  return None

            # 3. Mode confirmed -- now (and only now) compute the dedup key.
            if state is None:
                  if hasattr(self.env.unwrapped, 'get_state'):
                        state = self.env.unwrapped.get_state(obs.reshape(1, -1) if obs.ndim == 1 else obs)
                        if isinstance(state, (list, np.ndarray)) and len(state) > 0:
                              state = state[0]
                  else:
                        state = obs
            if isinstance(state, str):
                  return state
            elif isinstance(state, (list, tuple, np.ndarray)):
                  return tuple(state) if isinstance(state, (list, np.ndarray)) else state
            else:
                  return str(state)

      def _track_mode(self, obs, reward=None):
            """Add a mode to ``self.discovered_modes`` if ``obs`` is one.

            ``reward`` (the trajectory's scaled reward, available at the
            terminal step) lets envs short-circuit the check to a scalar
            comparison -- see ``_get_mode_key``.
            """
            if not self.track_modes:
                  return
            mode_key = self._get_mode_key(obs, reward=reward)
            record_discovered_mode(
                  self.discovered_modes,
                  self.discovered_mode_rewards,
                  mode_key,
                  reward,
            )
      
      def _add_terminal_state(self, obs, reward):
            """Add a terminal state to the discovered set and check for modes if applicable.
            Maintains only the top max_state_size states by reward.
            Modes are always tracked (never removed) to represent states of high interest."""
            if self.has_terminal_sampling:
                  return  # No-op if terminal sampling is available
            key = self._state_to_key(obs)
            
            # If state already exists, update its reward
            if key in self.terminal_states:
                  return
            
            # New state - check if we need to remove the lowest reward state
            if len(self.terminal_states) >= self.max_state_size:
                  # Get the minimum reward state (last element in descending sorted list)
                  if len(self.terminal_states_sorted) > 0:
                        min_reward, min_key = self.terminal_states_sorted[-1]
                  
                  # Only add if the new reward is higher than or equal tothe minimum
                  if reward >= min_reward:
                        # Remove the lowest reward state (from end of descending list)
                        removed_obs, removed_reward = self.terminal_states.pop(min_key)
                        self.current_mass -= removed_reward
                        # Remove from end of sorted list (O(1))
                        self.terminal_states_sorted.pop()
                        
                        # Add the new state
                        self.terminal_states[key] = (obs.copy() if isinstance(obs, np.ndarray) else obs, reward)
                        self.current_mass += reward
                        # Insert new entry in descending sorted order
                        # Find insertion point (descending: higher rewards first)
                        inserted = False
                        for i, (r, k) in enumerate(self.terminal_states_sorted):
                              if reward >= r:
                                    self.terminal_states_sorted.insert(i, (reward, key))
                                    inserted = True
                                    break
                        if not inserted:
                              # Append to end if it's the smallest reward
                              self.terminal_states_sorted.append((reward, key))
                  else:
                        # New state has lower reward than all existing states, don't add it
                        # But still check if it's a mode (modes are always tracked)
                        self._track_mode(obs, reward=reward)
                        return
            else:
                  # Not at capacity, just add the new state
                  self.terminal_states[key] = (obs.copy() if isinstance(obs, np.ndarray) else obs, reward)
                  self.current_mass += reward
                  # Insert new entry in descending sorted order
                  # Find insertion point (descending: higher rewards first)
                  inserted = False
                  for i, (r, k) in enumerate(self.terminal_states_sorted):
                        if reward > r:
                              self.terminal_states_sorted.insert(i, (reward, key))
                              inserted = True
                              break
                  if not inserted:
                        # Append to end if it's the smallest reward
                        self.terminal_states_sorted.append((reward, key))
            
            # Check for modes - always track modes (states of high interest)
            self._track_mode(obs, reward=reward)

      
