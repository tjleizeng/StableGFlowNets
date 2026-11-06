import numpy as np
import torch
from collections import deque, namedtuple
import pickle
import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state')) # store the transition of each step

class Memory:
      # different from RL, the size is the number of episodes but not the number of transitions
      # each trajectory is hashed so that no duplicate trajectories are stored
      def __init__(self, size, device = 'cpu', is_onpolicy = False):
            self.memory = [None] * size  # Preallocate a fixed-size list
            self.rewards = np.zeros(size, dtype=np.float64)  # Initialize as NumPy array, float64 for not losing any precision in this stage
            # self.reference_flows = np.zeros(size, dtype=np.float64) # This is trajectory-level augmented rewards

            self.size = size
            self.device = device
            self.is_onpolicy = is_onpolicy  # On-policy mode: no random sampling, no threshold updates

            self.threshold_for_reward_prioritized = 0 # 50% for top 10% (augmented) rewards
            self.threshold_for_reference_flow_prioritized = 0 # 50% for top 10% (augmented) flows + reward
            self.top_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
            self.bottom_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)

            self.threshold_needs_update = False

            self.current_index = 0  # Tracks the circular position
            self.num_elements = 0  # Tracks the number of valid elements

      # def update_reference_flows(self, idx, reference_flow):
      #       """Update reference flows. Supports both single index and batch updates."""
      #       self.reference_flows[idx] = reference_flow

      def push_traj(self, state_buffer, action_buffer, next_state_buffer, reward_buffer):
            traj = [
                  Transition(state, action, next_state)
                  for state, action, next_state in zip(state_buffer, action_buffer, next_state_buffer)
            ]
            # Add the trajectory at the current index
            self.memory[self.current_index] = traj

            # Update rewards and losses
            self.rewards[self.current_index] = reward_buffer

            # Update the circular index and number of elements
            self.current_index = (self.current_index + 1) % self.size
            if self.num_elements < self.size:
                  self.num_elements += 1

            # Only mark for threshold update in off-policy mode
            if not self.is_onpolicy:
                  self.threshold_needs_update = True

      def push_trajs(self, state_buffer, action_buffer, next_state_buffer, reward_buffer, length_buffer): # push multiple trajectories
            for i in range(len(length_buffer)):
                  self.push_traj(
                        [state_buffer[j][i] for j in range(length_buffer[i])],
                        [action_buffer[j][i] for j in range(length_buffer[i])],
                        [next_state_buffer[j][i] for j in range(length_buffer[i])],
                        reward_buffer[i]
                  )

      def update_threshold(self, batch_size):
            if self.num_elements < 10:
                  # Use the existing uniform fallback during replay warmup.
                  self.top_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
                  self.bottom_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
                  return
            self.threshold_for_reward_prioritized = np.percentile(self.rewards[:self.num_elements], 90)
            self.top_idx_for_reward_prioritized =  np.where(self.rewards[:self.num_elements] >= self.threshold_for_reward_prioritized)[0]
            self.bottom_idx_for_reward_prioritized = np.where(self.rewards[:self.num_elements] < self.threshold_for_reward_prioritized)[0]
            
            if len(self.top_idx_for_reward_prioritized) > 0:
                  if len(self.top_idx_for_reward_prioritized) < batch_size // 2:
                        self.top_idx_for_reward_prioritized = np.tile(self.top_idx_for_reward_prioritized, (batch_size // 2) // len(self.top_idx_for_reward_prioritized) + 1)
            else:
                  self.top_idx_for_reward_prioritized = self.bottom_idx_for_reward_prioritized.copy()

            if len(self.bottom_idx_for_reward_prioritized) > 0:
                  if len(self.bottom_idx_for_reward_prioritized) < batch_size // 2:
                        self.bottom_idx_for_reward_prioritized = np.tile(self.bottom_idx_for_reward_prioritized, (batch_size // 2) // len(self.bottom_idx_for_reward_prioritized) + 1)
            else:
                  self.bottom_idx_for_reward_prioritized = self.top_idx_for_reward_prioritized.copy()

            # augmented_flows = self.reference_flows[:self.num_elements] # + self.rewards[:self.num_elements]
            # self.threshold_for_reference_flow_prioritized = np.percentile(augmented_flows, 90)
            # self.top_idx_for_reference_flow_prioritized =  np.where(augmented_flows >= self.threshold_for_reference_flow_prioritized)[0]
            # self.bottom_idx_for_reference_flow_prioritized = np.where(augmented_flows < self.threshold_for_reference_flow_prioritized)[0]
            
            # if len(self.top_idx_for_reference_flow_prioritized) > 0:
            #       if len(self.top_idx_for_reference_flow_prioritized) < batch_size // 2:
            #             self.top_idx_for_reference_flow_prioritized = np.tile(self.top_idx_for_reference_flow_prioritized, (batch_size // 2) // len(self.top_idx_for_reference_flow_prioritized) + 1)

      def uniform_sample(self, batch_size):
            # sample a batch of full trajectories
            if self.is_onpolicy:
                  # On-policy: return all stored trajectories in order (no random sampling)
                  idx = np.arange(self.num_elements)
            else:
                  # Off-policy: random sampling
                  idx = np.random.choice(self.num_elements, batch_size, replace = False)
            return self.sample_from_indices(idx)
      
      def reward_prioritized_sample(self, batch_size):
            """
                  Select trajecotries for offline training, returns a batch of trajectories
                  Draw 50% from top 10% of the rewards, and 50% from bottom 90%.
            """
            # edge case: top_idx or bottom_idx is empty
            if len(self.top_idx_for_reward_prioritized) > 0 and len(self.bottom_idx_for_reward_prioritized) > 0:
                  idx1 = np.random.choice(self.top_idx_for_reward_prioritized, batch_size // 2, replace=False)
                  idx2 = np.random.choice(self.bottom_idx_for_reward_prioritized, batch_size // 2, replace=False)
                  idx = np.concatenate([idx1, idx2])
            else:
                  idx = np.random.choice(self.num_elements, batch_size, replace=False)

            return self.sample_from_indices(idx)
      
      def sample_from_indices(self, idx):
            sample_traj = [self.memory[i] for i in idx]
            state_batches = [torch.tensor(np.array([t[0] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            action_batches = [torch.tensor(np.array([t[1] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            next_state_batches = [torch.tensor(np.array([t[2] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            reward_batches = [torch.tensor(np.array([self.rewards[i]]), dtype=torch.float, device = self.device) for i in idx]
            return state_batches, action_batches, next_state_batches, reward_batches, idx
      
      def latest_sample(self, batch_size = 1, offset = 0): # get n trajectories from N - batch_size-offset to N - offset, used for PBP (Pessimistic Backward Policy) and stable GFlowNets
            # Handle the wrap-around for the circular buffer
            start_idx = self.current_index - batch_size - offset
            if start_idx < 0:
                  start_idx += self.size
            if start_idx + batch_size <= self.size:
                  sample_indices = list(range(start_idx, start_idx + batch_size))
            else:
                  sample_indices = (
                        list(range(start_idx, self.size)) +
                        list(range(0, (start_idx + batch_size) % self.size))
                  )
            sample_traj = [self.memory[i] for i in sample_indices]
            # process the input
            state_batches = [torch.tensor(np.array([t[0] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            action_batches = [torch.tensor(np.array([t[1] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            next_state_batches = [torch.tensor(np.array([t[2] for t in traj]), dtype=torch.float, device = self.device) for traj in sample_traj]
            reward_batches = [torch.tensor(np.array([self.rewards[i]]), dtype=torch.float, device = self.device) for i in sample_indices]
            return state_batches, action_batches, next_state_batches, reward_batches, sample_indices

      def clear(self):
            self.memory = [None] * self.size  # Preallocate a fixed-size list
            self.rewards = np.zeros(self.size, dtype=np.float64)
            self.reference_flows = np.ones(self.size, dtype=np.float64)
            self.current_index = 0
            self.num_elements = 0
            self.threshold_needs_update = False
            self.top_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
            self.bottom_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)

      def save(self, directory): # since we don't support resume training, we don't need to save the full memory
            # save the memory with state information
            with open(directory, 'wb') as f:
                  pickle.dump({
                        'memory': self.memory,
                        'rewards': self.rewards,
                        # 'reference_flows': self.reference_flows,
                        'current_index': self.current_index,
                        'num_elements': self.num_elements
                  }, f)
      
      def load(self, directory):
            # load the memory and restore state
            with open(directory, 'rb') as f:
                  data = pickle.load(f)
            # Handle both old format (list) and new format (dict)
            if isinstance(data, dict):
                  self.memory = data['memory']
                  self.rewards = data['rewards']
                  # self.reference_flows = data['reference_flows']
                  self.current_index = data.get('current_index', 0)
                  self.num_elements = data.get('num_elements', 0)
            else:
                  # Backward compatibility with old format
                  self.memory, self.rewards, self.reference_flows = data
                  # Restore state: count non-None elements and set current_index
                  self.num_elements = sum(1 for item in self.memory if item is not None)
                  if self.num_elements > 0:
                        # Find the last non-None index to set current_index
                        for i in range(self.size - 1, -1, -1):
                              if self.memory[i] is not None:
                                    self.current_index = (i + 1) % self.size
                                    break
                  else:
                        self.current_index = 0
            # Saved memories do not store the reward-priority index sets.
            self.top_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
            self.bottom_idx_for_reward_prioritized = np.empty(0, dtype=np.int64)
            self.threshold_needs_update = not self.is_onpolicy
      
      def __len__(self):
            return self.num_elements

