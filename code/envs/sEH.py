import os
import torch
import numpy as np
import gymnasium as gym
import pickle
import pandas as pd
from polyleven import levenshtein

# Get the directory containing this file for data file paths
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sEH')

class Environment(gym.Env):
    """SEH string MDP environment from the GFlowNets paper.
    
    Modified to match rna14.py API structure for consistency.
    """
    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        """Initialize SEH environment with compatible API to rna14."""
        super().__init__()
        
        self.__init_from_blocks_file(os.path.join(_DATA_DIR, 'block_18.json'))
        symbols = '0123456789abcdefghijklmnopqrstuvwxyz' + \
              'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~'
        assert len(self.blocks) <= len(symbols)
        self.alphabet = list(symbols[:len(self.blocks)])
        self.alphabet_set = set(self.alphabet)       
        self.forced_stop_len = 6  

        self.observation_space = gym.spaces.Box(
            shape=(len(self.alphabet) * self.forced_stop_len,), 
            low=0, high=1, dtype=np.float32
        )
        
        # Actions: add at beginning (0 to len-1), add at end (len to 2*len-1), stop (2*len)
        self.action_space = gym.spaces.Discrete(2 * len(self.alphabet) + 1)

        # Load precomputed rewards
        with open(os.path.join(_DATA_DIR, 'sehstr_gbtr_allpreds.pkl'), 'rb') as f:
            self.rewards = pickle.load(f)

        # Scale rewards
        py = np.array(list(self.rewards))
        self.SCALE_REWARD_MAX = 10
        self.SCALE_REWARD_MIN = 1e-3
        self.REWARD_EXP = 6

        py = np.maximum(py, self.SCALE_REWARD_MIN)
        py = py ** self.REWARD_EXP
        self.scale = self.SCALE_REWARD_MAX / max(py)
        py = py * self.scale

        self.scaled_rewards = py

        # Define modes as top % of samples
        mode_percentile = 0.0001
        self.mode_r_threshold = np.percentile(py, 100 * (1 - mode_percentile))
        
        # Compute modes set (indices of high-reward sequences)
        self.modes = set(np.where(py >= self.mode_r_threshold)[0])

        # Initialize state
        self.state = ''
        self.powers = np.array([len(self.alphabet)**i for i in range(self.forced_stop_len)])

    def __init_from_blocks_file(self, blocks_file):
        self.blocks = pd.read_json(blocks_file)
        self.block_smi = self.blocks['block_smi'].to_list()
        self.block_rs = self.blocks['block_r'].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        
        assert all(nr == 2 for nr in self.block_nrs)
        return
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.state = ''
        return self._get_obs(), {}

    def step(self, action):
        """Take an action in the environment"""
        # Stop action
        if action == 2 * len(self.alphabet): 
            reward = self._compute_reward()
            return self._get_obs(), reward, True, False, {'augmented_rew': 1}
        
        # Add character actions
        if len(self.state) < self.forced_stop_len:
            if action < len(self.alphabet):
                # Add at beginning
                self.state = self.alphabet[action] + self.state
            else:
                # Add at end
                self.state += self.alphabet[action - len(self.alphabet)]
        
        return self._get_obs(), 0, False, False, {}
    
    def _compute_reward(self):
        """Compute reward for current sequence"""
        if len(self.state) == 0:
            return 0
        
        digits = np.argmax(self._get_obs().reshape(self.forced_stop_len, len(self.alphabet)), axis=1)
        index = int(np.dot(digits, self.powers))
        return self.scaled_rewards[index]

    def real_reward(self):
        """Get unscaled reward for current sequence"""
        if len(self.state) == 0:
            return 0
        
        digits = np.argmax(self._get_obs().reshape(self.forced_stop_len, len(self.alphabet)), axis=1)
        index = int(np.dot(digits, self.powers))
        # Return pre-scaled reward
        r = list(self.rewards)[index]
        return max(r, self.SCALE_REWARD_MIN)
    
    def symbol_ohe(self, symbol):
        """One-hot encode a symbol"""
        zs = np.zeros(len(self.alphabet))
        zs[self.alphabet.index(symbol)] = 1.0
        return zs

    def _get_obs(self):
        """Get current observation as one-hot encoded sequence"""
        x_ft = np.concatenate(
            [self.symbol_ohe(c) for c in self.state] + 
            [np.zeros(len(self.alphabet))] * (self.forced_stop_len - len(self.state))
        )
        return x_ft.astype(np.float32)

    def get_state(self, obs):
        """Convert one or multiple one-hot encoded observations back to sequences.
        
        Args:
            obs: One-hot encoded observation(s). Can be 1D (single) or 2D (batch).
            
        Returns:
            String sequence or list of string sequences.
        """
        # Ensure obs is a numpy array for consistent processing
        obs = np.asarray(obs)

        # Check if the input is a single observation (1D) or a batch (2D)
        if obs.ndim == 1:
            obs_batch = [obs]
        else:
            obs_batch = obs

        sequences = []
        for single_obs in obs_batch:
            seq = ""
            for i in range(self.forced_stop_len):
                # Extract the one-hot vector for the character at position i
                char_onehot = single_obs[i * len(self.alphabet):(i + 1) * len(self.alphabet)]

                # Check if a character exists at this position
                if np.sum(char_onehot) > 0.5:
                    char_idx = np.argmax(char_onehot)
                    seq += self.alphabet[char_idx]
                else:
                    # If we find a position with no character, the sequence has ended
                    break
            sequences.append(seq)
        
        # If the original input was a single observation, return a single string
        return sequences[0] if obs.ndim == 1 else sequences
    
    def is_mode(self, sequence=None):
        """Check if current sequence (or given sequence) is a mode (high reward)"""
        if sequence is None:
            sequence = self.state
        
        # Convert sequence to index
        if len(sequence) == 0:
            return False
        
        # Build observation from sequence
        obs = np.zeros(len(self.alphabet) * self.forced_stop_len, dtype=np.float32)
        for i, char in enumerate(sequence):
            if i < self.forced_stop_len and char in self.alphabet_set:
                char_idx = self.alphabet.index(char)
                obs[i * len(self.alphabet) + char_idx] = 1.0
        
        digits = np.argmax(obs.reshape(self.forced_stop_len, len(self.alphabet)), axis=1)
        index = int(np.dot(digits, self.powers))
        
        return index in self.modes
    
    def get_forward_action_masks(self, obs):
        """Get mask for valid forward actions (handles 1D, 2D, and 3D inputs)"""
        action_dim = 2 * len(self.alphabet) + 1
        is_torch = isinstance(obs, torch.Tensor)
        
        if not is_torch:
            obs = np.asarray(obs)
        original_ndim = obs.ndim
        
        # Flatten input to 2D to unify logic
        if original_ndim == 1:
            obs_flat = obs.reshape(1, -1) if not is_torch else obs.unsqueeze(0)
        else:
            obs_flat = obs.reshape(-1, obs.shape[-1]) if not is_torch else obs.view(-1, obs.shape[-1])

        # Core masking logic
        if is_torch:
            seq_len = torch.sum(obs_flat.view(-1, self.forced_stop_len, len(self.alphabet)), dim=-1).sum(-1)
            at_max_len = seq_len >= self.forced_stop_len
            
            masks_flat = torch.zeros(obs_flat.shape[0], action_dim, device=obs.device, dtype=torch.bool)
            masks_flat[~at_max_len, :-1] = True
            masks_flat[at_max_len, -1] = True
        else:
            seq_len = np.sum(obs_flat.reshape(-1, self.forced_stop_len, len(self.alphabet)), axis=-1).sum(-1)
            at_max_len = seq_len >= self.forced_stop_len

            masks_flat = np.zeros((obs_flat.shape[0], action_dim), dtype=bool)
            masks_flat[~at_max_len, :-1] = True
            masks_flat[at_max_len, -1] = True
        
        # Reshape back to match input dimensionality
        if original_ndim == 3:
            output_shape = (obs.shape[0], obs.shape[1], action_dim)
            masks = masks_flat.view(output_shape) if is_torch else masks_flat.reshape(output_shape)
        elif original_ndim == 2:
            masks = masks_flat
        else:
            masks = masks_flat.squeeze(0) if is_torch else masks_flat.squeeze(0)
            
        return masks
        
    def get_backward_action_masks(self, obs):
        """Get mask for valid backward actions (handles 1D, 2D and 3D inputs)
        
        Returns masks with same shape as action space (2 * len(alphabet) + 1),
        where the last action (stop) is always False for backward.
        """
        char_action_dim = 2 * len(self.alphabet)
        full_action_dim = char_action_dim + 1
        is_torch = isinstance(obs, torch.Tensor)
        
        if not is_torch:
            obs = np.asarray(obs)
        original_ndim = obs.ndim
        original_shape = obs.shape
        
        # Flatten input to 2D
        if original_ndim == 1:
            obs_flat = obs.reshape(1, -1) if not is_torch else obs.unsqueeze(0)
        else:
            obs_flat = obs.reshape(-1, original_shape[-1]) if not is_torch else obs.view(-1, original_shape[-1])

        # Core masking logic
        if is_torch:
            reshaped = obs_flat.view(-1, self.forced_stop_len, len(self.alphabet))
            any_char = torch.sum(reshaped, dim=-1) > 0
            
            masks_flat = torch.zeros(obs_flat.shape[0], full_action_dim, device=obs.device, dtype=torch.bool)
            
            batch_indices = torch.arange(obs_flat.shape[0], device=obs.device)
            
            first_pos = torch.argmax(any_char.float(), dim=-1)
            last_pos = any_char.shape[-1] - 1 - torch.argmax(any_char.flip(dims=[-1]).float(), dim=-1)
            
            has_chars = torch.any(any_char, dim=-1)
            
            if torch.any(has_chars):
                valid_indices = batch_indices[has_chars]
                
                first_char_idx = torch.argmax(reshaped[valid_indices, first_pos[valid_indices]].float(), dim=-1)
                last_char_idx = torch.argmax(reshaped[valid_indices, last_pos[valid_indices]].float(), dim=-1)
                
                masks_flat[valid_indices, first_char_idx] = True
                masks_flat[valid_indices, last_char_idx + len(self.alphabet)] = True

        else:  # Numpy version
            reshaped = obs_flat.reshape(-1, self.forced_stop_len, len(self.alphabet))
            any_char = np.sum(reshaped, axis=-1) > 0
            
            masks_flat = np.zeros((obs_flat.shape[0], full_action_dim), dtype=bool)

            for i in range(obs_flat.shape[0]):
                if np.any(any_char[i]):
                    char_positions = np.where(any_char[i])[0]
                    first_pos, last_pos = char_positions[0], char_positions[-1]
                    
                    first_char_idx = np.argmax(reshaped[i, first_pos])
                    masks_flat[i, first_char_idx] = True
                    
                    last_char_idx = np.argmax(reshaped[i, last_pos])
                    masks_flat[i, last_char_idx + len(self.alphabet)] = True
            
        # Reshape back to match input dimensionality
        if original_ndim == 3:
            output_shape = (original_shape[0], original_shape[1], full_action_dim)
            masks = masks_flat.view(output_shape) if is_torch else masks_flat.reshape(output_shape)
        elif original_ndim == 2:
            masks = masks_flat
        else:
            masks = masks_flat.squeeze(0) if is_torch else masks_flat.squeeze(0)
            
        return masks

    def is_initial_state(self, state):
        """Check if the given state is the initial (empty) state.
        
        Args:
            state: Either a string state or numpy observation
            
        Returns:
            bool: True if state is the initial empty state
        """
        if isinstance(state, str):
            return state == ""
        # Handle numpy observation - check if all zeros
        return np.all(state == 0)
    
    def apply_backward_action(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Apply a backward action to the given obs.
        
        Backward actions undo forward actions:
        - Actions 0 to len(alphabet)-1: Remove first character
        - Actions len(alphabet) to 2*len(alphabet)-1: Remove last character
        
        Args:
            obs: One-hot encoded observation
            action: Backward action to apply
            
        Returns:
            New observation after applying backward action
        """
        state = self.get_state(obs)
        
        if len(state) == 0:
            return obs.copy() if isinstance(obs, np.ndarray) else obs
        
        if action < len(self.alphabet):
            # Remove first character
            new_state = state[1:]
        else:
            # Remove last character
            new_state = state[:-1]
        
        # Convert state back to observation
        new_obs = np.zeros(len(self.alphabet) * self.forced_stop_len, dtype=np.float32)
        for i, char in enumerate(new_state):
            if i < self.forced_stop_len and char in self.alphabet_set:
                char_idx = self.alphabet.index(char)
                new_obs[i * len(self.alphabet) + char_idx] = 1.0
        
        return new_obs

    def get_num_of_reachable_end_states(self, obs):
        """Get the number of reachable end states from the given observation
        
        From a current sequence, we can continue adding characters (at beginning or end)
        until we reach forced_stop_len. The number of reachable end states is 
        alphabet_size^(remaining_length).
        
        Args:
            obs: One-hot encoded observation (1D array or batched)
            
        Returns:
            Number of reachable terminal states
        """
        # Handle batched observations
        if len(obs.shape) > 1:
            obs = obs.flatten() if obs.shape[0] == 1 else obs[0]
        
        # Get the current state (sequence) from observation
        state = self.get_state(obs)
        
        # Calculate remaining length
        current_length = len(state)
        remaining_length = self.forced_stop_len - current_length
        
        # If already at max length, only 1 reachable end state
        if remaining_length <= 0:
            return 1
        
        # Number of reachable end states = alphabet_size^(remaining_length)
        return len(self.alphabet) ** remaining_length

    def unnormalize(self, r):
        """Unnormalize reward back to original scale"""
        r = r / self.scale
        r = r ** (1 / self.REWARD_EXP)
        return r

    def dist_func(self, seq1, seq2):
        """Distance function between sequences (Levenshtein distance)"""
        return levenshtein(seq1, seq2)

    def get_error(self, samples):
        """Get the number of unique high reward (mode) samples visited.
        
        This matches the logic used during training in gfn_base.py's _get_mode_key:
        1. Convert observations to states using get_state
        2. Check each state with is_mode
        3. Count unique modes using a set
        
        NOTE: This counts unique modes in the provided samples only, not cumulative
        modes discovered during training. During training, W&B logs cumulative unique
        modes discovered over the entire training run, which will typically be higher
        than the count from a single batch of samples.
        """
        # Handle both state strings (like rna14) and numpy array observations
        if isinstance(samples, list) and len(samples) > 0 and isinstance(samples[0], str):
            # Samples are state strings - use is_mode to check each one
            unique_modes = set([s for s in samples if self.is_mode(s)])
            return len(unique_modes)
        else:
            # Samples are numpy arrays (observations) - convert to states first
            # Match the logic from gfn_base.py _get_mode_key:
            # - Reshape 1D obs to (1, -1) before calling get_state
            # - Take first element if get_state returns a list/array
            if isinstance(samples, np.ndarray):
                # Handle both 1D and 2D arrays
                if len(samples.shape) == 1:
                    # Single observation - reshape to (1, -1) to match training logic
                    state = self.get_state(samples.reshape(1, -1))
                    # get_state returns a list for batch input, take first element
                    if isinstance(state, (list, np.ndarray)) and len(state) > 0:
                        state_strings = [state[0]]
                    else:
                        state_strings = [state]
                else:
                    # Batch of observations
                    state_strings = self.get_state(samples)
                    # get_state returns a list for batch input
                    if not isinstance(state_strings, list):
                        state_strings = [state_strings]
                
                # Now use is_mode to check each state and get unique modes
                unique_modes = set([s for s in state_strings if self.is_mode(s)])
                return len(unique_modes)
            else:
                # Fallback: try to convert to numpy array
                try:
                    samples = np.asarray(samples)
                    if len(samples.shape) == 1:
                        # Reshape to match training logic
                        state = self.get_state(samples.reshape(1, -1))
                        if isinstance(state, (list, np.ndarray)) and len(state) > 0:
                            state_strings = [state[0]]
                        else:
                            state_strings = [state]
                    else:
                        state_strings = self.get_state(samples)
                        if not isinstance(state_strings, list):
                            state_strings = [state_strings]
                    unique_modes = set([s for s in state_strings if self.is_mode(s)])
                    return len(unique_modes)
                except:
                    # Last resort: old method (should not be reached with proper inputs)
                    reshaped = samples.reshape(-1, self.forced_stop_len, len(self.alphabet))
                    digits = np.argmax(reshaped, axis=2)
                    indices = digits.dot(self.powers) 

                    visited = np.zeros_like(self.scaled_rewards, dtype=bool)
                    visited[np.unique(indices)] = True
                    high_reward_mask = self.scaled_rewards > 1e-10
                    num_high_reward_visited = np.sum(visited & high_reward_mask)

                    return num_high_reward_visited

    def render(self):
        """Render current state"""
        print(f"Current sequence: '{self.state}' (length: {len(self.state)})")


if __name__ == "__main__":
    # =========================================================================
    # Show sparsity analysis
    np.random.seed(42)
    from tqdm import trange
    
    num_samples = 100_000
    env = Environment()
    
    print("Testing sEH environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Alphabet size: {len(env.alphabet)}")
    print(f"Max sequence length: {env.forced_stop_len}")
    
    rs = []
    modes_found = 0
    for _ in trange(num_samples):
        obs, _ = env.reset()
        done = False
        while not done:
            valid_actions = env.get_forward_action_masks(obs)
            action_index = np.random.choice(int(np.sum(valid_actions)))
            action = np.where(valid_actions == 1)[0][action_index]
            obs, r, done, _, _ = env.step(action)
        rs.append(r)
        if env.is_mode():
            modes_found += 1
    
    rs = np.array(rs)
    print(f"\nTotal high-reward states: {np.sum(env.scaled_rewards > 1e-3)}")
    print(f"High rewards sampled: {np.sum(rs > 1e-3)}")
    print(f"Modes found: {modes_found}")
    print(f"Mean reward: {np.mean(rs):.6f}")
    print(f"Max reward: {np.max(rs):.6f}")
