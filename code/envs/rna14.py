'''
    RNA from flexs
    Converted to gym API compatibility
'''
import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import gymnasium as gym
from polyleven import levenshtein
import argparse

import flexs

# Get the directory containing this file for data file paths
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rna')

class Environment(gym.Env):
    """RNA environment compatible with gym API"""
    metadata = {"render_modes": []}
    def __init__(self, render_mode = None):
        """Initialize RNA environment with flexs oracle and gym API compatibility"""
        super().__init__()
        
        defaults = {
            "rna_task": 1,
            "rna_length": 14,
            "forced_stop_len": 14,
            "scale_reward_min": 1e-3,
            "scale_reward_max": 10,
            "beta": 40,
            "mode_metric": "hammingball",
            "mode_hammingball_dist": 1,
            "mode_percentile": 0.0001,
                }
        self.args = argparse.Namespace(**defaults)

        
        self.alphabet = ["U", "C", "G", "A"]
        self.alphabet_set = set(self.alphabet)
        self.rna_task = self.args.rna_task
        self.rna_length = self.args.rna_length
        self.forced_stop_len = self.args.forced_stop_len
        
        # Set up observation and action spaces
        # Observation space: one-hot encoding for each position in the sequence
        self.observation_space = gym.spaces.Box(
            shape=(len(self.alphabet) * self.forced_stop_len,), 
            low=0, high=1, dtype=np.float32
        )
        
        # Action space: add character at beginning, add at end, or stop
        # Actions 0-3: add U,C,G,A at beginning
        # Actions 4-7: add U,C,G,A at end  
        # Action 8: stop
        self.action_space = gym.spaces.Discrete(2 * len(self.alphabet) + 1)
        
        # Initialize RNA oracle and reward system
        self._setup_oracle()
        self._setup_reward_system()
        self._setup_modes()
        
        # Initialize state
        self.state = ""
        self.powers = np.array([len(self.alphabet)**i for i in range(self.forced_stop_len)])

    def _setup_oracle(self):
        """Setup the RNA oracle from flexs"""
        # print(f'Loading oracle ...')
        problem = flexs.landscapes.rna.registry()[f'L{self.rna_length}_RNA{self.rna_task}']
        self.oracle = flexs.landscapes.rna.RNABinding(**problem['params'])
        # print(problem)

    def _setup_reward_system(self):
        """Setup reward scaling system"""
        self.SCALE_REWARD_MIN = self.args.scale_reward_min
        self.SCALE_REWARD_MAX = self.args.scale_reward_max
        self.REWARD_EXP = self.args.beta
        
        # Load or compute reward statistics
        df = pd.read_csv(os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_info.csv"))
        if self.REWARD_EXP in df["reward_exp"].values.tolist():
            self.logZ = df[df["reward_exp"] == self.REWARD_EXP]["logz"].values[0]
            self.expected_reward = df[df["reward_exp"] == self.REWARD_EXP]["expected_reward"].values[0]
            self.logZ_scaled = df[df["reward_exp"] == self.REWARD_EXP]["logz_scaled"].values[0]
            self.expected_reward_scaled = df[df["reward_exp"] == self.REWARD_EXP]["expected_reward_scaled"].values[0]
            self.scale = df[df["reward_exp"] == self.REWARD_EXP]["scale"].values[0]
        else:
            self._compute_reward_statistics()
            
        # print(f"Beta: {self.REWARD_EXP}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}")

    def _compute_reward_statistics(self):
        """Compute reward statistics from oracle data"""
        # print("Loading data...")
        start_time = time.time()
        with open(os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_allpreds.pkl"), 'rb') as f:
            oracle_d = pickle.load(f)
        py = np.array(list(oracle_d.values()))
        # print(f"Completed\t {time.time() - start_time:.2f}s")
        
        py = np.maximum(py, self.SCALE_REWARD_MIN)
        log_py = np.log(py)
        
        self.logZ = torch.logsumexp(torch.from_numpy(log_py * self.REWARD_EXP), 0).item()
        logr_square = torch.logsumexp(torch.from_numpy(log_py * (self.REWARD_EXP + 1)), 0).item()
        self.expected_reward = np.exp(logr_square - self.logZ)
        
        log_py_scaled = log_py * self.REWARD_EXP
        log_scale = np.log(self.SCALE_REWARD_MAX) - max(log_py_scaled)
        log_py_scaled = log_py_scaled + log_scale
        self.logZ_scaled = torch.logsumexp(torch.from_numpy(log_py_scaled).float(), dim=0).item()
        logr_square_scaled = torch.logsumexp(torch.from_numpy(log_py_scaled + log_py).float(), dim=0).item()
        self.expected_reward_scaled = np.exp(logr_square_scaled - self.logZ_scaled)
        
        self.scale = np.exp(log_scale)
        del py, log_py, oracle_d

    def _setup_modes(self):
        """Setup mode computation based on args.mode_metric"""
        if self.args.mode_metric == "threshold":
            mode_file = os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_modes_percentile_{self.args.mode_percentile}.pkl")
            # print(mode_file)
            if os.path.exists(mode_file):
                with open(mode_file, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                self._compute_threshold_modes(mode_file)
        elif self.args.mode_metric == "hammingball":
            mode_file = os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_modes_hammingball_dist{self.args.mode_hammingball_dist}_percentile_{self.args.mode_percentile}.pkl")
            if os.path.exists(mode_file):
                with open(mode_file, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                self._compute_hammingball_modes(mode_file)
        
        # print(f"Mode metric: {self.args.mode_metric}\tFound num modes: {len(self.modes)}")

    def _compute_threshold_modes(self, mode_file):
        """Compute modes using threshold method"""
        # print("Loading data...")
        start_time = time.time()
        with open(os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_allpreds.pkl"), 'rb') as f:
            oracle_d = pickle.load(f)
        py = np.array(list(oracle_d.values()))
        # print(f"Completed\t {time.time() - start_time:.2f}s")
        
        mode_percentile = self.args.mode_percentile
        # print(f"Computing modes with threshold {mode_percentile}...")
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))
        self.modes = set([x for x, y in tqdm(oracle_d.items()) if y >= self.mode_r_threshold])
        
        with open(mode_file, "wb") as f:
            pickle.dump(self.modes, f)
        del oracle_d, py

    def _compute_hammingball_modes(self, mode_file):
        """Compute modes using hamming ball method"""
        # print("Loading data...")
        start_time = time.time()
        with open(os.path.join(_DATA_DIR, f"L{self.rna_length}_RNA{self.rna_task}_allpreds.pkl"), 'rb') as f:
            oracle_d = pickle.load(f)
        py = np.array(list(oracle_d.values()))
        # print(f"Completed\t {time.time() - start_time:.2f}s")
        
        mode_percentile = self.args.mode_percentile
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))
        self.mode_hammingball_dist = self.args.mode_hammingball_dist
        
        # print(f"Computing modes with hamming ball distance {self.mode_hammingball_dist}...")
        self.modes = set()
        for x, y in tqdm(oracle_d.items()):
            if y >= self.mode_r_threshold:
                if len(self.modes) == 0:
                    self.modes.add(x)
                else:
                    flag = False
                    for mode in self.modes:
                        edit_dist = levenshtein(x, mode)
                        if edit_dist <= self.mode_hammingball_dist:
                            flag = True
                            break
                    if not flag:
                        self.modes.add(x)
        
        with open(mode_file, "wb") as f:
            pickle.dump(self.modes, f)
        del oracle_d, py

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.state = ""
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

    def _get_obs(self):
        """Get current observation as one-hot encoded sequence"""
        obs = np.zeros(len(self.alphabet) * self.forced_stop_len, dtype=np.float32)
        for i, char in enumerate(self.state):
            if i < self.forced_stop_len and char in self.alphabet:
                char_idx = self.alphabet.index(char)
                obs[i * len(self.alphabet) + char_idx] = 1.0
        return obs

    def get_state(self, obs):
        """Convert one or multiple one-hot encoded observations back to sequences."""
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

                # Check if a character exists at this position.
                # Using > 0.5 is slightly more robust for float arrays than a strict == 1.
                if np.sum(char_onehot) > 0.5:
                    char_idx = np.argmax(char_onehot)
                    seq += self.alphabet[char_idx]
                else:
                    # If we find a position with no character (all zeros), the sequence has ended.
                    break
            sequences.append(seq)
        
        # If the original input was a single observation, return a single string.
        # Otherwise, return the list of strings.
        return sequences[0] if obs.ndim == 1 else sequences
    
    def _compute_reward(self):
        """Compute reward for current sequence"""
        if len(self.state) == 0:
            return 0
        
        r = self.oracle.get_fitness([self.state]).item()
        r = np.maximum(r, self.SCALE_REWARD_MIN)
        r = r ** self.REWARD_EXP
        r = r * self.scale + 1e-12
        return r

    def real_reward(self):
        """Get unscaled reward for current sequence"""
        if len(self.state) == 0:
            return 0
        
        return np.maximum(self.oracle.get_fitness([self.state]).item(), self.SCALE_REWARD_MIN)

    def is_mode(self, sequence=None):
        """Check if current sequence (or given sequence) is a mode"""
        seq = sequence if sequence is not None else self.state
        return seq in self.modes

    def get_forward_action_masks(self, obs):
        """Get mask for valid forward actions (handles 1D, 2D, and 3D inputs)"""
        action_dim = 2 * len(self.alphabet) + 1
        is_torch = isinstance(obs, torch.Tensor)
        
        # Ensure obs is a proper array/tensor and store its original dimensionality
        if not is_torch:
            obs = np.asarray(obs)
        original_ndim = obs.ndim
        
        # --- Flatten input to 2D to unify logic ---
        if original_ndim == 1:
            # Add a dummy batch dimension to the 1D observation
            obs_flat = obs.reshape(1, -1) if not is_torch else obs.unsqueeze(0)
        else:
            # For 2D or 3D, flatten all but the last dimension
            obs_flat = obs.reshape(-1, obs.shape[-1]) if not is_torch else obs.view(-1, obs.shape[-1])

        # --- Core masking logic on the flattened tensor ---
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
        
        # --- Conditionally reshape the mask back to match original input dimensionality ---
        if original_ndim == 3:  # Input was (batch, time, feats)
            output_shape = (obs.shape[0], obs.shape[1], action_dim)
            masks = masks_flat.reshape(output_shape)
        elif original_ndim == 2: # Input was (batch, feats)
            masks = masks_flat # Already the correct shape
        else: # Input was 1D (feats,)
            masks = masks_flat.squeeze(0) # Remove the dummy batch dimension
            
        return masks

    def get_backward_action_masks(self, obs):
        """Get mask for valid backward actions (handles 1D, 2D and 3D inputs)
        
        Returns masks with same shape as action space (2 * len(alphabet) + 1),
        where the last action (stop) is always False for backward.
        """
        char_action_dim = 2 * len(self.alphabet)
        full_action_dim = char_action_dim + 1  # Include stop action (always False)
        is_torch = isinstance(obs, torch.Tensor)
        
        # Ensure obs is a proper array/tensor and store its original dimensionality
        if not is_torch:
            obs = np.asarray(obs)
        original_ndim = obs.ndim
        original_shape = obs.shape
        
        # --- Flatten input to 2D to unify logic ---
        if original_ndim == 1:
            # Add a dummy batch dimension to the 1D observation
            obs_flat = obs.reshape(1, -1) if not is_torch else obs.unsqueeze(0)
        else:
            # For 2D or 3D, flatten all but the last dimension
            obs_flat = obs.reshape(-1, original_shape[-1]) if not is_torch else obs.view(-1, original_shape[-1])

        # --- Core masking logic on the flattened tensor ---
        if is_torch:
            reshaped = obs_flat.view(-1, self.forced_stop_len, len(self.alphabet))
            any_char = torch.sum(reshaped, dim=-1) > 0
            
            # Create mask with full action dim (including stop action)
            masks_flat = torch.zeros(obs_flat.shape[0], full_action_dim, device=obs.device, dtype=torch.bool)
            
            # Create indices for the FLATTENED dimension
            batch_indices = torch.arange(obs_flat.shape[0], device=obs.device)
            
            # Find first and last characters
            first_pos = torch.argmax(any_char.float(), dim=-1)
            last_pos = any_char.shape[-1] - 1 - torch.argmax(any_char.flip(dims=[-1]).float(), dim=-1)
            
            # Identify sequences that are not empty
            has_chars = torch.any(any_char, dim=-1)
            
            # Only operate on non-empty sequences to avoid index errors on empty states
            if torch.any(has_chars):
                valid_indices = batch_indices[has_chars]
                
                # Get the index of the first character for valid sequences
                first_char_idx = torch.argmax(reshaped[valid_indices, first_pos[valid_indices]].float(), dim=-1)
                
                # Get the index of the last character for valid sequences
                last_char_idx = torch.argmax(reshaped[valid_indices, last_pos[valid_indices]].float(), dim=-1)
                
                # Apply masks for removing the first character
                masks_flat[valid_indices, first_char_idx] = True
                
                # Apply masks for removing the last character
                masks_flat[valid_indices, last_char_idx + len(self.alphabet)] = True
            
            # Stop action (index -1) is always False - already initialized as zeros

        else: # Numpy version
            reshaped = obs_flat.reshape(-1, self.forced_stop_len, len(self.alphabet))
            any_char = np.sum(reshaped, axis=-1) > 0
            
            # Create mask with full action dim (including stop action)
            masks_flat = np.zeros((obs_flat.shape[0], full_action_dim), dtype=bool)

            for i in range(obs_flat.shape[0]):
                if np.any(any_char[i]):
                    char_positions = np.where(any_char[i])[0]
                    first_pos, last_pos = char_positions[0], char_positions[-1]
                    
                    first_char_idx = np.argmax(reshaped[i, first_pos])
                    masks_flat[i, first_char_idx] = True
                    
                    last_char_idx = np.argmax(reshaped[i, last_pos])
                    masks_flat[i, last_char_idx + len(self.alphabet)] = True
            
            # Stop action (index -1) is always False - already initialized as zeros
            
        # --- Conditionally reshape the mask back to match original input dimensionality ---
        if original_ndim == 3:  # Input was (batch, time, feats)
            output_shape = (original_shape[0], original_shape[1], full_action_dim)
            masks = masks_flat.view(output_shape) if is_torch else masks_flat.reshape(output_shape)
        elif original_ndim == 2:  # Input was (batch, feats)
            masks = masks_flat  # Already the correct shape
        else:  # Input was 1D (feats,)
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
        - Actions 0-3: Remove first character (U,C,G,A)
        - Actions 4-7: Remove last character (U,C,G,A)
        
        Args:
            obs: One-hot encoded observation
            action: Backward action to apply
            
        Returns:
            New observation after applying backward action
        """
        state = self.get_state(obs)
        
        if len(state) == 0:
            return obs
        
        if action < len(self.alphabet):
            # Remove first character
            new_state = state[1:]
        else:
            # Remove last character
            new_state = state[:-1]
        
        # Convert state back to observation
        new_obs = np.zeros(len(self.alphabet) * self.forced_stop_len, dtype=np.float32)
        for i, char in enumerate(new_state):
            if i < self.forced_stop_len and char in self.alphabet:
                char_idx = self.alphabet.index(char)
                new_obs[i * len(self.alphabet) + char_idx] = 1.0
        
        return new_obs

    def render(self):
        """Render current state"""
        print(f"Current RNA sequence: '{self.state}' (length: {len(self.state)})")

    def unnormalize(self, r):
        """Unnormalize reward"""
        r = r / self.scale
        r = r ** (1 / self.REWARD_EXP)
        return r

    def dist_func(self, seq1, seq2):
        """Distance function between sequences"""
        return levenshtein(seq1, seq2)

    def get_error(self, samples):
        """Get the % of high reward samples"""
        # return the unique number of modes
        unique_modes = set([s for s in samples if self.is_mode(s)])
        return len(unique_modes)

    def get_num_of_reachable_end_states(self, obs):
        """Get the number of reachable end states from the given observation
        
        From a current sequence, we can continue adding characters (at beginning or end)
        until we reach forced_stop_len. The number of reachable end states is 4^(remaining_length),
        where remaining_length = forced_stop_len - current_length.
        
        Args:
            obs: One-hot encoded observation (1D array or batched)
            
        Returns:
            Number of reachable terminal states
        """
        # Handle batched observations
        if len(obs.shape) > 1:
            # Handle single observation from batch
            obs = obs.flatten() if obs.shape[0] == 1 else obs[0]
        
        # Get the current state (sequence) from observation
        state = self.get_state(obs)
        
        # Calculate remaining length
        current_length = len(state)
        remaining_length = self.forced_stop_len - current_length
        
        # If already at max length, only 1 reachable end state (itself)
        if remaining_length <= 0:
            return 1
        
        # Number of reachable end states = 4^(remaining_length)
        # where 4 is the alphabet size (U, C, G, A)
        return len(self.alphabet) ** remaining_length



if __name__ == "__main__":
    import itertools

    env = Environment()

    allpreds_path = os.path.join(
        _DATA_DIR,
        f"L{env.rna_length}_RNA{env.rna_task}_allpreds.pkl"
    )

    # -------------------------------------------------------------------------
    # Load or compute rewards for every possible sequence
    # -------------------------------------------------------------------------
    if os.path.exists(allpreds_path):
        print(f"Loading precomputed predictions from {allpreds_path} ...")
        with open(allpreds_path, "rb") as f:
            oracle_d = pickle.load(f)
        print(f"  Loaded {len(oracle_d):,} sequences.")
    else:
        total_seqs = len(env.alphabet) ** env.rna_length
        print(
            f"allpreds.pkl not found. Computing oracle scores for all "
            f"{total_seqs:,} length-{env.rna_length} sequences "
            f"(alphabet size {len(env.alphabet)}) ..."
        )
        print("  This may take a very long time — consider generating "
              "allpreds.pkl as a separate preprocessing step.")
        batch_size = 512
        oracle_d = {}
        seq_gen = itertools.product(env.alphabet, repeat=env.rna_length)
        batch = []
        for seq_tuple in tqdm(seq_gen, total=total_seqs, desc="Scoring"):
            batch.append("".join(seq_tuple))
            if len(batch) == batch_size:
                scores = env.oracle.get_fitness(batch)
                for seq, score in zip(batch, scores.tolist()):
                    oracle_d[seq] = score
                batch = []
        if batch:
            scores = env.oracle.get_fitness(batch)
            for seq, score in zip(batch, scores.tolist()):
                oracle_d[seq] = score
        print(f"  Scored {len(oracle_d):,} sequences. Saving to {allpreds_path} ...")
        with open(allpreds_path, "wb") as f:
            pickle.dump(oracle_d, f)

    # -------------------------------------------------------------------------
    # Top-k reward concentration analysis
    # -------------------------------------------------------------------------
    top_k = 10_000
    beta   = env.REWARD_EXP

    raw_rewards = np.array(list(oracle_d.values()), dtype=np.float64)
    raw_rewards = np.maximum(raw_rewards, env.SCALE_REWARD_MIN)

    # Partition function (unnormalised, pre-scaling) and expected reward
    log_raw  = np.log(raw_rewards)
    log_r_beta = log_raw * beta
    logZ     = torch.logsumexp(torch.from_numpy(log_r_beta), 0).item()

    # Sort sequences by raw reward (descending) and pick the top k
    sorted_indices = np.argsort(raw_rewards)[::-1]
    top_k_indices  = sorted_indices[:top_k]
    top_k_rewards  = raw_rewards[top_k_indices]

    # Proportion of the partition function captured by the top-k states
    log_r_beta_topk = log_raw[top_k_indices] * beta
    logZ_topk  = torch.logsumexp(torch.from_numpy(log_r_beta_topk), 0).item()
    pct_of_Z   = np.exp(logZ_topk - logZ) * 100.0

    # Proportion of total raw reward captured by the top-k states
    total_raw_reward = raw_rewards.sum()
    topk_raw_reward  = top_k_rewards.sum()
    pct_raw          = (topk_raw_reward / total_raw_reward) * 100.0

    print(f"\n{'='*60}")
    print(f"RNA{env.rna_task}  |  length {env.rna_length}  |  beta={beta}")
    print(f"{'='*60}")
    print(f"Total sequences          : {len(oracle_d):>15,}")
    print(f"Top-{top_k:,} raw reward range : "
          f"{top_k_rewards[-1]:.6f}  –  {top_k_rewards[0]:.6f}")
    print(f"Overall raw reward range : "
          f"{raw_rewards.min():.6f}  –  {raw_rewards.max():.6f}")
    print(f"Top-{top_k:,} share of Z (beta={beta}): {pct_of_Z:.4f} %")
    print(f"Top-{top_k:,} share of raw reward    : {pct_raw:.4f} %")