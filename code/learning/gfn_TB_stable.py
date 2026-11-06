import os
import sys
import time
import math
import pickle
import warnings
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from scipy import stats
import wandb
from torch.optim import Adam

from scipy.optimize import minimize_scalar

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from learning.gfn_base import GFlowNetBase
from learning.runtime_profiler import RuntimeProfilingMixin, profile_runtime


class TrajectoryBalanceStabilizedWithProbabilisticGuarantees(
    RuntimeProfilingMixin,
    GFlowNetBase,
):
    """
    Stable GFlowNets with TV Monitoring and Probabilistic Guarantees.
    
    This algorithm provides theoretical guarantees on the Total Variation (TV)
    distance between the learned distribution and the target distribution.
    It uses reference flows to stabilize training and includes a verification
    procedure to ensure performance guarantees.
    
    Two modes of operation:
    1. If env has `sample_terminal_states`: Uses full reward distribution sampling
    2. Otherwise: Uses Subgraph Certificate approach with discovered terminal states
    
    The Subgraph Certificate algorithm:
    - Maintains a set of discovered terminal states X_sub
    - Tracks cumulative reward mass and uses patience for performance guarantees
    - Samples backward trajectories from X_sub weighted by R(x)
    - Provides TV bounds certified over the discovered subgraph
    
    Attributes:
        target_TV (d): Target TV distance bound (default: 0.1)
        verification_budget (m): Number of samples for verification (default: 512)
        confidence_level (α): Confidence level for probabilistic guarantees (default: 0.95)
        patience (k_pat): Patience for subgraph certificate mode (default: 10)
        moving_average_factor (β): Moving average factor for loss threshold (default: 0.05)
    """
    
    def __init__(self, env, target_TV=0.1, 
                 verification_budget=512, confidence_level=0.95, 
                 patience=10, moving_average_factor=0.05, stabilizing_factor='max',
                 forward_traj=16, backward_traj=16, **kwargs):
        super().__init__(env, **kwargs)

        self.target_TV = target_TV  # d in the algorithm
        self.verification_budget = verification_budget  # m in the algorithm
        self.confidence_level = confidence_level  # α in the algorithm
        self.patience = patience  # k_pat in the algorithm
        self.moving_average_factor = moving_average_factor
        # 'null' disables the reference flow entirely: log_reference_flow
        # stays at -inf so the augmented loss reduces to the plain TB loss.
        valid_stabilizers = {'max', 'mean', 'median', 'null'}
        if stabilizing_factor not in valid_stabilizers:
            raise ValueError(
                f"stabilizing_factor must be one of {sorted(valid_stabilizers)}, "
                f"got {stabilizing_factor!r}"
            )
        self.stabilizing_factor = stabilizing_factor

        # Per-source trajectory budgets per training iteration. Their sum is
        # the effective batch_size / train_freq (set by the caller, e.g.
        # train.py); we sanity-check that here so misconfiguration is caught
        # early rather than producing silently mis-shaped batches.
        self.forward_traj = int(forward_traj)
        self.backward_traj = int(backward_traj)
        total = self.forward_traj + self.backward_traj
        if total != self.batch_size:
            raise ValueError(
                f"forward_traj({self.forward_traj}) + backward_traj({self.backward_traj})"
                f" = {total} must equal batch_size ({self.batch_size})"
            )
        if total != self.train_freq:
            raise ValueError(
                f"forward_traj + backward_traj = {total} must equal train_freq ({self.train_freq})"
            )

        # loss threshold and log_loss_divisor are used for log reference flow computation
        self.loss_threshold = 10.0
        self.log_loss_divisor = None

        # Initialize TV bounds
        self.tv_bound = 1.0
        self.loss_only_TV_bound = 1.0
        
        # Accumulator for verification with larger sample sizes
        # Store accumulated log_ratios (log(Z*P_F / R*P_B)) for recomputing bounds
        self.accumulated_log_ZPF = None  # Will be a numpy array when accumulating
        self.accumulated_log_RPB = None  # Will be a numpy array when accumulating
        self.accumulated_fwd_samples = 0
        self.accumulated_bwd_samples = 0
        
        # Track convergence (reach target TV bound)
        self.convergence_iteration = None
        
        # Patience counter for stability - only validate after stability_satisfied for patience consecutive iterations
        self.patience_counter = 0
        
        # Add tv_bound to logger for stable GFN
        self.logger['loss_only_TV_bound'] = []
        self.logger['tv_bound'] = []
        self.logger['optimal_c'] = []
        self.logger['loss_threshold'] = []
        self.logger['stability_patience'] = []
        # Augmented loss statistics (separate from original TB loss)
        self.logger['augmented_loss_means'] = []
        self.logger['augmented_loss_max'] = []
        self.logger['augmented_loss_std'] = []

    def _sample_from_terminal_states(self, n_samples):
        """
        Sample terminal states from X_sub weighted by R(x).
        
        Returns:
            List of (obs, reward) tuples
        """
        if len(self.terminal_states) == 0:
            return []
        
        states = list(self.terminal_states.values())
        rewards = np.array([s[1] for s in states])
        
        # Normalize to get probabilities
        probs = rewards / rewards.sum()
        
        # Sample indices
        indices = np.random.choice(len(states), size=min(n_samples, len(states)), 
                                   p=probs, replace=True)
        
        return [states[i] for i in indices]

    @profile_runtime("reference_flow")
    def compute_reference_flow_for_memory(self, logPF, logPB, logZ, log_reward):
        log_A = logZ + logPF
        log_B = log_reward + logPB
        
        # Calculate log( |A - B| ) strictly in log-space
        condition = log_A > log_B
        log_large = torch.where(condition, log_A, log_B)
        log_small = torch.where(condition, log_B, log_A)
        
        log_diff = log_small - log_large # Strictly negative
        
        # Clamp to avoid log(0)
        log_diff = torch.clamp(log_diff, max=-1e-20) 
        
        # Result is log( |A - B| )
        log_priority = log_large + torch.log(-torch.expm1(log_diff))
        
        return log_priority.detach().cpu().numpy().clip(max=20)

    
    @profile_runtime("reference_flow")
    def compute_reference_flow(self, c, log_loss_divisor, logPF, logPB, logZ, log_reward):
        """
        Compute the log reference flow log(δ(τ)) for each trajectory.
        
        The reference flow is defined to ensure the stability condition can be checked.
        δ(τ) = max(0, Z * P_F(τ) - e^c * R(x) * P_B(τ|x)) / (e^c - 1)  if loss > c
        δ(τ) = max(0, -e^c * Z * P_F(τ) + R(x) * P_B(τ|x)) / (e^c - 1)  if loss < -c
        δ(τ) = 0  otherwise
        
        Args:
            logPF: Log forward probabilities [batch]
            logPB: Log backward probabilities [batch]
            logZ: Log partition function (scalar)
            log_reward: Log rewards [batch]
            
        Returns:
            log_reference_flow: Log reference flow for each trajectory [batch]
            Note: Returns -inf for trajectories where reference_flow = 0
        """
        with torch.no_grad():
            
            # Compute the TB loss (before squaring)
            tb_loss = logPF + logZ - logPB - log_reward
            
            # Initialize log reference flow to -inf (corresponds to reference_flow = 0)
            log_reference_flow = torch.full_like(logPF, float('-inf'))

            # Case 1: loss > c (forward flow too high)
            # Implies: log_Z_PF > log_c_R_PB
            high_loss_mask = tb_loss > c
            if high_loss_mask.any():
                # A = log_Z_PF (Larger), B = log_c_R_PB (Smaller)
                log_A = logZ + logPF[high_loss_mask]
                log_B = c + log_reward[high_loss_mask] + logPB[high_loss_mask]
                
                # Compute log(e^A - e^B) in log-space
                # log_numerator = A + log(1 - exp(B-A))
                # Using expm1 for stability: 1 - exp(x) = -expm1(x)
                log_diff = log_B - log_A
                log_numerator = log_A + torch.log(-torch.expm1(log_diff))
                
                # Final calc: log_num - log_div (keep in log space)
                log_reference_flow[high_loss_mask] = log_numerator - log_loss_divisor

            # # Case 2: loss < -c (backward flow too high)
            # # Implies: log_R_PB > log_c_Z_PF
            low_loss_mask = tb_loss < -c
            if low_loss_mask.any():
                # A = log_R_PB (Larger), B = log_c_Z_PF (Smaller)
                log_A = log_reward[low_loss_mask] + logPB[low_loss_mask]
                log_B = c + logZ + logPF[low_loss_mask]
                
                # Compute log(e^A - e^B) in log-space
                log_diff = log_B - log_A
                log_numerator = log_A + torch.log(-torch.expm1(log_diff))
                
                # Final calc: log_num - log_div (keep in log space)
                log_reference_flow[low_loss_mask] = log_numerator - log_loss_divisor
        
        return log_reference_flow.clip(max=20)
    
    def compute_loss_only_TV_bound(self, log_ratios, num_samples_fwd, num_samples_bwd):
        """
        Compute TV bound using only the loss threshold (simple bound).
        
        If L_TB < c^2, then TV <= e^{2c} - 1 + 2*log(1/α)/n
        where c is the maximum absolute value of log_ratios.
        
        Args:
            log_ratios: numpy array of log(r_i) = log(Z * P_F / (R * P_B)) values
            num_samples: Number of samples for statistical error term
            
        Returns:
            tv_bound: Upper bound on TV distance
        """
        # Compute c as the maximum absolute log ratio
        c = float(np.abs(log_ratios).max())
        if c <= 0.3466: # c <= ln(2)/2
            alpha = (1 - self.confidence_level) / 2
            tv_bound = np.exp(2 * c) - 1 + np.log(1.0 / alpha) / num_samples_fwd + np.log(1.0 / alpha) / num_samples_bwd if num_samples_fwd > 0 and num_samples_bwd > 0 else float('inf')
            return min(tv_bound, 1.0)
        else:
            return 1.0

    
    def compute_TV_bound_from_log_ratios(self, log_ZPF, log_RPB, num_samples_fwd, num_samples_bwd):
        """
        Compute TV bound from pre-computed log values (for accumulated samples).
        
        The key constraint is: M < 1/(exp(c) - 1), i.e., K = (exp(c)-1)*M < 1
        
        From the constraint, we derive explicit c_min for Case 1 (log_r > c):
            c > log(exp(log_r) - 1) = log(expm1(log_r))
        
        This allows choosing c much lower than max(|log_r|) when R is small and ZPF is large,
        resulting in tighter TV bounds.
        
        Args:
            log_ZPF: numpy array of log(Z * P_F(τ)) values
            log_RPB: numpy array of log(R(τ) * P_B(τ)) values
            num_samples_fwd: Number of samples drawn from the student forward policy (n).
            num_samples_bwd: Number of reward-target/backward samples (m).
                Teacher samples may enlarge the empirical maximum but not m or n.
            
        Returns:
            tuple: (optimal_c, bound_without_error, bound_with_error)
        """
        alpha = (1 - self.confidence_level) / 2
        # Theorem 3.11 / Eq. (16): c is optimized over the same samples,
        # so the two-sided tolerance correction has both factors of 2.
        # m counts reward-target/backward samples; n counts student-PF samples.
        # Teacher trajectories never increase either confidence denominator.
        error_term = (
            2 * np.log(2.0 / alpha) / num_samples_bwd
            + 2 * np.log(2.0 / alpha) / num_samples_fwd
            if num_samples_fwd > 0 and num_samples_bwd > 0 else float('inf')
        )
        
        # log_ratio = log(ZP_F / RP_B). Float64 is important near zero.
        log_ratios = np.asarray(log_ZPF, dtype=np.float64) - np.asarray(log_RPB, dtype=np.float64)
        if log_ratios.size == 0 or not np.all(np.isfinite(log_ratios)):
            return 0.0, 1.0, 1.0 + error_term

        c_upper = float(np.max(np.abs(log_ratios)))
        if c_upper == 0.0:
            # All ratios equal one. The infimum over c > 0 is zero as c -> 0+.
            return 0.0, 0.0, error_term

        largest_log_ratio = float(np.max(log_ratios))
        smallest_log_ratio = float(np.min(log_ratios))
        c_lower = max(np.nextafter(0.0, 1.0), min(1e-9, c_upper * 1e-6))
        if largest_log_ratio > np.log(2.0):
            # Only the positive-ratio tail imposes a lower feasibility bound:
            # K = exp(log_r) - exp(c) < 1 implies c > log(expm1(log_r)).
            feasible_boundary = largest_log_ratio + np.log1p(-np.exp(-largest_log_ratio))
            c_lower = max(c_lower, np.nextafter(feasible_boundary, np.inf))

        def calculate_bound(c):
            """The paper's empirical term with direct checking of K < 1."""
            if c <= 0:
                return np.inf
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                exp_c = np.exp(c)
                # K = (exp(c)-1)*M. Compute the product directly instead of
                # dividing by exp(c)-1 and multiplying again near c = 0.
                # Maxima in the two tails depend only on the extreme ratios.
                positive_tail = (
                    exp_c * np.expm1(largest_log_ratio - c)
                    if largest_log_ratio > c else 0.0
                )
                negative_tail = (
                    -np.expm1(c + smallest_log_ratio)
                    if smallest_log_ratio < -c else 0.0
                )
                k_value = max(positive_tail, negative_tail)
                if not np.isfinite(k_value) or k_value >= 1.0:
                    return np.inf
                # Equivalent to exp(c)*(exp(c)+K)/(1-K)-1, with less
                # cancellation when both c and K are small.
                bound = (np.expm1(2 * c) + k_value * (exp_c + 1)) / (1 - k_value)
            return float(bound) if np.isfinite(bound) and bound >= 0 else np.inf

        # Include endpoints explicitly: bounded minimization alone misses the
        # c -> 0+ limit (negative ratios) and the zero-reference-flow endpoint.
        candidates = [(c_upper, calculate_bound(c_upper))]
        if c_lower <= c_upper:
            candidates.append((c_lower, calculate_bound(c_lower)))
        if c_lower < c_upper:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                result = minimize_scalar(
                    calculate_bound, bounds=(c_lower, c_upper), method='bounded',
                    options={'xatol': 1e-12, 'maxiter': 500},
                )
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.x), float(result.fun)))
        optimal_c, optimal_bound = min(candidates, key=lambda item: item[1])
        if not np.isfinite(optimal_bound):
            # Extreme ratios can overflow even float64. TV <= 1 is always valid.
            return 0.0, 1.0, 1.0 + error_term
        return optimal_c, optimal_bound, optimal_bound + error_term
    

    def _collect_rollouts_impl(self):
        """
        Collect ``forward_traj`` forward trajectories and ``backward_traj``
        backward trajectories per iteration. Their sum equals ``batch_size``
        and ``train_freq`` (enforced in __init__).

        Forward: Sample from initial state using forward policy
        Backward: Sample terminal states proportional to R(x), then use backward policy

        Uses parallel collection if data_env is available, otherwise sequential.
        """
        forward_count = self.forward_traj
        backward_count = self.backward_traj

        buffer_lens = []
        buffer_rews_report = []
        num_collected_steps, num_collected_episodes = 0, 0
        
        with torch.no_grad():
            # Collect forward trajectories
            if self.data_env is not None:
                forward_steps, forward_eps, forward_lens, forward_rews = self.collect_rollouts_parallel(
                    predict_fn=self.predict, num_trajectories=forward_count
                )
            else:
                forward_steps, forward_eps, forward_lens, forward_rews = self.collect_rollouts_sequential(
                    predict_fn=self.predict, num_trajectories=forward_count
                )
            
            num_collected_steps += forward_steps
            num_collected_episodes += forward_eps
            buffer_lens.extend(forward_lens)
            buffer_rews_report.extend(forward_rews)
            
            # Collect backward trajectories
            if self.has_terminal_sampling:
                # Sample terminal states from environment
                terminal_states, rewards = self.env.unwrapped.sample_terminal_states(backward_count)
                terminal_states = np.array(terminal_states)
                rewards = np.array(rewards)
            else:
                # Sample from discovered terminal states (subgraph mode)
                sampled_states = self._sample_from_terminal_states(backward_count)
                if len(sampled_states) > 0:
                    terminal_states = np.array([s[0] for s in sampled_states])
                    rewards = np.array([s[1] for s in sampled_states])
                else:
                    terminal_states = np.array([])
                    rewards = np.array([])

            if len(terminal_states) > 0:
                backward_steps, backward_eps, backward_lens, backward_rews = self.collect_backward_rollouts_parallel(
                    terminal_states, rewards
                )
                num_collected_steps += backward_steps
                num_collected_episodes += backward_eps
                buffer_lens.extend(backward_lens)
                buffer_rews_report.extend(backward_rews)
        
        # For subgraph certificate mode, check if mass improved
        if not self.has_terminal_sampling:
            if self.current_mass > self.best_mass:
                self.best_mass = self.current_mass
                self.patience_counter = 0  # Reset patience on improvement
            else:
                self.patience_counter += 1
        
        self.logger['buffer_lens'] = buffer_lens
        self.logger['buffer_rews'] = buffer_rews_report
        
        return num_collected_steps, num_collected_episodes
    
    def _train_impl(self):
        """
        Train using the Stable GFlowNet algorithm with probabilistic guarantees.
        
        Uses subgraph certificate approach when sample_terminal_states is not available.
        
        The verification is integrated into the training loop:
        1. Verify using recent samples from memory
        2. If stability is satisfied, accumulate samples until TV bound is achieved
        3. If TV bound is achieved (< target_TV) and no new mass discovered, skip training
        4. Otherwise, perform gradient updates
        """
        # Step 1: Verification using latest samples (not overlapping with training batch)
        batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
            self.memory.latest_sample(self.train_freq, 0)
        
        batch_obs_pad = pad_sequence(batch_obs, batch_first=True)
        batch_acts_pad = pad_sequence(batch_acts, batch_first=True)
        batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first=True)
        
        lengths = torch.tensor([len(obs) for obs in batch_obs], device=self.device, dtype=torch.long)
        
        # Forward pass for verification
        forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
        backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
        
        
        logPF = self.forward_policy.evaluate_actions(
        batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
        )
        logPB = self.backward_policy.evaluate_actions(
        batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
        )
        
        logPF = torch.sum(logPF, dim=1)
        logPB = torch.sum(logPB, dim=1)
        
        # Extract scalar rewards from list of tensors (optimized)
        if isinstance(batch_rews[0], torch.Tensor):
            batch_rews_tensor = torch.stack([r.squeeze() for r in batch_rews]).to(self.device)
        else:
            batch_rews_tensor = torch.tensor(batch_rews, device=self.device, dtype=torch.float32)
        original_log_reward = torch.log(batch_rews_tensor)
        log_reward = original_log_reward / self.reward_temperature
        
        if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
            raise ValueError("Infinite logprobs found in verification")
        
        logZ = self.get_logZ()
        
        with torch.no_grad():
            if (self.patience_counter >= self.patience or self.has_terminal_sampling):
                # Calculate TV bounds
                # Decompose into log_ZPF and log_RPB for the reference flow bound computation
                current_log_ZPF = (logZ + logPF).detach().cpu().numpy()
                current_log_RPB = (original_log_reward + logPB).detach().cpu().numpy()
                current_log_ratios = current_log_ZPF - current_log_RPB

                if self.accumulated_log_ZPF is None:
                    self.accumulated_log_ZPF = current_log_ZPF
                    self.accumulated_log_RPB = current_log_RPB
                else:
                    self.accumulated_log_ZPF = np.concatenate([
                        self.accumulated_log_ZPF, current_log_ZPF
                    ])
                    self.accumulated_log_RPB = np.concatenate([
                        self.accumulated_log_RPB, current_log_RPB
                    ])
                self.accumulated_fwd_samples += self.forward_traj
                self.accumulated_bwd_samples += self.backward_traj
                accumulated_log_ratios = self.accumulated_log_ZPF - self.accumulated_log_RPB

                # 1. Simple bound using only the loss threshold
                self.loss_only_TV_bound = self.compute_loss_only_TV_bound(accumulated_log_ratios, self.accumulated_fwd_samples, self.accumulated_bwd_samples)
                
                # 2. Tighter bound using reference flow optimization with accumulated samples
                # Compute bound using accumulated samples (decomposed for tighter c_min)
                optimal_c, ref_flow_tv_without_error_term, ref_flow_tv_bound = self.compute_TV_bound_from_log_ratios(
                    self.accumulated_log_ZPF, self.accumulated_log_RPB,
                    self.accumulated_fwd_samples, self.accumulated_bwd_samples
                )
                self.tv_bound = min(ref_flow_tv_bound, 1.0)
                
                # Log optimal loss threshold
                self.logger['optimal_c'].append(optimal_c)

                if ref_flow_tv_without_error_term < self.target_TV:
                    if ref_flow_tv_bound < self.target_TV: #and self.accumulated_num_samples >= self.verification_budget:
                        # TV bound achieved - can skip training if converged
                        should_update = False
                        self.converged = True
                        self.convergence_iteration = self.logger['iteration']
                        # Reset accumulator since we've converged
                        self.accumulated_log_ZPF = None
                        self.accumulated_log_RPB = None
                        self.accumulated_fwd_samples = 0
                        self.accumulated_bwd_samples = 0
                    else:
                        # Close to convergence but need more samples to reduce error term
                        # Keep accumulating - don't reset, don't update
                        should_update = False

                else:
                    # TV bound not achieved - must continue training
                    should_update = True
                    # Reset accumulator since we're not close to convergence
                    self.accumulated_log_ZPF = None
                    self.accumulated_log_RPB = None
                    self.accumulated_fwd_samples = 0
                    self.accumulated_bwd_samples = 0
            else:
                should_update = True
        
        # Step 2: Training (if needed)
        if should_update: 
            original_loss = (logPF + logZ - logPB - log_reward).pow(2)
            # Log original TB loss statistics
            self.logger['loss_means'].append(original_loss.mean().detach().item())
            self.logger['loss_max'].append(original_loss.max().detach().item())
            self.logger['loss_std'].append(original_loss.std().detach().item())

            if self.stabilizing_factor == 'null':
                # No reference flow -- augmented loss == original TB loss.
                log_reference_flow = torch.full_like(logPF, float('-inf'))
            else:
                if self.stabilizing_factor == 'max':
                    new_threshold = original_loss.max().detach().item()
                elif self.stabilizing_factor == 'mean':
                    new_threshold = original_loss.mean().detach().item()
                else:  # 'median'
                    new_threshold = original_loss.median().detach().item()

                self.loss_threshold = min(20, np.sqrt(new_threshold) * self.moving_average_factor + self.loss_threshold * (1 - self.moving_average_factor))
                self.log_loss_divisor = torch.log(torch.exp(torch.tensor(self.loss_threshold, device=self.device)) - 1.0)
                log_reference_flow = self.compute_reference_flow(
                        self.loss_threshold, self.log_loss_divisor, logPF, logPB, logZ, log_reward
                    )
                
            # Compute augmented loss for optimization
            log_Z_PF = logZ + logPF
            log_R_PB = log_reward + logPB
            
            log_augmented_forward = torch.logaddexp(log_reference_flow, log_Z_PF)
            log_augmented_backward = torch.logaddexp(log_reference_flow, log_R_PB)
            
            augmented_loss = (log_augmented_forward - log_augmented_backward).pow(2)
            
            if torch.any(torch.isinf(augmented_loss)) or torch.any(torch.isnan(augmented_loss)):
                raise ValueError(f"Invalid augmented loss found, loss: {augmented_loss}")
            
            # Log augmented loss statistics
            self.logger['augmented_loss_means'].append(augmented_loss.mean().detach().item())
            self.logger['augmented_loss_max'].append(augmented_loss.max().detach().item())
            self.logger['augmented_loss_std'].append(augmented_loss.std().detach().item())
            
            self.logger['log_Z'].append(logZ.item())
            
            self.logger['tv_bound'].append(self.tv_bound)
            self.logger['loss_only_TV_bound'].append(self.loss_only_TV_bound)
            # Log loss_threshold (append current value if not already logged this iteration)
            if len(self.logger['loss_threshold']) == 0 or self.logger['loss_threshold'][-1] != self.loss_threshold:
                self.logger['loss_threshold'].append(self.loss_threshold)
            self.logger['stability_patience'].append(self.patience_counter)

            
            # Optimize
            loss = augmented_loss.mean()
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            if self.backward_optim is not None:
                torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            if self.logZ is not None:
                torch.nn.utils.clip_grad_norm_([self.logZ], 1e1)
            
            for optim in self.GFN_optim:
                optim.step()
            
        for _ in range(self.gradient_steps - 1):
            # In on-policy mode with single gradient step, reuse padded tensors from verification
            batch_obs, batch_acts, batch_next_obs, batch_rews, batch_idx = \
                    self.sample_from_memory(self.batch_size)
                
            batch_obs_pad = pad_sequence(batch_obs, batch_first=True)
            batch_acts_pad = pad_sequence(batch_acts, batch_first=True)
            batch_next_obs_pad = pad_sequence(batch_next_obs, batch_first=True)
                
            lengths = torch.tensor([len(obs) for obs in batch_obs], device=self.device, dtype=torch.long)
                
            forward_mask = self.env.unwrapped.get_forward_action_masks(batch_obs_pad)
            backward_mask = self.env.unwrapped.get_backward_action_masks(batch_next_obs_pad)
                
            batch_rews_train = torch.tensor(batch_rews, device=self.device)
            original_log_reward = torch.log(batch_rews_train) 
            log_reward = original_log_reward / self.reward_temperature
            
            # Forward pass (always needed with gradients for training)
            logPF = self.forward_policy.evaluate_actions(
                batch_obs_pad, batch_acts_pad, forward_mask, lengths=lengths
            )
            logPB = self.backward_policy.evaluate_actions(
                batch_next_obs_pad, batch_acts_pad, backward_mask, lengths=lengths
            )
            
            logPF = torch.sum(logPF, dim=1)
            logPB = torch.sum(logPB, dim=1)
            
            if torch.any(torch.isinf(logPF)) or torch.any(torch.isinf(logPB)):
                raise ValueError("Infinite logprobs found")
            
            logZ = self.get_logZ()
            
            # Compute original TB loss for logging
            original_loss = (logPF + logZ - logPB - log_reward).pow(2)

            # Log original TB loss statistics
            self.logger['loss_means'].append(original_loss.mean().detach().item())
            self.logger['loss_max'].append(original_loss.max().detach().item())
            self.logger['loss_std'].append(original_loss.std().detach().item())

            if self.stabilizing_factor == 'null':
                # No reference flow -- augmented loss == original TB loss.
                log_reference_flow = torch.full_like(logPF, float('-inf'))
            else:
                self.loss_threshold = min(20, np.sqrt(original_loss.max().detach().item()) * self.moving_average_factor + self.loss_threshold * (1 - self.moving_average_factor))
                self.log_loss_divisor = torch.log(torch.exp(torch.tensor(self.loss_threshold, device=self.device)) - 1.0)
                # Compute log reference flow for training
                log_reference_flow = self.compute_reference_flow(
                    self.loss_threshold, self.log_loss_divisor, logPF, logPB, logZ, log_reward
                )
            
            # Compute augmented loss for optimization
            log_Z_PF = logZ + logPF
            log_R_PB = log_reward + logPB
            
            log_augmented_forward = torch.logaddexp(log_reference_flow, log_Z_PF)
            log_augmented_backward = torch.logaddexp(log_reference_flow, log_R_PB)
            
            augmented_loss = (log_augmented_forward - log_augmented_backward).pow(2)
            
            if torch.any(torch.isinf(augmented_loss)) or torch.any(torch.isnan(augmented_loss)):
                raise ValueError(f"Invalid augmented loss found, loss: {augmented_loss}")
            
            # Log augmented loss statistics
            self.logger['augmented_loss_means'].append(augmented_loss.mean().detach().item())
            self.logger['augmented_loss_max'].append(augmented_loss.max().detach().item())
            self.logger['augmented_loss_std'].append(augmented_loss.std().detach().item())
            
            self.logger['log_Z'].append(logZ.item())
            
            self.logger['tv_bound'].append(self.tv_bound)
            self.logger['loss_only_TV_bound'].append(self.loss_only_TV_bound)
            # Log loss_threshold (append current value if not already logged this iteration)
            if len(self.logger['loss_threshold']) == 0 or self.logger['loss_threshold'][-1] != self.loss_threshold:
                self.logger['loss_threshold'].append(self.loss_threshold)
            self.logger['stability_patience'].append(self.patience_counter)

            
            # Optimize
            loss = augmented_loss.mean()
            for optim in self.GFN_optim:
                optim.zero_grad()
            
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            if self.backward_optim is not None:
                torch.nn.utils.clip_grad_norm_(self.backward_policy.parameters(), 1e1)
            if self.logZ is not None:
                torch.nn.utils.clip_grad_norm_([self.logZ], 1e1)
            
            for optim in self.GFN_optim:
                optim.step()
            
            self._n_updates += self.gradient_steps
        else:
            # Not updating - log zeros but still track state
            logZ = self.get_logZ()
            self.logger['loss_means'].append(0.0)
            self.logger['loss_max'].append(0.0)
            self.logger['loss_std'].append(0.0)
            self.logger['augmented_loss_means'].append(0.0)
            self.logger['augmented_loss_max'].append(0.0)
            self.logger['augmented_loss_std'].append(0.0)
            self.logger['log_Z'].append(logZ.item())
            self.logger['tv_bound'].append(self.tv_bound)
            self.logger['loss_only_TV_bound'].append(self.loss_only_TV_bound)
            self.logger['loss_threshold'].append(self.loss_threshold)
            self.logger['stability_patience'].append(self.patience_counter)
            # Log tmp_M (keep current value, already logged in verification step)
            # tmp_M is already logged above, so we don't need to log it again here
    
    def _log_summary(self):
        """
        Override base class logging to include TV bound, stability patience, and augmented loss.
        """
        # Call parent logging first
        super()._log_summary()
        
        # Add TV bound, stability patience, and augmented loss logging
        if len(self.logger['tv_bound']) > 0 or len(self.logger['augmented_loss_means']) > 0:
            current_tv_bound = self.logger['tv_bound'][-1] if len(self.logger['tv_bound']) > 0 else self.tv_bound
            current_loss_only_TV_bound = self.logger['loss_only_TV_bound'][-1] if len(self.logger['loss_only_TV_bound']) > 0 else getattr(self, 'loss_only_TV_bound', 1.0)
            current_patience = self.logger['stability_patience'][-1] if len(self.logger['stability_patience']) > 0 else self.patience_counter
            current_optimal_c = self.logger['optimal_c'][-1] if len(self.logger['optimal_c']) > 0 else self.loss_threshold
            current_loss_threshold = self.logger['loss_threshold'][-1] if len(self.logger['loss_threshold']) > 0 else self.loss_threshold

            # Augmented loss statistics
            aug_loss_mean = np.mean(self.logger['augmented_loss_means']) if len(self.logger['augmented_loss_means']) > 0 else 0.0
            aug_loss_max = np.mean(self.logger['augmented_loss_max']) if len(self.logger['augmented_loss_max']) > 0 else 0.0
            aug_loss_std = np.mean(self.logger['augmented_loss_std']) if len(self.logger['augmented_loss_std']) > 0 else 0.0
            
            if self.verbose:
                print(f"Augmented Loss Mean: {aug_loss_mean:.5f}", flush=True)
                if current_tv_bound < float('inf'):
                    print(f"TV Bound: {current_tv_bound:.5f}", flush=True)
                else:
                    print(f"TV Bound: inf", flush=True)
                if current_loss_only_TV_bound < float('inf'):
                    print(f"Loss-only TV Bound: {current_loss_only_TV_bound:.5f}", flush=True)
                print(f"Optimal c: {current_optimal_c:.5f}", flush=True)
                print(f"Loss threshold (c): {current_loss_threshold:.5f}", flush=True)
                print(f"Accumulated samples: {self.accumulated_fwd_samples + self.accumulated_bwd_samples}", flush=True)
                print(f"Stability Patience: {current_patience}/{self.patience}", flush=True)
            
            if self.use_wandb:
                iteration = self.logger['iteration']
                log_dict = {
                    'train/augmented_loss_mean': aug_loss_mean,
                    'train/augmented_loss_max': aug_loss_max,
                    'train/augmented_loss_std': aug_loss_std,
                    'train/tv_bound': current_tv_bound,
                    'train/loss_only_TV_bound': current_loss_only_TV_bound,
                    'train/optimal_c': current_optimal_c,
                    'train/loss_threshold': current_loss_threshold,
                    'train/accumulated_samples': self.accumulated_fwd_samples + self.accumulated_bwd_samples,
                    'train/stability_patience': current_patience,
                }
                wandb.log(log_dict, step=iteration)
            
            # Reset logging
            self.logger['tv_bound'] = []
            self.logger['loss_only_TV_bound'] = []
            self.logger['optimal_c'] = []
            self.logger['loss_threshold'] = []
            self.logger['stability_patience'] = []
            self.logger['augmented_loss_means'] = []
            self.logger['augmented_loss_max'] = []
            self.logger['augmented_loss_std'] = []
            if not self.has_terminal_sampling:
                self.logger['current_mass'] = []
    
    def get_discovered_mass(self):
        """Get the cumulative reward mass of discovered terminal states (subgraph mode only)."""
        if self.has_terminal_sampling:
            return None
        return self.current_mass
    
    def get_num_discovered_states(self):
        """Get the number of discovered terminal states (subgraph mode only)."""
        if self.has_terminal_sampling:
            return None
        return len(self.terminal_states)
