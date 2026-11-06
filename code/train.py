"""
Training script for GFlowNet experiments.

Supports multiple learning algorithms and environments.
"""

import os
import sys
import argparse
import random
import json
import re

# This must be set before the first CUDA BLAS handle is created.  The Slurm
# launchers set it before Python starts; this fallback also covers direct
# ``python train.py`` invocations in the common case where CUDA has not yet
# been initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import wandb

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import environments
from envs.hypergrid import HyperGridEnv
from envs.regular_tree import RegularTreeEnv

# Import GFlowNet learning algorithms
from learning.gfn_TB import TrajectoryBalance
from learning.gfn_subTB import SubTrajectoryBalance
from learning.gfn_DB import DetailedBalance
from learning.gfn_DB_WDB import DetailedBalanceWeightedByReachableEndStates
from learning.gfn_FM import FlowMatching
from learning.gfn_TB_adaptive_teacher import TrajectoryBalanceAdaptiveTeacher
from learning.gfn_TB_stable import TrajectoryBalanceStabilizedWithProbabilisticGuarantees
from learning.gfn_TB_stable_teacher import TrajectoryBalanceStabilizedWithProbabilisticGuaranteesAndAdaptiveTeacher
from utils.mode_evaluation import (
    MODE_METRICS_FILENAME,
    evaluate_and_store_mode_metrics,
)
from utils.reproducibility import (
    REPRODUCIBILITY_FILENAME,
    collect_reproducibility_record,
    write_reproducibility_record,
)


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='GFlowNet Training Script')

    # Environment settings
    parser.add_argument('-s', '--sim', type=str, default='hypergrid-simple',
        help='Environment: hypergrid-simple, hypergrid-mild, hypergrid-hard, regular-tree, rna14, sEH, sEH_large')
    parser.add_argument('-se', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require deterministic PyTorch algorithms and deterministic math "
            "(default: enabled; use --no-deterministic to opt out)"
        ),
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="PyTorch CPU intra-op/inter-op thread count in deterministic mode",
    )
    parser.add_argument('--tree_g', type=int, default=2, help='Branching factor for regular tree')
    parser.add_argument('--tree_h', type=int, default=10, help='Depth for regular tree')

    # sEH_large environment (paper Sec. A.3 defaults; override here if desired)
    parser.add_argument('--seh_large_max_blocks', type=int, default=8,
                        help='sEH_large: max blocks per molecule (paper: 8)')
    parser.add_argument('--seh_large_min_blocks', type=int, default=2,
                        help='sEH_large: min blocks before stop is allowed (paper: 2)')
    parser.add_argument('--seh_large_max_stems', type=int, default=40,
                        help='sEH_large: max free stems for action-space sizing (action_dim = 105*max_stems+1)')
    parser.add_argument('--seh_large_reward_t', type=float, default=8.0,
                        help='sEH_large: reward T (paper: 8)')
    parser.add_argument('--seh_large_reward_exp', type=float, default=10.0,
                        help='sEH_large: reward beta (paper: 10)')
    parser.add_argument('--seh_large_reward_min', type=float, default=0.01,
                        help='sEH_large: R_min, the proxy-reward floor (paper: 0.01)')
    parser.add_argument(
        "--seh_large_proxy_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help=(
            "Device for the pretrained sEH proxy. In deterministic mode, "
            "'auto' resolves to CPU and explicit CUDA is rejected because "
            "the PyG graph reductions are not guaranteed deterministic."
        ),
    )
    parser.add_argument(
        '--seh_large_mode_r_threshold',
        type=float,
        default=7.5,
        help=(
            'sEH_large: inclusive raw proxy-reward cutoff for online mode '
            'tracking and final diversity evaluation (default: R>=7.5)'
        ),
    )

    # Learning algorithm
    parser.add_argument('-m', '--method', type=str, default='TB',
        choices=['TB', 'SubTB', 'DB', 'WDB', 'FM', 'Teacher', 'Stable', 'StableTeacher'],
        help='GFlowNet learning algorithm')

    # Training hyperparameters
    parser.add_argument('-t', '--total_iterations', type=int, default=10000, help='Total iterations')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=1000, help='Replay memory size')
    parser.add_argument('--train_freq', type=int, default=32, help='Episodes between updates')
    parser.add_argument('--gradient_steps', type=int, default=1, help='Gradient steps per update')
    parser.add_argument('--off_policy', action='store_true', help='Use off-policy training')
    parser.add_argument('--sample_method', type=int, default=0, help='Sample method: 0 - uniform, 1 - reward prioritized')

    # Temperature settings
    parser.add_argument('--policy_temperature', type=float, default=1.0, help='Policy temperature') # log PF/policy_temperature
    parser.add_argument('--reward_temperature', type=float, default=1.0, help='Reward temperature') # log reward/reward_temperature
    parser.add_argument('--temperature_decay_rate', type=float, default=0.999, help='Temperature decay')

    # GFlowNet settings
    parser.add_argument('--initial_z', type=float, default=0.0, help='Initial log Z')
    parser.add_argument('--use_state_flow', action='store_true', help='Use logF instead of logZ')
    parser.add_argument('--fixed_PB', action='store_true', help='Use fixed (uniform) backward policy instead of trainable neural network.')

    # epsilon greedy
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon for epsilon greedy')

    # SubTB settings
    parser.add_argument('--weighting', type=str, default='geometric_within',
        choices=['DB', 'ModifiedDB', 'TB', 'equal_within', 'equal', 'geometric_within', 'geometric'],
        help='Sub-trajectory weighting for SubTB')
    parser.add_argument('--lamda', type=float, default=0.9, help='Lambda for geometric weighting')

    # Adaptive Teacher settings
    parser.add_argument('--alpha', type=float, default=0.0, help='Teacher mixing coefficient (for Teacher and StableTeacher methods, default: 0.0 for Teacher, 0.5 recommended for StableTeacher)')

    # Stable GFlowNets settings
    parser.add_argument('--target_TV', type=float, default=0.1, help='Target TV distance')
    parser.add_argument('--verification_budget', type=int, default=512, help='Verification samples')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='Confidence level')
    parser.add_argument('--patience', type=int, default=10, help='Patience for subgraph verification')
    parser.add_argument('--moving_average_factor', type=float, default=0.05, help='Moving average factor for loss threshold')
    parser.add_argument('--max_state_size', type=int, default=10000, help='Maximum number of terminal states to keep (top N by reward)')
    parser.add_argument('--stabilizing_factor', type=str, default='max',
                        choices=['max', 'mean', 'median', 'null'],
                        help='For stable GFlowNets: how to set the reference-flow threshold c '
                             "(max/mean/median of the per-batch TB loss). 'null' disables the "
                             'reference flow entirely (augmented loss == original TB loss).')
    # Per-source trajectory budgets used by Stable / StableTeacher.
    # batch_size (and train_freq) is derived as the sum of these values for
    # those methods, overriding any value passed via -b/--batch_size.
    parser.add_argument('--forward_traj', type=int, default=16,
                        help='Stable/StableTeacher: # forward trajectories collected per iteration')
    parser.add_argument('--backward_traj', type=int, default=16,
                        help='Stable/StableTeacher: # backward trajectories collected per iteration')
    parser.add_argument('--teacher_traj', type=int, default=16,
                        help='StableTeacher: # teacher forward trajectories collected per iteration')

    # Neural network architecture
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 256], help='Hidden sizes')
    parser.add_argument('--activation', type=str, default='leakyrelu',
        choices=['relu', 'tanh', 'sigmoid', 'leakyrelu'], help='Activation function')

    # Logging
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--output_dir', type=str, default='../output', help='Output directory')

    # Optional execution/output controls; existing defaults remain unchanged.
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Training device (auto preserves automatic CUDA selection)')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default=None,
                        help='W&B mode; omitted preserves WANDB_MODE/default behavior')
    parser.add_argument('--runtime-profile', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable existing runtime instrumentation on supported methods')
    parser.add_argument('--cuda-profile', action=argparse.BooleanOptionalAction, default=True,
                        help='Collect sparse CUDA kernel timings when runtime profiling is enabled')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help='Iterations between checkpoints; 0 disables periodic checkpoints')
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True,
                        help='Display the training progress bar')
    parser.add_argument('--save-final-model', action=argparse.BooleanOptionalAction, default=True,
                        help='Save final model, optimizer and progress files')
    parser.add_argument('--save-memory', action=argparse.BooleanOptionalAction, default=True,
                        help='Save the final replay buffer')
    parser.add_argument('--final-evaluation', action=argparse.BooleanOptionalAction, default=True,
                        help='Run the existing final mode/diversity evaluation')
    parser.add_argument('--skip-existing', action=argparse.BooleanOptionalAction, default=True,
                        help='Skip runs with an existing final forward-policy checkpoint')
    parser.add_argument('--run-tag', default=None,
                        help='Append a unique run label to output and automatic W&B names')
    parser.add_argument('--metrics-jsonl', action=argparse.BooleanOptionalAction, default=False,
                        help='Write local training_metrics.jsonl without requiring W&B')

    # Model loading
    parser.add_argument('--load_model', type=str, default='', help='Model checkpoint to load')
    
    # Parallel collection settings
    parser.add_argument('--single_threaded', action='store_true', 
                       help='Use single-threaded collection (disable parallel environments)')

    
    args = parser.parse_args(argv)
    if args.checkpoint_interval < 0:
        parser.error('--checkpoint-interval must be nonnegative')
    if args.run_tag is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_tag):
        parser.error('--run-tag must start with a letter/digit and contain only letters, digits, _, . or -')
    return args


def seed_all(seed, deterministic=True, torch_num_threads=1):
    """Seed all RNGs and configure PyTorch's global determinism contract."""
    if torch_num_threads < 1:
        raise ValueError(
            f"torch_num_threads must be at least 1, got {torch_num_threads}"
        )

    if deterministic:
        torch.set_num_threads(torch_num_threads)
        if torch.get_num_interop_threads() != torch_num_threads:
            try:
                torch.set_num_interop_threads(torch_num_threads)
            except RuntimeError as error:
                raise RuntimeError(
                    "PyTorch inter-op threads were initialized before the "
                    "deterministic configuration. Call seed_all() before "
                    "creating tensors, models, or data loaders."
                ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    if deterministic:
        # TF32 changes float32 accumulation on Ampere GPUs such as the A10.
        # Disable it so the numerical contract is explicit and portable across
        # machines with otherwise identical software and GPU models.
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")


def resolve_reproducibility_args(args):
    """Resolve device choices that are part of the determinism contract."""
    if (
        args.sim.lower() == "seh_large"
        and args.deterministic
        and args.seh_large_proxy_device == "cuda"
    ):
        raise ValueError(
            "Deterministic sEH_large training requires the pretrained proxy "
            "on CPU. Use --seh_large_proxy_device cpu (or leave it as auto), "
            "or explicitly opt out with --no-deterministic."
        )
    if (
        args.sim.lower() == "seh_large"
        and args.deterministic
        and args.seh_large_proxy_device == "auto"
    ):
        args.seh_large_proxy_device = "cpu"
    return args


def get_activation_fn(name):
    """Get activation function from name."""
    activations = {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid,
        'leakyrelu': torch.nn.LeakyReLU,
    }
    return activations.get(name.lower(), torch.nn.ReLU)


def create_environment(args):
    """Create environment based on arguments."""
    sim = args.sim.lower()
    if sim == 'hypergrid-simple':
        return HyperGridEnv(preset='simple')
    elif sim == 'hypergrid-mild':
        return HyperGridEnv(preset='mild')
    elif sim == 'hypergrid-hard':
        return HyperGridEnv(preset='hard')
    elif sim == 'regular-tree':
        return RegularTreeEnv(g=args.tree_g, h=args.tree_h)
    elif sim == 'rna14':
        from envs.rna14 import Environment as RNA14Env
        return RNA14Env()
    elif sim == 'seh':
        from envs.sEH import Environment as SEHEnv
        return SEHEnv()
    elif sim == 'seh_large':
        from envs.sEH_large import Environment as SEHLargeEnv
        proxy_device = (
            None
            if args.seh_large_proxy_device == "auto"
            else args.seh_large_proxy_device
        )
        return SEHLargeEnv(
            max_blocks=args.seh_large_max_blocks,
            min_blocks=args.seh_large_min_blocks,
            max_stems=args.seh_large_max_stems,
            reward_t=args.seh_large_reward_t,
            reward_exp=args.seh_large_reward_exp,
            scale_reward_min=args.seh_large_reward_min,
            mode_r_raw_threshold=args.seh_large_mode_r_threshold,
            evaluation_mode_r_raw_threshold=args.seh_large_mode_r_threshold,
            device=proxy_device,
        )
    else:
        raise ValueError(f"Unknown environment: {args.sim}")


def create_model(env, args, data_env=None, model_dir=None):
    """Create GFlowNet model based on arguments."""
    method = args.method.upper()
    activation_fn = get_activation_fn(args.activation)

    common_kwargs = {
        'env': env,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'memory_size': args.memory_size,
        'train_freq': args.train_freq,
        'gradient_steps': args.gradient_steps,
        'policy_temperature': args.policy_temperature,
        'reward_temperature': args.reward_temperature,
        'temperature_decay_rate': args.temperature_decay_rate,
        'sample_method': args.sample_method,
        'verbose': args.verbose,
        'hidden_sizes': args.hidden_sizes,
        'activation_fn': activation_fn,
        'initial_z': args.initial_z,
        'use_state_flow': args.use_state_flow,
        'wandb_project': args.wandb_project,
        'wandb_name': args.wandb_name,
        'device': args.device,
        'runtime_profile': args.runtime_profile,
        'cuda_profile': args.cuda_profile,
        'checkpoint_interval': args.checkpoint_interval,
        'progress': args.progress,
        'metrics_jsonl': args.metrics_jsonl,
        'data_env': data_env,
        'model_dir': model_dir,
        'trainable_logPB': not args.fixed_PB,  # fixed_PB=True means trainable_logPB=False
        'max_state_size': args.max_state_size,
        'epsilon': args.epsilon,
    }

    if method == 'TB':
        return TrajectoryBalance(**common_kwargs)
    elif method == 'SUBTB':
        return SubTrajectoryBalance(weighting=args.weighting, lamda=args.lamda, **common_kwargs)
    elif method == 'DB':
        return DetailedBalance(**common_kwargs)
    elif method == 'WDB':
        return DetailedBalanceWeightedByReachableEndStates(**common_kwargs)
    elif method == 'FM':
        return FlowMatching(**common_kwargs)
    elif method == 'TEACHER':
        return TrajectoryBalanceAdaptiveTeacher(alpha=args.alpha, **common_kwargs)
    elif method == 'STABLE':
        return TrajectoryBalanceStabilizedWithProbabilisticGuarantees(
            target_TV=args.target_TV,
            verification_budget=args.verification_budget,
            confidence_level=args.confidence_level,
            patience=args.patience,
            moving_average_factor=args.moving_average_factor,
            stabilizing_factor=args.stabilizing_factor,
            forward_traj=args.forward_traj,
            backward_traj=args.backward_traj,
            **common_kwargs
        )
    elif method == 'STABLETEACHER':
        return TrajectoryBalanceStabilizedWithProbabilisticGuaranteesAndAdaptiveTeacher(
            target_TV=args.target_TV,
            verification_budget=args.verification_budget,
            confidence_level=args.confidence_level,
            patience=args.patience,
            moving_average_factor=args.moving_average_factor,
            teacher_alpha=args.alpha,
            stabilizing_factor=args.stabilizing_factor,
            forward_traj=args.forward_traj,
            backward_traj=args.backward_traj,
            teacher_traj=args.teacher_traj,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")


def main():
    args = resolve_reproducibility_args(get_arguments(sys.argv[1:]))

    seed_all(
        args.seed,
        deterministic=args.deterministic,
        torch_num_threads=args.torch_num_threads,
    )

    print("=" * 60)
    print("GFlowNet Training Configuration")
    print("=" * 60)

    # Stable / StableTeacher derive batch_size from per-source trajectory
    # budgets (forward/backward [+ teacher]). Any value passed via -b is
    # overridden so the in-class assertion (sum == batch_size) holds.
    method_upper = args.method.upper()
    if method_upper == 'STABLE':
        derived_bs = args.forward_traj + args.backward_traj
        if args.batch_size != derived_bs:
            print(f"  [Stable] batch_size: overriding {args.batch_size} -> {derived_bs}"
                  f" (forward_traj={args.forward_traj} + backward_traj={args.backward_traj})")
        args.batch_size = derived_bs
        args.train_freq = derived_bs  # latest_sample(self.train_freq) must cover all new traj
    elif method_upper == 'STABLETEACHER':
        derived_bs = args.forward_traj + args.backward_traj + args.teacher_traj
        if args.batch_size != derived_bs:
            print(f"  [StableTeacher] batch_size: overriding {args.batch_size} -> {derived_bs}"
                  f" (forward_traj={args.forward_traj} + backward_traj={args.backward_traj}"
                  f" + teacher_traj={args.teacher_traj})")
        args.batch_size = derived_bs
        args.train_freq = derived_bs

    if not args.off_policy:
      args.memory_size = args.batch_size
      args.train_freq = args.batch_size
      args.sample_method = 0
    # Keep an immutable snapshot. Later we temporarily change args.wandb_project
    # while constructing the model; that implementation detail must not alter
    # the saved/W&B experiment configuration.
    config = vars(args).copy()

    for key, value in config.items():
        # Show actual values used in on-policy mode
        if not args.off_policy and key == 'memory_size':
            print(f"  {key}: {args.batch_size} (on-policy: equals batch_size)")
        elif not args.off_policy and key == 'train_freq':
            print(f"  {key}: {args.batch_size} (on-policy: equals batch_size)")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)

    print(f"\nCreating environment: {args.sim}")
    env = create_environment(args)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Create vectorized environment for parallel collection (unless single-threaded)
    data_env = None
    if not args.single_threaded:
        from utils.vectorized_env import VectorizedEnv, VectorizedEnvWrapper
        # Create data_env with train_freq//2 instances for parallel collection
        # (half for forward, half for backward/local_search/teacher)
        num_parallel_envs = max(1, args.train_freq // 2)
        vectorized_env = VectorizedEnv(lambda: create_environment(args), num_envs=num_parallel_envs)
        data_env = VectorizedEnvWrapper(vectorized_env)
        data_env.reset(seed=args.seed + 1)
        data_env.action_space.seed(args.seed + 1)
        data_env.observation_space.seed(args.seed + 1)
        print(f"  Created vectorized environment with {num_parallel_envs} parallel instances")
    else:
        print(f"  Using single-threaded collection (no parallel environments)")

    # Create model directory before model creation
    # Include distinguishing parameters to avoid conflicts
    model_name_parts = [args.method, args.sim, str(args.seed)]
    if args.deterministic:
        # Do not silently reuse a checkpoint produced under the historical,
        # weaker reproducibility settings.
        model_name_parts.append("det")
    
    # Add environment-specific parameters
    if args.sim == 'regular-tree':
        model_name_parts.append(f"g{args.tree_g}_h{args.tree_h}")
    if args.sim == 'seh_large':
        if args.seh_large_max_blocks != 8:
            model_name_parts.append(f"mb{args.seh_large_max_blocks}")
        if args.seh_large_max_stems != 40:
            model_name_parts.append(f"ms{args.seh_large_max_stems}")
        if args.seh_large_reward_t != 8.0:
            model_name_parts.append(f"T{args.seh_large_reward_t}")
        if args.seh_large_reward_exp != 10.0:
            model_name_parts.append(f"beta{args.seh_large_reward_exp}")
        if args.seh_large_mode_r_threshold != 8.0:
            model_name_parts.append(f"mthr{args.seh_large_mode_r_threshold}")

    # Add key hyperparameters that might vary between runs
    # Only include if they differ from defaults to keep names shorter
    if args.learning_rate != 1e-3:  # default is 1e-3
        model_name_parts.append(f"lr{args.learning_rate}")
    if args.target_TV != 0.1:  # default is 0.1
        model_name_parts.append(f"tv{args.target_TV}")
    if args.fixed_PB:  # if using fixed PB
        model_name_parts.append("fixedPB")
    if args.sample_method != 0:  # if using sample method
        model_name_parts.append(f"sample{args.sample_method}")
    if args.hidden_sizes != [256, 256]:  # if using non-default hidden sizes
        hidden_str = "_".join(map(str, args.hidden_sizes))
        model_name_parts.append(f"hidden{hidden_str}")
    if args.initial_z != 0.0:  # if using non-zero initial z, a few GFlowNets literature use 0.1 as a default, I found it does matter with gradient clipping.
        model_name_parts.append(f"initial_z{args.initial_z}")
    if args.epsilon != 0.0:  # if using epsilon-greedy exploration
        model_name_parts.append(f"eps{args.epsilon}")
    if args.stabilizing_factor != 'max':
        model_name_parts.append(f"stabfac_{args.stabilizing_factor}")
    if args.max_state_size != 10000:
        model_name_parts.append(f"maxstates{args.max_state_size}")
    if method_upper == 'STABLE':
        model_name_parts.append(f"fwd{args.forward_traj}_bwd{args.backward_traj}")
    elif method_upper == 'STABLETEACHER':
        model_name_parts.append(
            f"fwd{args.forward_traj}_bwd{args.backward_traj}_tch{args.teacher_traj}"
        )

    if args.run_tag:
        model_name_parts.append(args.run_tag)
    model_name = "_".join(model_name_parts)
    model_dir = os.path.join(args.output_dir, args.sim, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Check if model already exists (skip rerun if forward_policy.pth exists)
    # Do this early to avoid creating wandb runs or model instances
    forward_policy_path = os.path.join(model_dir, 'forward_policy.pth')
    if args.skip_existing and os.path.exists(forward_policy_path) and not args.load_model:
        print(f"\nModel already exists at: {forward_policy_path}")
        print("Skipping training to avoid rerunning the same experiment.")
        print("To rerun, either:")
        print(f"  1. Delete the model directory: {model_dir}")
        print(f"  2. Use --load_model to continue training from checkpoint")
        print("  3. Use --run-tag for a distinct run or --no-skip-existing to rerun")
        return

    reproducibility_record = collect_reproducibility_record(
        seed=args.seed,
        deterministic=args.deterministic,
        source_root=os.path.dirname(os.path.abspath(__file__)),
        proxy_device=(
            args.seh_large_proxy_device
            if args.sim.lower() == "seh_large"
            else None
        ),
    )
    write_reproducibility_record(model_dir, reproducibility_record)
    print(
        "Reproducibility record: "
        f"{os.path.join(model_dir, REPRODUCIBILITY_FILENAME)}"
    )

    print(f"\nCreating model: {args.method}")
    
    # Determine wandb project and name
    # Use gfn_[env_name] as project name if not specified
    if args.wandb_project is None:
        env_name = args.sim.lower().replace('-', '_')
        wandb_project = f"gfn_{env_name}"
    else:
        wandb_project = args.wandb_project
    
    # Temporarily disable wandb in model creation, we'll initialize it after
    original_wandb_project = args.wandb_project
    args.wandb_project = None  # Disable during model creation
    
    model = create_model(env, args, data_env, model_dir)
    args.wandb_project = original_wandb_project
    
    # Set wandb name to algorithm class name if not specified
    # Include distinguishing parameters to make names unique
    if args.wandb_name is None:
        algorithm_class_name = model.__class__.__name__
        wandb_name_parts = [algorithm_class_name]
        
        # Add environment-specific parameters
        if args.sim == 'regular-tree':
            wandb_name_parts.append(f"g{args.tree_g}_h{args.tree_h}")
        
        # Add seed to make it unique
        wandb_name_parts.append(f"seed{args.seed}")
        if args.deterministic:
            wandb_name_parts.append("det")
        
        # Add key hyperparameters that might vary between runs
        if args.learning_rate != 1e-3:  # default is 1e-3
            wandb_name_parts.append(f"lr{args.learning_rate}")
        if args.target_TV != 0.1:  # default is 0.2
            wandb_name_parts.append(f"tv{args.target_TV}")
        if args.fixed_PB:  # if using fixed PB
            wandb_name_parts.append("fixedPB")
        if args.sample_method != 0:  # if using sample method
            wandb_name_parts.append(f"sample{args.sample_method}")
        if args.hidden_sizes != [256, 256]:  # if using non-default hidden sizes
            hidden_str = "_".join(map(str, args.hidden_sizes))
            wandb_name_parts.append(f"hidden{hidden_str}")
        if args.initial_z != 0.0:  # if using non-default initial z
            wandb_name_parts.append(f"initial_z{args.initial_z}")
        if args.epsilon != 0.0:  # if using epsilon-greedy exploration
            wandb_name_parts.append(f"eps{args.epsilon}")
        if args.stabilizing_factor != 'max':
            wandb_name_parts.append(f"stabfac_{args.stabilizing_factor}")
        if args.max_state_size != 10000:
            wandb_name_parts.append(f"maxstates{args.max_state_size}")
        if method_upper == 'STABLE':
            wandb_name_parts.append(f"fwd{args.forward_traj}_bwd{args.backward_traj}")
        elif method_upper == 'STABLETEACHER':
            wandb_name_parts.append(
                f"fwd{args.forward_traj}_bwd{args.backward_traj}_tch{args.teacher_traj}"
            )

        if args.run_tag:
            wandb_name_parts.append(args.run_tag)
        wandb_name = "_".join(wandb_name_parts)
    else:
        wandb_name = args.wandb_name
    
    # Initialize wandb with correct project and name
    if wandb_project is not None and args.wandb_mode != "disabled":
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config,
            **({"mode": args.wandb_mode} if args.wandb_mode is not None else {})
        )
        model.use_wandb = True

    with open(os.path.join(model_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if args.load_model:
        print(f"\nLoading model from: {args.load_model}")
        model.load(args.load_model)

    print(f"\nStarting training for {args.total_iterations} iterations...")
    if args.wandb_project is not None:
        print(f"  W&B Project: {args.wandb_project}")
        print(f"  W&B Run Name: {args.wandb_name}")
    model.learn(total_iterations=args.total_iterations)

    if args.save_final_model:
        print(f"\nSaving model to: {model_dir}")
        model.save(model_dir)
    if args.save_memory:
        model.save_memory('memory.pkl')

    final_mode_metrics = None
    if args.final_evaluation:
        final_mode_metrics = evaluate_and_store_mode_metrics(
            model,
            env,
            model_dir,
        )
    if final_mode_metrics is not None:
        counts = final_mode_metrics["counts_tuple"]
        print(
            "Final mode metrics "
            "(unique, Tanimoto-separated, Bemis-Murcko): "
            f"{tuple(counts)}"
        )
        print(
            "Mode metrics saved to: "
            f"{os.path.join(model_dir, MODE_METRICS_FILENAME)}"
        )

        if getattr(model, "use_wandb", False) and wandb.run is not None:
            try:
                for metric_name, value in final_mode_metrics["metrics"].items():
                    wandb.run.summary[f"eval/{metric_name}"] = value
                wandb.run.summary["eval/num_tracked_modes"] = (
                    final_mode_metrics["num_tracked_modes"]
                )
            except Exception as error:
                print(f"Warning: could not update W&B mode summaries: {error}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()

      
      
