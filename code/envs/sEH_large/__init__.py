"""sEH_large: faithful port of the molecule-graph MDP described in
Bengio et al., NeurIPS 2021 (Supplementary A.3, OpenReview Arn2E4IRjEB).

From the paper appendix:
  > "We allow the agent to choose from a library of 72 predefined blocks.
  > We duplicate blocks from the point of view of the agent to allow
  > attaching to different symmetry groups of a given block. This yields
  > a total of 105 actions per stem; stems are atoms where new blocks
  > can be attached to. ... We allow the agent to generate up to 8
  > blocks."

Translation to this codebase:
  * Library of 72 unique SMILES, exposed as 105 indexed blocks (the
    duplicates that exist in ``blocks_PDB_105.json`` to enumerate
    symmetry groups). ``num_blocks = 105`` is the "actions-per-stem"
    quantity.
  * State internally is a ``BlockMoleculeDataExtended`` (the real
    molecule graph). For the gym interface it is exposed as a
    fixed-shape build-trace observation: at most ``max_blocks=8``
    sequential additions, each encoded as one-hot of
    ``(block_idx, parent_block_position, parent_stem_atom)``. This
    encoding is bijective with the molecule.
  * Action space is ``Discrete(num_blocks * MAX_STEMS + 1)`` with the
    GFNOrg/gflownet convention
        block_idx = a % num_blocks
        stem_idx  = a // num_blocks
        a == num_blocks * MAX_STEMS  => stop
    ``MAX_STEMS`` is a fixed upper bound on the number of free stems
    (default 40, large enough that masking — not the action-space
    size — is what restricts the policy in practice).
  * Backward action shares the ``Discrete(action_dim)`` shape
    (required by ``gfn_base.DiscreteBackwardPolicy(action_dim)``).
    A backward action ``a`` is decoded with the same
    ``(block_idx, stem_idx)`` rule and means "undo the addition that
    recorded this exact ``(block_idx, stem_idx)`` in the build trace".
    The mask is True at positions whose corresponding leaf still
    exists in the current molecule, exactly mirroring the parent set
    of the molecule MDP.
  * Reward is the same pretrained NN proxy used in the paper, loaded
    from ``envs/sEH_large/pretrained_proxy/``. Reward scaling follows
    the paper hyperparameter table exactly:
        R_hat(x) = (max(R(x), R_min) / T) ** beta
        T = 8        (paper: "Reward T 8")
        beta = 10    (paper: "Reward beta 10")
        R_min = 0.01 (paper: "R is clipped below R_min")
    ``unnormalize(R_hat) = R_hat ** (1/beta) * T`` is the trivial inverse.
    Both rollout-time tracking and final evaluation use the inclusive raw
    proxy cutoff ``R(x) >= mode_r_raw_threshold`` (default ``R >= 7.5``).

Required packages: ``rdkit``, ``torch``, ``torch_geometric``,
``polyleven``. ``torch_sparse`` is NOT required (uses
``torch_geometric.utils.coalesce``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym

# --- dependency check ----------------------------------------------------------
_MISSING = []
try:
    import torch  # noqa: F401
except ImportError:
    _MISSING.append("torch")
try:
    import torch_geometric  # noqa: F401
except ImportError:
    _MISSING.append("torch_geometric")
try:
    import rdkit  # noqa: F401
except ImportError:
    _MISSING.append("rdkit")
try:
    from polyleven import levenshtein  # noqa: F401
except ImportError:
    _MISSING.append("polyleven")

if _MISSING:
    raise ImportError(
        "envs.sEH_large requires the following packages that are not installed: "
        + ", ".join(_MISSING)
        + ". Install them e.g. with:\n"
        "  pip install torch torch_geometric polyleven\n"
        "  conda install -c conda-forge rdkit\n"
        "torch_sparse is NOT required (the env uses torch_geometric.utils.coalesce)."
    )

import torch
from rdkit import Chem
from polyleven import levenshtein

from .mode_metrics import (
    PAPER_TANIMOTO_SIMILARITY_THRESHOLD,
    count_mode_metrics,
)
from .proxy import SEHGraphProxy
from .mol_mdp_ext import BlockMoleculeDataExtended


_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_BLOCKS_FILE = os.path.join(_DATA_DIR, "blocks_PDB_105.json")
_PROXY_DIR = os.path.join(_DATA_DIR, "pretrained_proxy")


__all__ = ["Environment"]


# --- build-trace dataclass -----------------------------------------------------
@dataclass(frozen=True)
class _BuildStep:
    """One step of the molecule build trace.

    ``block_idx``  - which block was added (0..num_blocks-1)
    ``parent_pos`` - position (in the molecule) of the block this one
                     attached to. ``-1`` for the root block (step 0).
    ``parent_atom``- atom index on the parent block at which the bond
                     was made. ``-1`` for the root block.
    ``stem_idx``   - the stem index in the molecule's ``mol.stems``
                     list AT the time of attachment. Stored so we can
                     reconstruct the (block_idx, stem_idx)
                     forward-action that originally produced this leaf
                     (used for the backward action mask).
                     ``-1`` for the root block.
    """

    block_idx: int
    parent_pos: int
    parent_atom: int
    stem_idx: int


# --- main env ------------------------------------------------------------------
class Environment(gym.Env):
    """Molecule-graph sEH env from Bengio et al. (2021), Sec. A.3.

    Subclasses :class:`gymnasium.Env` directly (mirroring ``envs.rna14``).
    The env intentionally does NOT implement ``sample_terminal_states``;
    backward-trajectory training (Stable / StableTeacher) drives a
    top-K reward-buffer maintained inside ``learning/gfn_base.py`` --
    no proxy evaluation is needed at sampling time.

    Implements every method that ``learning/gfn_*.py`` calls via
    ``env.unwrapped``:
      - ``reset``, ``step``                    (gym)
      - ``get_forward_action_masks``           (forward policy mask)
      - ``get_backward_action_masks``          (backward policy mask)
      - ``apply_backward_action``              (backward trajectory rollout)
      - ``get_state`` / ``is_initial_state``   (obs <-> SMILES)
      - ``is_mode`` / ``get_error``            (validation / mode counting)
      - ``_compute_reward(state=None)``        (paper-scaled reward)
      - ``real_reward`` / ``unnormalize``      (raw proxy / inverse scale)
      - ``get_num_of_reachable_end_states``    (upper bound, used by WDB)

    Constants below match the paper hyperparameters (Sec. A.3 / Table of
    flow-model hyperparameters):
        max_blocks = 8     (# of blocks per molecule, upper bound)
        min_blocks = 2     (lower bound before stop is allowed)
    """

    metadata = {"render_modes": []}

    NUM_BLOCKS = 105  # paper: "105 actions per stem"
    MAX_BLOCKS = 8  # paper: "up to 8 blocks"
    MIN_BLOCKS = 2  # paper hyperparameter
    MAX_STEM_ATOM = 8  # max stem atom-index across the 105 blocks (block 100 has atom 8)

    # Paper hyperparameters (Sec. A.3, "summary of the flow model hyperparameters"):
    #   "Reward T 8   Reward beta 10   R_hat(x) = (R(x)/T)^beta
    #    R is clipped below R_min, i.e. R_hat_min = (R_min/T)^beta   R_min = 0.01"
    REWARD_T = 8.0  # paper: T
    REWARD_EXP = 10.0  # paper: beta
    SCALE_REWARD_MIN = 0.01  # paper: R_min

    # Mode thresholds expressed in *raw* proxy-reward units.  The default
    # cutoff is 7.5 for compatibility with the archived sEH_large runs.  Keep
    # the paper's R >= 8 cutoff available as an explicit evaluation setting.
    MODE_R_RAW_THRESHOLD = 7.5
    EVALUATION_MODE_R_RAW_THRESHOLD = MODE_R_RAW_THRESHOLD
    PAPER_EVALUATION_R_RAW_THRESHOLD = 8.0
    TANIMOTO_SIMILARITY_THRESHOLD = PAPER_TANIMOTO_SIMILARITY_THRESHOLD
    MODE_METRIC_ENVIRONMENT = "sEH_large"
    MODE_METRIC_NAMES = (
        "unique_modes",
        "tanimoto_separated_modes",
        "bemis_murcko_scaffolds",
    )

    def __init__(
        self,
        render_mode=None,
        max_blocks: int = MAX_BLOCKS,
        min_blocks: int = MIN_BLOCKS,
        max_stems: int = 40,
        device: Optional[str] = None,
        reward_t: float = REWARD_T,
        reward_exp: float = REWARD_EXP,
        scale_reward_min: float = SCALE_REWARD_MIN,
        mode_r_raw_threshold: float = MODE_R_RAW_THRESHOLD,
        evaluation_mode_r_raw_threshold: float = (
            EVALUATION_MODE_R_RAW_THRESHOLD
        ),
        tanimoto_similarity_threshold: float = TANIMOTO_SIMILARITY_THRESHOLD,
    ):
        super().__init__()

        self.max_blocks = int(max_blocks)
        self.min_blocks = int(min_blocks)
        self.max_stems = int(max_stems)
        self.REWARD_T = float(reward_t)
        self.REWARD_EXP = float(reward_exp)
        self.SCALE_REWARD_MIN = float(scale_reward_min)
        self.mode_r_raw_threshold = float(mode_r_raw_threshold)
        self.evaluation_mode_r_raw_threshold = float(
            evaluation_mode_r_raw_threshold
        )
        self.tanimoto_similarity_threshold = float(tanimoto_similarity_threshold)
        if not 0.0 <= self.tanimoto_similarity_threshold <= 1.0:
            raise ValueError(
                "tanimoto_similarity_threshold must be between 0 and 1, "
                f"got {self.tanimoto_similarity_threshold}"
            )

        # --- Load proxy + the underlying 105-block MDP ---------------------------
        torch_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.proxy = SEHGraphProxy(
            blocks_file=_BLOCKS_FILE, proxy_dir=_PROXY_DIR, device=torch_device
        )
        self.mdp = self.proxy.mdp
        assert self.mdp.num_blocks == self.NUM_BLOCKS, (
            f"blocks_PDB_105.json should contain {self.NUM_BLOCKS} blocks, "
            f"got {self.mdp.num_blocks}"
        )
        self.num_unique_blocks = len(set(self.mdp.block_smi))  # = 72 for paper library

        # --- Per-step observation layout (one-hot, fixed-size) -------------------
        # Convention matches envs.sEH / envs.rna14: an "empty" or
        # "padded" slot is the all-zero vector (no explicit EMPTY bit).
        # This is required so PyTorch's ``pad_sequence`` (which pads
        # with zeros) produces obs that decode back to a shorter (or
        # empty) trace -- otherwise zero-padded rows would be
        # mis-decoded as ``block_idx=0`` etc.
        #   block_idx   in {0..104}           -> NUM_BLOCKS dims  (all 0 = no block here)
        #   parent_pos  in {0..max_blocks-1}  -> max_blocks dims  (all 0 = root step)
        #   parent_atom in {0..MAX_STEM_ATOM} -> MAX_STEM_ATOM+1  (all 0 = root step)
        self._block_oh_dim = self.NUM_BLOCKS
        self._parent_pos_oh_dim = self.max_blocks
        self._parent_atom_oh_dim = self.MAX_STEM_ATOM + 1
        self._slot_dim = (
            self._block_oh_dim + self._parent_pos_oh_dim + self._parent_atom_oh_dim
        )
        self._obs_dim = self.max_blocks * self._slot_dim

        self.observation_space = gym.spaces.Box(
            shape=(self._obs_dim,), low=0, high=1, dtype=np.float32
        )

        # --- Action space (forward = backward, shared shape) ---------------------
        self.action_dim = self.NUM_BLOCKS * self.max_stems + 1
        self.STOP_ACTION = self.action_dim - 1
        self.action_space = gym.spaces.Discrete(self.action_dim)

        # --- Reward scaling (paper formula) --------------------------------------
        # R_hat(x) = (max(R(x), R_min) / T)^beta -- computed inline in
        # `_scaled_reward_of_smiles`; `unnormalize` is the trivial
        # inverse `R_hat ** (1/beta) * T`. No precomputed scale field
        # is needed.
        self.reward_cache: dict = {}
        # Translate the configured raw mode cutoff to scaled-reward units.
        # Both ``get_error`` and ``is_mode`` apply the inclusive cutoff.
        clipped_thr = max(self.mode_r_raw_threshold, self.SCALE_REWARD_MIN)
        self.mode_r_threshold: float = (clipped_thr / self.REWARD_T) ** self.REWARD_EXP
        clipped_eval_thr = max(
            self.evaluation_mode_r_raw_threshold,
            self.SCALE_REWARD_MIN,
        )
        self.evaluation_mode_r_threshold: float = (
            clipped_eval_thr / self.REWARD_T
        ) ** self.REWARD_EXP

        # --- Internal state ------------------------------------------------------
        self.mol: BlockMoleculeDataExtended = BlockMoleculeDataExtended()
        self.build_trace: List[_BuildStep] = []

    # ============================================================== chain build
    def _add_block_step(self, block_idx: int, stem_idx: Optional[int]) -> None:
        """Apply one forward addition and record the build step."""
        if len(self.build_trace) == 0:
            assert stem_idx is None or stem_idx == 0, (
                "First block must be added with stem_idx=None"
            )
            self.mol = self.mdp.add_block_to(self.mol, int(block_idx))
            self.build_trace.append(_BuildStep(int(block_idx), -1, -1, -1))
            return
        assert stem_idx is not None
        parent_pos, parent_atom = (int(self.mol.stems[stem_idx][0]),
                                   int(self.mol.stems[stem_idx][1]))
        self.mol = self.mdp.add_block_to(self.mol, int(block_idx), stem_idx=int(stem_idx))
        self.build_trace.append(
            _BuildStep(int(block_idx), parent_pos, parent_atom, int(stem_idx))
        )

    def _rebuild_from_trace(self, trace: List[_BuildStep]) -> BlockMoleculeDataExtended:
        """Rebuild a BlockMoleculeDataExtended from a build trace."""
        mol = BlockMoleculeDataExtended()
        for i, step in enumerate(trace):
            if i == 0:
                mol = self.mdp.add_block_to(mol, step.block_idx)
                continue
            target = (step.parent_pos, step.parent_atom)
            stem_idx = None
            for j, s in enumerate(mol.stems):
                if int(s[0]) == target[0] and int(s[1]) == target[1]:
                    stem_idx = j
                    break
            if stem_idx is None:
                raise RuntimeError(
                    f"Could not locate stem ({target}) in molecule when "
                    f"replaying build trace; mol.stems = {list(mol.stems)}"
                )
            mol = self.mdp.add_block_to(mol, step.block_idx, stem_idx=stem_idx)
        return mol


    # ============================================================== rewards
    def _sample_random_terminal_with_trace(
        self, rng: np.random.RandomState
    ) -> Tuple[BlockMoleculeDataExtended, List[_BuildStep]]:
        """Sample a random terminal molecule by uniform random forward
        rollout, returning both the molecule and its build trace (so the
        caller can encode it as an obs without having to replay the
        trace through ``_rebuild_from_trace``)."""
        mol = BlockMoleculeDataExtended()
        trace: List[_BuildStep] = []
        nb = rng.randint(self.min_blocks, self.max_blocks + 1)
        for t in range(nb):
            if t == 0:
                bidx = int(rng.randint(self.NUM_BLOCKS))
                mol = self.mdp.add_block_to(mol, bidx)
                trace.append(_BuildStep(bidx, -1, -1, -1))
            else:
                if len(mol.stems) == 0:
                    break
                bidx = int(rng.randint(self.NUM_BLOCKS))
                sidx = int(rng.randint(len(mol.stems)))
                parent_pos = int(mol.stems[sidx][0])
                parent_atom = int(mol.stems[sidx][1])
                mol = self.mdp.add_block_to(mol, bidx, stem_idx=sidx)
                trace.append(_BuildStep(bidx, parent_pos, parent_atom, sidx))
        return mol, trace

    def _sample_random_terminal(self, rng: np.random.RandomState) -> BlockMoleculeDataExtended:
        return self._sample_random_terminal_with_trace(rng)[0]

    def _real_reward(self, mol: BlockMoleculeDataExtended) -> float:
        if mol.numblocks == 0:
            return 0.0
        return float(self.proxy(mol))

    def _scaled_reward_of_smiles(self, smi: str, mol: BlockMoleculeDataExtended) -> float:
        """Paper formula: R_hat(x) = (max(R(x), R_min) / T) ** beta."""
        cached = self.reward_cache.get(smi)
        if cached is not None:
            return cached
        r = self._real_reward(mol)
        r = max(r, self.SCALE_REWARD_MIN)
        r = (r / self.REWARD_T) ** self.REWARD_EXP
        self.reward_cache[smi] = float(r)
        return float(r)

    def _compute_reward(self, state=None) -> float:
        """Return the paper-scaled reward for ``state``.

        ``state`` may be:
          * ``None``        - score the env's current ``self.mol``.
          * a SMILES string - look up in cache; if uncached, fall back
                              to ``self.mol`` (which the caller in
                              ``gfn_base._get_terminal_reward`` will
                              have just decoded from the same obs).
          * an ``np.ndarray`` obs - decode -> rebuild -> score.
          * a ``BlockMoleculeDataExtended`` - score directly.

        The signature matches ``hypergrid.Environment._compute_reward(state)``
        so ``learning/gfn_base._get_terminal_reward`` works unchanged.
        """
        if state is None:
            if self.mol.numblocks == 0:
                return 0.0
            return self._scaled_reward_of_smiles(self.mol.smiles, self.mol)

        if isinstance(state, str):
            if state == "":
                return 0.0
            cached = self.reward_cache.get(state)
            if cached is not None:
                return cached
            if self.mol.numblocks > 0 and self.mol.smiles == state:
                return self._scaled_reward_of_smiles(state, self.mol)
            return 0.0  # uncached & cannot rebuild from SMILES alone

        if isinstance(state, BlockMoleculeDataExtended):
            if state.numblocks == 0:
                return 0.0
            return self._scaled_reward_of_smiles(state.smiles, state)

        obs = np.asarray(state)
        trace = self._decode_obs_to_trace(obs.flatten() if obs.ndim > 1 else obs)
        if len(trace) == 0:
            return 0.0
        mol = self._rebuild_from_trace(trace)
        return self._scaled_reward_of_smiles(mol.smiles, mol)

    def real_reward(self) -> float:
        return self._real_reward(self.mol)

    def unnormalize(self, r):
        """Inverse of the paper scaling ``R_hat = (R/T)**beta``.

        Take the beta-th root, then multiply by T:
            R = R_hat ** (1/beta) * T
        """
        return r ** (1.0 / self.REWARD_EXP) * self.REWARD_T

    # ============================================================== gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mol = BlockMoleculeDataExtended()
        self.build_trace = []
        return self._get_obs(), {}

    def step(self, action: int):
        if int(action) == self.STOP_ACTION:
            reward = self._compute_reward()
            return self._get_obs(), reward, True, False, {"augmented_rew": 1}

        block_idx, stem_idx = self._decode_action(int(action))
        if self.mol.numblocks >= self.max_blocks:
            return self._get_obs(), 0.0, False, False, {}
        if len(self.mol.stems) == 0 and self.mol.numblocks > 0:
            return self._get_obs(), 0.0, False, False, {}

        if self.mol.numblocks == 0:
            self._add_block_step(block_idx, None)
        else:
            if stem_idx >= len(self.mol.stems):
                return self._get_obs(), 0.0, False, False, {}
            self._add_block_step(block_idx, stem_idx)

        return self._get_obs(), 0.0, False, False, {}

    # ============================================================== action enc
    def _decode_action(self, a: int) -> Tuple[int, int]:
        """Decode a forward action index into (block_idx, stem_idx).

        Convention matches GFNOrg/gflownet's ``mols/gflownet.py``
        (`action = (action % num_blocks, action // num_blocks)`).
        """
        return int(a) % self.NUM_BLOCKS, int(a) // self.NUM_BLOCKS

    def _encode_action(self, block_idx: int, stem_idx: int) -> int:
        return int(block_idx) + int(stem_idx) * self.NUM_BLOCKS

    # ============================================================== obs / state
    def _state_to_obs(self, trace: List[_BuildStep]) -> np.ndarray:
        """Encode a build trace as a fixed-size obs.

        Slots beyond ``len(trace)`` are LEFT AS ZEROS so that
        downstream zero-padding (e.g. ``torch.nn.utils.rnn.pad_sequence``)
        decodes back to the correct trace length. The root step (t=0)
        also has zero-encoded ``parent_pos`` / ``parent_atom``, which
        the decoder interprets as "no parent".
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        for t, step in enumerate(trace[: self.max_blocks]):
            base = t * self._slot_dim
            block_off = base
            ppos_off = block_off + self._block_oh_dim
            patom_off = ppos_off + self._parent_pos_oh_dim
            obs[block_off + step.block_idx] = 1.0
            if step.parent_pos >= 0:
                obs[ppos_off + step.parent_pos] = 1.0
            if step.parent_atom >= 0:
                obs[patom_off + step.parent_atom] = 1.0
        return obs

    def _get_obs(self) -> np.ndarray:
        return self._state_to_obs(self.build_trace)

    def _decode_obs_to_trace(self, obs: np.ndarray) -> List[_BuildStep]:
        """Decode a fixed-size obs back into a build trace.

        Convention: an all-zero slot terminates the trace (matches both
        the env's own EMPTY encoding and any external zero padding,
        e.g. PyTorch's ``pad_sequence``). For the root step the
        ``parent_pos`` / ``parent_atom`` slots are also all zero, which
        we interpret as ``-1`` (no parent). ``stem_idx`` is not stored
        in the obs and is set to ``-1`` (the only consumer is the
        backward mask, which uses the env's stored ``build_trace``).
        """
        trace: List[_BuildStep] = []
        for t in range(self.max_blocks):
            base = t * self._slot_dim
            block_off = base
            ppos_off = block_off + self._block_oh_dim
            patom_off = ppos_off + self._parent_pos_oh_dim
            block_block = obs[block_off : block_off + self._block_oh_dim]
            if float(block_block.sum()) < 0.5:
                break
            block_idx = int(np.argmax(block_block))
            ppos_block = obs[ppos_off : ppos_off + self._parent_pos_oh_dim]
            patom_block = obs[patom_off : patom_off + self._parent_atom_oh_dim]
            parent_pos = (
                int(np.argmax(ppos_block)) if float(ppos_block.sum()) >= 0.5 else -1
            )
            parent_atom = (
                int(np.argmax(patom_block)) if float(patom_block.sum()) >= 0.5 else -1
            )
            trace.append(_BuildStep(block_idx, parent_pos, parent_atom, -1))
        return trace

    def get_state(self, obs):
        """Return the canonical SMILES of the molecule (or list of SMILES for batched obs)."""
        obs = np.asarray(obs)
        if obs.ndim == 1:
            trace = self._decode_obs_to_trace(obs)
            mol = self._rebuild_from_trace(trace)
            return mol.smiles if mol.numblocks > 0 else ""
        out = []
        for row in obs:
            trace = self._decode_obs_to_trace(np.asarray(row))
            mol = self._rebuild_from_trace(trace)
            out.append(mol.smiles if mol.numblocks > 0 else "")
        return out

    def is_initial_state(self, state):
        """An obs is the initial state iff its first slot's block-one-hot is empty."""
        if isinstance(state, str):
            return state == ""
        obs = np.asarray(state)
        if obs.ndim == 0:
            return bool(obs == 0)
        if obs.ndim > 1:
            obs = obs.reshape(-1, obs.shape[-1])[0]
        return self._decode_obs_is_empty(obs)

    def _decode_obs_is_empty(self, obs: np.ndarray) -> bool:
        first_block_block = obs[: self._block_oh_dim]
        return float(first_block_block.sum()) < 0.5

    # ============================================================== modes
    def is_mode(self, reward=None, state=None) -> bool:
        """Track a mode when its **scaled** reward reaches ``mode_r_threshold``.

        With the compatibility default (T=8, beta=10,
        mode_r_raw_threshold=7.5), the scaled threshold is
        ``(7.5 / 8) ** 10`` -- equivalently raw proxy R(x) >= 7.5.  The
        cheapest, least error-prone check is a scalar comparison on the
        trajectory's already-computed scaled reward, which is what the
        rollout collectors pass in here.

        Args:
            reward: Scaled reward of the trajectory (preferred path).
            state:  Optional SMILES string. Used only as a fallback
                    when ``reward`` is None and the molecule has been
                    scored before (lookup in ``self.reward_cache``).
                    The obs-decode / molecule-rebuild fallback present
                    in earlier versions has been removed -- it was
                    expensive on the hot path and silently incorrect
                    for padded obs.

        Returns:
            True iff scaled reward >= ``self.mode_r_threshold``.
        """
        if reward is not None:
            return float(reward) >= self.mode_r_threshold
        if isinstance(state, str) and state in self.reward_cache:
            return self.reward_cache[state] >= self.mode_r_threshold
        return False

    # ============================================================== forward mask
    def _trace_to_mol(self, obs_or_trace) -> Tuple[BlockMoleculeDataExtended, List[_BuildStep]]:
        """Helper: take a 1D obs (or a build trace list) and return (mol, trace)."""
        if isinstance(obs_or_trace, list):
            trace = obs_or_trace
        else:
            trace = self._decode_obs_to_trace(np.asarray(obs_or_trace))
        mol = self._rebuild_from_trace(trace)
        return mol, trace

    def _forward_mask_single(self, obs: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        mol, trace = self._trace_to_mol(obs)
        nb = mol.numblocks

        if nb == 0:
            # First block: any block_idx is valid, stem_idx forced to 0
            mask[: self.NUM_BLOCKS] = True
            return mask

        if nb < self.max_blocks and len(mol.stems) > 0:
            num_stems = min(len(mol.stems), self.max_stems)
            for sidx in range(num_stems):
                start = sidx * self.NUM_BLOCKS
                mask[start : start + self.NUM_BLOCKS] = True

        if nb >= self.min_blocks or len(mol.stems) == 0 or nb >= self.max_blocks:
            mask[self.STOP_ACTION] = True

        return mask

    def get_forward_action_masks(self, obs):
        is_torch = isinstance(obs, torch.Tensor)
        obs_np = obs.detach().cpu().numpy() if is_torch else np.asarray(obs)
        original_ndim = obs_np.ndim

        if original_ndim == 1:
            mask = self._forward_mask_single(obs_np)
        else:
            flat = obs_np.reshape(-1, obs_np.shape[-1])
            masks = np.stack([self._forward_mask_single(row) for row in flat], axis=0)
            if original_ndim == 2:
                mask = masks
            else:
                mask = masks.reshape(*obs_np.shape[:-1], self.action_dim)

        if is_torch:
            return torch.as_tensor(mask, device=obs.device, dtype=torch.bool)
        return mask

    # ============================================================== backward mask
    def _backward_mask_from_trace(self, trace: List[_BuildStep]) -> np.ndarray:
        """The backward mask is computed from the build trace stored on
        ``self`` -- not from the observation -- because it needs the
        ``stem_idx`` recorded at addition time, which is not part of
        the observation. For the purposes of training the backward
        policy this is fine: the env exposes a fresh mask after each
        step, and for off-policy backward sampling the backward
        trajectory generator should hand build traces / molecules
        directly to ``apply_backward_action`` rather than try to mask
        from an arbitrary obs.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        if len(trace) == 0:
            return mask
        mol = self._rebuild_from_trace(trace)
        for pos in range(mol.numblocks):
            if not _node_is_leaf(mol, pos):
                continue
            step = trace[pos]
            if pos == 0:
                if mol.numblocks == 1:
                    a = self._encode_action(step.block_idx, 0)
                    if 0 <= a < self.action_dim - 1:
                        mask[a] = True
                continue
            a = self._encode_action(step.block_idx, step.stem_idx)
            if 0 <= a < self.action_dim - 1:
                mask[a] = True
        return mask

    def get_backward_action_masks(self, obs):
        is_torch = isinstance(obs, torch.Tensor)
        obs_np = obs.detach().cpu().numpy() if is_torch else np.asarray(obs)
        original_ndim = obs_np.ndim

        # We rely on self.build_trace for accurate stem_idx info.
        # If the caller passes a batch of obs, we recompute leaf
        # positions and use action encodings (block_idx, stem_idx=0) as
        # a best-effort fallback for stem_idx (the "first stem"
        # heuristic), which still flags the correct LEAF set even when
        # the original stem_idx is unknown.
        if original_ndim == 1 and self._obs_matches_trace(obs_np, self.build_trace):
            mask = self._backward_mask_from_trace(self.build_trace)
        else:
            if original_ndim == 1:
                mask = self._backward_mask_single_from_obs(obs_np)
            else:
                flat = obs_np.reshape(-1, obs_np.shape[-1])
                masks = np.stack(
                    [self._backward_mask_single_from_obs(row) for row in flat], axis=0
                )
                if original_ndim == 2:
                    mask = masks
                else:
                    mask = masks.reshape(*obs_np.shape[:-1], self.action_dim)

        if is_torch:
            return torch.as_tensor(mask, device=obs.device, dtype=torch.bool)
        return mask

    def _obs_matches_trace(self, obs: np.ndarray, trace: List[_BuildStep]) -> bool:
        # Cheap check: same number of non-empty slots and same block_idx sequence.
        decoded = self._decode_obs_to_trace(obs)
        if len(decoded) != len(trace):
            return False
        return all(
            d.block_idx == s.block_idx
            and d.parent_pos == s.parent_pos
            and d.parent_atom == s.parent_atom
            for d, s in zip(decoded, trace)
        )

    def _backward_mask_single_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """Best-effort backward mask when only an obs is available.

        Without the per-step ``stem_idx`` (which is not encoded in the
        obs), we approximate by encoding leaves with stem_idx=0. This
        at least preserves the *set* of removable blocks; only the
        action-index encoding is approximate. The on-policy code path
        (using ``self.build_trace``) is exact.
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        trace = self._decode_obs_to_trace(obs)
        if len(trace) == 0:
            return mask
        mol = self._rebuild_from_trace(trace)
        for pos in range(mol.numblocks):
            if not _node_is_leaf(mol, pos):
                continue
            step = trace[pos]
            a = self._encode_action(step.block_idx, 0)
            if 0 <= a < self.action_dim - 1:
                mask[a] = True
        return mask

    # ============================================================== backward apply
    def apply_backward_action(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Apply a backward action and return the resulting obs.

        If ``obs`` matches the env's current ``build_trace`` (the common
        case), we use the trace's exact stem_idx to find the leaf to
        remove. Otherwise we fall back to matching by block_idx only
        (and pick the latest leaf with that block_idx).
        """
        obs = np.asarray(obs)
        block_idx, stem_idx_at_add = self._decode_action(int(action))

        if self._obs_matches_trace(obs, self.build_trace):
            trace = list(self.build_trace)
            mol = self.mol
        else:
            trace = self._decode_obs_to_trace(obs)
            mol = self._rebuild_from_trace(trace)

        # Find the leaf position to remove
        target_pos = None
        for pos in reversed(range(mol.numblocks)):
            if not _node_is_leaf(mol, pos):
                continue
            step = trace[pos]
            if step.block_idx != block_idx:
                continue
            # Prefer exact stem match if known; otherwise accept any leaf with this block
            if pos == 0 or step.stem_idx == -1 or step.stem_idx == stem_idx_at_add:
                target_pos = pos
                break
        if target_pos is None:
            return obs.copy()

        new_trace = [s for i, s in enumerate(trace) if i != target_pos]
        # Adjust parent_pos for any step whose parent was at a position > target_pos
        adjusted = []
        for s in new_trace:
            if s.parent_pos > target_pos:
                adjusted.append(_BuildStep(s.block_idx, s.parent_pos - 1, s.parent_atom, s.stem_idx))
            else:
                adjusted.append(s)

        # Update env state if we modified our own trace
        if self._obs_matches_trace(obs, self.build_trace):
            self.build_trace = adjusted
            self.mol = self._rebuild_from_trace(adjusted) if len(adjusted) > 0 else BlockMoleculeDataExtended()

        return self._state_to_obs(adjusted)

    # ============================================================== misc
    def get_num_of_reachable_end_states(self, obs):
        # Without a closed form we return an upper bound based on remaining steps.
        if obs.ndim > 1:
            obs = obs.flatten() if obs.shape[0] == 1 else obs[0]
        trace = self._decode_obs_to_trace(np.asarray(obs))
        nb = len(trace)
        if nb >= self.max_blocks:
            return 1
        # Rough upper bound: at each remaining step, num_blocks * max_stems choices
        return (self.NUM_BLOCKS * self.max_stems) ** (self.max_blocks - nb)

    def dist_func(self, seq1, seq2):
        return levenshtein(str(seq1), str(seq2))

    def _evaluation_records(self, samples, rewards=None):
        """Resolve evaluation inputs to ordered ``(SMILES, scaled reward)`` pairs.

        Observation inputs contain the full block-build trace, so uncached
        rewards can be evaluated exactly.  A bare SMILES does not preserve
        that trace; consequently an uncached SMILES requires its associated
        scaled reward to be supplied explicitly.
        """
        if isinstance(samples, str):
            states = [samples]
            observations = None
        elif isinstance(samples, (list, tuple)) and (
            len(samples) == 0 or all(isinstance(sample, str) for sample in samples)
        ):
            states = list(samples)
            observations = None
        elif (
            isinstance(samples, np.ndarray)
            and samples.ndim == 0
            and samples.dtype.kind in {"U", "S", "O"}
            and isinstance(samples.item(), (str, bytes, np.str_, np.bytes_))
        ):
            sample = samples.item()
            states = [
                sample.decode()
                if isinstance(sample, (bytes, np.bytes_))
                else str(sample)
            ]
            observations = None
        elif (
            isinstance(samples, np.ndarray)
            and samples.ndim == 1
            and samples.dtype.kind in {"U", "S", "O"}
            and all(
                isinstance(sample, (str, bytes, np.str_, np.bytes_))
                for sample in samples
            )
        ):
            states = [
                sample.decode() if isinstance(sample, (bytes, np.bytes_)) else str(sample)
                for sample in samples
            ]
            observations = None
        else:
            observations = np.asarray(samples)
            if observations.size == 0:
                return []
            if observations.ndim == 1:
                observations = observations.reshape(1, -1)
            elif observations.ndim > 2:
                observations = observations.reshape(-1, observations.shape[-1])
            if observations.shape[-1] != self._obs_dim:
                raise ValueError(
                    f"Expected observations with final dimension {self._obs_dim}, "
                    f"got shape {observations.shape}"
                )
            states = []

        resolved_rewards = None
        if rewards is not None:
            resolved_rewards = np.asarray(rewards, dtype=float).reshape(-1)
            expected = len(states) if observations is None else len(observations)
            if len(resolved_rewards) != expected:
                raise ValueError(
                    f"Expected {expected} rewards for {expected} samples, "
                    f"got {len(resolved_rewards)}"
                )

        if observations is not None:
            records = []
            for index, obs in enumerate(observations):
                trace = self._decode_obs_to_trace(obs)
                mol = self._rebuild_from_trace(trace)
                if mol.numblocks == 0:
                    state = ""
                    reward = 0.0 if resolved_rewards is None else resolved_rewards[index]
                else:
                    state = mol.smiles
                    reward = (
                        self._scaled_reward_of_smiles(state, mol)
                        if resolved_rewards is None
                        else resolved_rewards[index]
                    )
                records.append((state, float(reward)))
            return records

        if resolved_rewards is not None:
            return [
                (state, float(reward))
                for state, reward in zip(states, resolved_rewards)
            ]

        records = []
        uncached = []
        for index, state in enumerate(states):
            if state == "":
                records.append((state, 0.0))
                continue

            mol = Chem.MolFromSmiles(state)
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {index}: {state!r}")
            canonical_state = Chem.MolToSmiles(mol, canonical=True)
            reward = self.reward_cache.get(canonical_state)
            if reward is None:
                # Preserve compatibility with callers that populated a cache
                # using the exact input string rather than canonical SMILES.
                reward = self.reward_cache.get(state)
            if reward is None:
                uncached.append((index, state))
                continue
            records.append((canonical_state, float(reward)))

        if uncached:
            examples = ", ".join(repr(state) for _, state in uncached[:3])
            raise ValueError(
                "Cannot determine mode eligibility for uncached SMILES "
                f"({examples}). Pass the corresponding scaled rewards via "
                "get_error(samples, rewards=...) or pass encoded observations."
            )
        return records

    def get_error(
        self,
        samples,
        rewards=None,
        mode_r_raw_threshold=None,
    ) -> Tuple[int, int, int]:
        """Return the three paper-style high-reward molecule mode counts.

        All counts use one inclusive raw proxy cutoff.  By default this is the
        archived-experiment cutoff ``evaluation_mode_r_raw_threshold=7.5``.
        Pass ``mode_r_raw_threshold`` here to evaluate a different shared
        cutoff (for example, the paper's R >= 8 setting) without changing
        rollout-time tracking.
        ``rewards``, when provided, must be the scaled rewards returned by
        this environment.

        Returns:
            ``(unique_smiles, tanimoto_separated, bemis_murcko_scaffolds)``.
            Tanimoto representatives are selected greedily in encounter order
            with the paper cutoff 0.7 and RDKit's default ``RDKFingerprint``.
        """
        if mode_r_raw_threshold is None:
            scaled_threshold = self.evaluation_mode_r_threshold
        else:
            raw_threshold = float(mode_r_raw_threshold)
            clipped_threshold = max(raw_threshold, self.SCALE_REWARD_MIN)
            scaled_threshold = (
                clipped_threshold / self.REWARD_T
            ) ** self.REWARD_EXP

        mode_smiles = [
            state
            for state, reward in self._evaluation_records(samples, rewards=rewards)
            if state and float(reward) >= scaled_threshold
        ]
        return count_mode_metrics(
            mode_smiles,
            tanimoto_similarity_threshold=self.tanimoto_similarity_threshold,
        )

    def render(self):
        smi = self.mol.smiles if self.mol.numblocks > 0 else ""
        print(
            f"Molecule: {smi!r} | numblocks={self.mol.numblocks} | "
            f"free stems={len(self.mol.stems)} | "
            f"build_trace_len={len(self.build_trace)}"
        )


# --- helpers -------------------------------------------------------------------
def _node_is_leaf(mol: BlockMoleculeDataExtended, pos: int) -> bool:
    """True if block at ``pos`` is a leaf of the molecule's block-graph."""
    if mol.numblocks == 1:
        return True
    deg = 0
    for a, b, _, _ in mol.jbonds:
        if int(a) == pos or int(b) == pos:
            deg += 1
            if deg > 1:
                return False
    return deg <= 1


