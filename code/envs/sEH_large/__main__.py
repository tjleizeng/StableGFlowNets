"""Smoke test: ``python -m envs.sEH_large``.

Verifies the new molecule-graph MDP env from Bengio et al. (2021)
Sec. A.3 — 72 unique blocks / 105 actions per stem / max 8 blocks.
"""

import numpy as np
from tqdm import trange

from . import Environment


def main():
    np.random.seed(42)

    env = Environment(max_stems=40)

    print("Testing sEH_large environment (paper Sec. A.3)...")
    print(f"  unique blocks (SMILES) : {env.num_unique_blocks}        (expect 72)")
    print(f"  num_blocks library     : {env.NUM_BLOCKS}        (expect 105)")
    print(f"  max_blocks             : {env.max_blocks}          (expect 8)")
    print(f"  min_blocks             : {env.min_blocks}          (expect 2)")
    print(f"  max_stems              : {env.max_stems}")
    print(f"  forward action_dim     : {env.action_dim}      (= 105 * max_stems + 1)")
    print(f"  obs dim                : {env.observation_space.shape[0]}")
    print(f"  reward T               : {env.REWARD_T}        (paper: 8)")
    print(f"  reward beta            : {env.REWARD_EXP}       (paper: 10)")
    print(f"  reward R_min           : {env.SCALE_REWARD_MIN}      (paper: 0.01)")
    print(f"  mode raw R threshold   : {env.mode_r_raw_threshold}")
    print(f"  mode scaled-R thr      : {env.mode_r_threshold:.6g}")
    print(f"  evaluation raw R thr   : {env.evaluation_mode_r_raw_threshold}")
    print(f"  Tanimoto mode cutoff   : {env.tanimoto_similarity_threshold} (paper: 0.7)")
    assert env.num_unique_blocks == 72
    assert env.NUM_BLOCKS == 105
    assert env.action_dim == 105 * env.max_stems + 1

    rs = []
    smiles_seen = set()
    modes_found = 0
    num_trajs = 200
    for _ in trange(num_trajs):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_forward_action_masks(obs)
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            a = int(valid[np.random.randint(len(valid))])
            obs, r, done, _, _ = env.step(a)
        rs.append(r)
        smi = env.mol.smiles if env.mol.numblocks > 0 else ""
        smiles_seen.add(smi)
        if env.is_mode(reward=r):
            modes_found += 1

    rs = np.array(rs)
    print()
    print(f"  unique SMILES sampled  : {len(smiles_seen)} / {num_trajs}")
    print(f"  modes found            : {modes_found}")
    print(f"  mean scaled reward     : {np.mean(rs):.6f}")
    print(f"  max  scaled reward     : {np.max(rs):.6f}")


if __name__ == "__main__":
    main()
