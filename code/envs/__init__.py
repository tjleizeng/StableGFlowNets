from gymnasium.envs.registration import register

register(
      id="gfn_challenges/gmm-hard",
      entry_point="gym_envs.gmm_hard:Environment",
      max_episode_steps=1001,
)

register(
      id="gfn_challenges/pusher-mild",
      entry_point="gym_envs.pusher_mild:Environment",
      max_episode_steps=1001,
)

register(
      id="gfn_challenges/pusher-simple",
      entry_point="gym_envs.pusher_simple:Environment",
      max_episode_steps=1001,
)


register(
      id="gfn_challenges/hypergrid-simple",
      entry_point="gym_envs.hypergrid_simple:Environment",
      max_episode_steps=1001,     # higher than max steps
)

register(
      id="gfn_challenges/hypergrid-mild",
      entry_point="gym_envs.hypergrid_mild:Environment",
      max_episode_steps=1001,     # higher than max steps
)

register(
      id="gfn_challenges/hypergrid-hard",
      entry_point="gym_envs.hypergrid_hard:Environment",
      max_episode_steps=1001,     # higher than max steps
)

register(
      id="gfn_challenges/molecular-generation",
      entry_point="gym_envs.molecular_generation:Environment",
      max_episode_steps=1001,     # higher than max steps
)

register(
      id="gfn_challenges/rna14",
      entry_point="gym_envs.rna14:Environment",
      max_episode_steps=1001,     # higher than max steps
)

register(
      id="gfn_challenges/seh_large",
      entry_point="envs.sEH_large:Environment",
      max_episode_steps=1001,     # higher than max steps (max_blocks=8 + stop)
)
