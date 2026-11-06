"""Local flow matching on the environment's explicit-stop state graph.

Edge flows retain the existing checkpoint parameterization:
    log F(s, a) = log F(s) + log P_F(a | s).
The loss matches summed incoming and outgoing flows at every visited
non-source state, and matches the sampled stop edge to its terminal reward.
See docs/FM_CORRECTION.md for the graph semantics and molecular limitation.
"""

import numpy as np
import torch

from learning.gfn_base import GFlowNetBase


class FlowMatching(GFlowNetBase):
    """State-local FM, with exact parent-action sums and reward boundaries.

    Distinct actions leading from the same parent to the same child contribute
    separately. No backward-policy probability or trajectory-balance residual
    is used. The source has no conservation residual; its outgoing flow learns
    through the conservation equations at its children.
    """

    def __init__(self, env, **kwargs):
        kwargs['use_state_flow'] = True
        super().__init__(env, **kwargs)
        self._fm_environment_module = type(env.unwrapped).__module__
        self._fm_has_parent_provider = callable(
            getattr(env.unwrapped, 'get_parent_edges', None)
        )
        supported = {
            'envs.regular_tree', 'envs.hypergrid', 'envs.rna14',
            'envs.sEH', 'envs.sEH_large',
        }
        if (self._fm_environment_module not in supported
                and not self._fm_has_parent_provider):
            raise NotImplementedError(
                'Exact FM requires a supported environment or '
                'get_parent_edges(obs) returning all (parent_obs, forward_action) '
                'pairs, including distinct actions with the same endpoints.'
            )

    def _parent_edges(self, obs):
        """Enumerate actual forward parents without mutating the environment."""
        env = self.env.unwrapped
        obs = np.asarray(obs, dtype=np.float32)
        if self._fm_has_parent_provider:
            return list(env.get_parent_edges(obs))
        if self._fm_environment_module == 'envs.sEH_large':
            # Observations encode an ORDERED build trace. Forward steps append
            # to this trace, so removing an arbitrary graph leaf is not an
            # inverse in the represented state graph. Reconstruct the exact
            # prefix, and enumerate every stem action that appends this slot.
            trace = env._decode_obs_to_trace(obs)
            if not trace:
                return []
            prefix = trace[:-1]
            parent = env._state_to_obs(prefix)
            last = trace[-1]
            if not prefix:
                return [(parent, env._encode_action(last.block_idx, 0))]
            molecule = env._rebuild_from_trace(prefix)
            target = (last.parent_pos, last.parent_atom)
            return [
                (parent.copy(), env._encode_action(last.block_idx, stem_index))
                for stem_index, stem in enumerate(molecule.stems[:env.max_stems])
                if tuple(map(int, stem)) == target
            ]
        # Tree/Hypergrid inverse actions have the same index as the forward
        # edge. RNA (and legacy string sEH) use separate prepend/append actions;
        # retain both even when they lead to identical parent observations.
        mask = np.asarray(env.get_backward_action_masks(obs), dtype=bool)
        return [
            (env.apply_backward_action(obs.copy(), int(action)), int(action))
            for action in np.flatnonzero(mask)
            if int(action) != self.action_dim - 1
        ]

    def _log_edge_flows(self, observations):
        """Return normalized, masked log edge flows for a batch of states."""
        mask = torch.as_tensor(
            self.env.unwrapped.get_forward_action_masks(observations),
            dtype=torch.bool, device=self.device,
        )
        if not bool(mask.any(dim=-1).all()):
            raise ValueError('FM received a state without any valid outgoing edge')
        logits = self.forward_policy.get_logits(observations).clamp(-50, 50)
        log_pf = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        # Do not clamp normalized log probabilities: doing so changes the sum
        # of outgoing edge flows and no longer describes the sampling policy.
        return self.logF(observations).reshape(-1, 1) + log_pf

    def loss_per_trajectory(self, batch_obs, batch_acts, batch_rewards):
        """Sum local squared log-flow residuals for each unpadded trajectory.

        The explicit stop transition leads to a terminal copy of the last
        observation. That copy has outgoing flow equal to its positive reward;
        its incoming flow is the stop edge. This avoids confusing environments
        whose stop returns the same observation with an actual self-loop.
        """
        if not batch_obs or not (len(batch_obs) == len(batch_acts) == len(batch_rewards)):
            raise ValueError('FM requires a nonempty batch of aligned trajectories')
        observations, observation_index = [], {}

        def add_observation(obs):
            array = np.asarray(obs, dtype=np.float32)
            key = array.tobytes()
            if key not in observation_index:
                observation_index[key] = len(observations)
                observations.append(array.copy())
            return observation_index[key]

        source = self.initial_state_obs.detach().cpu().numpy()
        trajectories, parents = [], {}
        for obs, acts in zip(batch_obs, batch_acts):
            obs_np = torch.as_tensor(obs).detach().cpu().numpy()
            acts_np = torch.as_tensor(acts).detach().cpu().numpy().reshape(-1)
            if obs_np.ndim != 2 or len(obs_np) != len(acts_np) or not len(obs_np):
                raise ValueError('FM requires complete, unpadded trajectories')
            if (acts_np[-1] != self.action_dim - 1
                    or np.any(acts_np[:-1] == self.action_dim - 1)):
                raise ValueError('FM requires exactly one final stop action')
            if not np.array_equal(obs_np[0], source):
                raise ValueError('FM trajectories must begin at the source')
            indices = []
            for state in obs_np:
                state_index = add_observation(state)
                indices.append(state_index)
                if np.array_equal(state, source) or state_index in parents:
                    continue
                incoming = self._parent_edges(state)
                if not incoming:
                    raise ValueError('FM found a non-source state with no parent edges')
                parents[state_index] = [
                    (add_observation(parent), int(action))
                    for parent, action in incoming
                ]
            trajectories.append(indices)

        state_tensor = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=self.device
        )
        log_edges = self._log_edge_flows(state_tensor)
        local_losses = {}
        for state_index, incoming in parents.items():
            parent_indices, actions = zip(*incoming)
            incoming_edges = log_edges[list(parent_indices), list(actions)]
            if not bool(torch.isfinite(incoming_edges).all()):
                raise ValueError('FM parent enumeration returned an invalid forward action')
            log_in = torch.logsumexp(incoming_edges, dim=0)
            log_out = torch.logsumexp(log_edges[state_index], dim=0)
            local_losses[state_index] = (log_in - log_out).square()

        losses = []
        for indices, reward in zip(trajectories, batch_rewards):
            reward = torch.as_tensor(reward, dtype=log_edges.dtype,
                                     device=self.device).reshape(())
            if not bool(torch.isfinite(reward)) or not bool(reward > 0):
                raise ValueError('FM requires a finite, strictly positive terminal reward')
            log_reward = reward.log() / self.reward_temperature
            boundary = (log_edges[indices[-1], -1] - log_reward).square()
            # Source conservation is excluded, even for immediate stop at s0.
            terms = [local_losses[index] for index in indices if index in local_losses]
            losses.append(boundary + (torch.stack(terms).sum() if terms else 0))
        return torch.stack(losses)

    def train(self):
        """Use the same local FM objective for latest-data and replay updates."""
        for update in range(self.gradient_steps):
            batch = (self.memory.latest_sample(self.train_freq, 0) if update == 0
                     else self.sample_from_memory(self.batch_size))
            batch_obs, batch_acts, _, batch_rewards, _ = batch
            losses = self.loss_per_trajectory(batch_obs, batch_acts, batch_rewards)
            if not bool(torch.isfinite(losses).all()):
                raise ValueError(f'Invalid FM loss: {losses.detach()}')
            self.logger['loss_means'].append(losses.mean().detach().item())
            self.logger['loss_max'].append(losses.max().detach().item())
            self.logger['loss_std'].append(
                losses.std(unbiased=losses.numel() > 1).detach().item()
            )
            self.logger['log_Z'].append(self.get_logZ().detach().item())
            for optimizer in self.GFN_optim:
                optimizer.zero_grad()
            losses.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.forward_policy.parameters(), 1e1)
            torch.nn.utils.clip_grad_norm_(self.logF.parameters(), 1e1)
            # PB remains in checkpoints for compatibility but has no FM gradient.
            for optimizer in self.GFN_optim:
                optimizer.step()
            self._n_updates += 1
