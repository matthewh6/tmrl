from __future__ import annotations

import numpy as np
import torch
from typing import List

from tmrl.utils.common import to_tensor
from tmrl.utils.logging import cprint


log = lambda msg, color='bright_cyan': cprint(msg, color)


def flush_traj_to_buffer(traj_data, env_id, terminated, rb):
    """Add completed trajectory to the appropriate replay buffer."""
    traj_len = len(traj_data[env_id]['action'])

    if traj_len == 0:
        return
    is_success = bool(terminated[env_id] if hasattr(terminated, '__getitem__') else terminated)

    rewards_arr = np.array(traj_data[env_id]['reward'], dtype=np.float32)
    terminals_arr = np.zeros(traj_len, dtype=np.float32)
    if is_success:
        terminals_arr[-1] = 1.0

    target_rb = rb.success_buffer if (is_success and rb.success_buffer is not None) else rb
    timesteps = traj_data[env_id]['timesteps'] if traj_data[env_id]['timesteps'] else None
    context_noises = traj_data[env_id]['context_noises'] if traj_data[env_id]['context_noises'] else None
    target_rb.add(
        env_idx=env_id,
        obs=traj_data[env_id]['obs'],
        action=traj_data[env_id]['action'],
        reward=rewards_arr,
        terminal=terminals_arr,
        timesteps=timesteps,
        context_noises=context_noises,
    )
    for key in traj_data[env_id]:
        traj_data[env_id][key] = []


class TrajectoryReplayBuffer:
    """
    LIBERO trajectory replay buffer (store by adding trajectories)
    """

    def __init__(
        self,
        method: str,
        dataset_name: str,
        capacity: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        action_len: int,
        context_dim: int,
        goal_dim: int = 0,
        use_success_buffer: bool = False,
        discount: float = 0.995,
    ):
        super().__init__()
        self.method = method

        self.dataset_name = dataset_name
        self.capacity = capacity
        self.n_envs = n_envs

        self.obs_dim = obs_dim
        self.action_len = action_len
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.goal_dim = goal_dim

        self.discount = discount
        self.device = torch.device('cpu')  # store data on cpu by default

        # log(f"discount: {self.discount}")
        # log(f"action_len: {self.action_len}")

        self.observations = np.empty(
            (self.capacity, self.n_envs, obs_dim),
            dtype=np.float32,
        )

        if self.method == 'tmrl':
            self.timesteps = np.empty(
                (self.capacity, self.n_envs, 1),  # TODO: for now
                dtype=np.float32,
            )

            self.context_noises = np.empty(
                (self.capacity, self.n_envs, self.context_dim),
                dtype=np.float32,
            )
        else:
            self.timesteps = None
            self.context_noises = None

        self.actions = np.empty(
            (self.capacity, self.n_envs, self.action_dim),
            dtype=np.float32,
        )

        self.rewards = np.empty(
            (self.capacity, self.n_envs),
            dtype=np.float32,
        )

        self.terminals = np.empty(
            (self.capacity, self.n_envs),
            dtype=np.float32,
        )

        self.idx = [0 for _ in range(self.n_envs)]
        self.traj_start_idx = [[] for _ in range(self.n_envs)]

        # Cached flat array of valid (env_idx, traj_start, traj_end) for fast sampling
        self._valid_ranges = np.empty((0, 3), dtype=np.int64)
        self._valid_ranges_dirty = True

        if use_success_buffer:
            self.success_buffer = TrajectoryReplayBuffer(
                method=method,
                dataset_name=dataset_name,
                capacity=capacity,
                n_envs=n_envs,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_len=action_len,
                context_dim=context_dim,
                goal_dim=goal_dim,
                use_success_buffer=False,
                discount=discount,
            )
        else:
            self.success_buffer = None

    def add(
        self,
        env_idx: int,
        obs: List[np.ndarray],  # [T, obs_dim]
        action: List[np.ndarray],  # [T, action_dim]
        reward: List[float],  # [T]
        terminal: List[float],  # [T]
        timesteps: List[float] | None = None,  # [T]
        context_noises: List[np.ndarray] | None = None,  # [T, 2048]
    ) -> None:
        """
        Adds a single trajectory to the replay buffer.

        All inputs should correspond to a single trajectory of length T.
        """
        T = len(reward)
        curr_idx = self.idx[env_idx]

        self.observations[curr_idx : curr_idx + T, env_idx] = np.stack(obs, axis=0)
        self.actions[curr_idx : curr_idx + T, env_idx] = np.stack(action, axis=0)
        self.rewards[curr_idx : curr_idx + T, env_idx] = np.stack(reward, axis=0)
        self.terminals[curr_idx : curr_idx + T, env_idx] = np.stack(terminal, axis=0)

        if self.method == 'tmrl':
            self.timesteps[curr_idx : curr_idx + T, env_idx] = np.stack(timesteps, axis=0)
            self.context_noises[curr_idx : curr_idx + T, env_idx] = np.stack(context_noises, axis=0)

        self.traj_start_idx[env_idx].append(curr_idx)
        self.idx[env_idx] += T
        self._valid_ranges_dirty = True

    def _rebuild_valid_ranges(self, H: int):
        """Rebuild the flat array of valid (env_idx, traj_start, traj_end) tuples."""
        ranges = []
        for env_idx, trajs in enumerate(self.traj_start_idx):
            if len(trajs) == 0:
                continue
            starts = np.array(trajs, dtype=np.int64)
            ends = np.concatenate([starts[1:], [self.idx[env_idx]]])
            mask = (ends - starts) >= H
            if mask.any():
                valid_starts = starts[mask]
                valid_ends = ends[mask]
                env_col = np.full(len(valid_starts), env_idx, dtype=np.int64)
                ranges.append(np.stack([env_col, valid_starts, valid_ends], axis=1))
        if ranges:
            self._valid_ranges = np.concatenate(ranges, axis=0)
        else:
            self._valid_ranges = np.empty((0, 3), dtype=np.int64)
        self._valid_ranges_dirty = False

    def sample(self, batch_size: int):
        """
        Samples a batch of sequences from the buffer (and success buffer if available).

        Returns a chunk of observations, actions, returns, next observations, and terminals.
            s_t, a_t:t+H, R_t:t+H, s'_t+H-1, terminal_t+H-1
        """
        device = torch.device('cpu')
        H = self.action_len

        if self.success_buffer is None:
            sampling_buffers = [self]
            buffer_batch_size = batch_size
        else:
            if self._valid_ranges_dirty:
                self._rebuild_valid_ranges(H)
            if self.success_buffer._valid_ranges_dirty:
                self.success_buffer._rebuild_valid_ranges(H)

            normal_has_data = len(self._valid_ranges) > 0
            success_has_data = len(self.success_buffer._valid_ranges) > 0

            if not success_has_data:
                sampling_buffers = [self]
                buffer_batch_size = batch_size
            elif not normal_has_data:
                sampling_buffers = [self.success_buffer]
                buffer_batch_size = batch_size
            else:
                sampling_buffers = [self, self.success_buffer]
                buffer_batch_size = batch_size // 2
                log('Sampling from both normal and success buffer')

        batches = []

        for buffer in sampling_buffers:
            if buffer._valid_ranges_dirty:
                buffer._rebuild_valid_ranges(H)

            vr = buffer._valid_ranges
            # Sample random trajectory ranges (with replacement)
            chosen = vr[np.random.randint(len(vr), size=buffer_batch_size)]
            batch_env_idxs = chosen[:, 0]
            traj_starts = chosen[:, 1]
            traj_ends = chosen[:, 2]

            # Vectorized random start within each trajectory
            max_offsets = traj_ends - traj_starts - H  # inclusive upper bound
            rand_offsets = (np.random.random(buffer_batch_size) * (max_offsets + 1)).astype(np.int64)
            batch_start_idxs = traj_starts + rand_offsets

            batch_env_idxs_t = torch.tensor(batch_env_idxs, device=device)
            batch_start_idxs_t = torch.tensor(batch_start_idxs, device=device)

            t_offsets = torch.arange(H, device=device)
            batch_idxs = batch_start_idxs_t[:, None] + t_offsets[None, :]

            obs = to_tensor(buffer.observations[batch_start_idxs, batch_env_idxs])
            next_obs = to_tensor(buffer.observations[batch_start_idxs + H, batch_env_idxs])
            actions = buffer.actions[batch_idxs, batch_env_idxs_t[:, None]]
            if buffer.method == 'tmrl':
                timesteps = buffer.timesteps[batch_start_idxs, batch_env_idxs]
                context_noises = buffer.context_noises[batch_start_idxs, batch_env_idxs]
            else:
                timesteps = None
                context_noises = None

            rewards = buffer.rewards[batch_idxs, batch_env_idxs_t[:, None]]
            terminals = buffer.terminals[batch_start_idxs + H - 1, batch_env_idxs][:, None]
            discounts = (float(buffer.discount) ** np.arange(H))[None, :]
            R = (rewards * discounts).sum(axis=1, keepdims=True)

            batches.append({
                'obs': obs,
                'actions': to_tensor(actions),
                'timesteps': to_tensor(timesteps),
                'context_noises': to_tensor(context_noises),
                'rewards': to_tensor(R),
                'next_obs': next_obs,
                'terminals': to_tensor(terminals),
            })

        if len(batches) == 2:
            combined = {}
            for key in ['actions', 'rewards', 'terminals', 'timesteps', 'context_noises']:
                if key in ['timesteps', 'context_noises'] and self.method != 'tmrl':
                    combined[key] = None
                    continue
                combined[key] = torch.cat([batches[0][key], batches[1][key]], dim=0)
            combined['obs'] = torch.cat([batches[0]['obs'], batches[1]['obs']], dim=0)
            combined['next_obs'] = torch.cat([batches[0]['next_obs'], batches[1]['next_obs']], dim=0)
            return combined

        return batches[0]
