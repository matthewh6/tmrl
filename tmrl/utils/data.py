from typing import Dict

import json
import os
import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import ogbench
from libero.libero import benchmark
from tmrl.utils.logging import cprint
import gc
import psutil


log = lambda msg, color='bright_green': cprint(msg, color)


class MazeDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], action_len: int, discount: float):
        self.data = data
        self.action_len = action_len
        self.discount = discount

        terminals = data['terminals']
        episode_start = 0
        valid_starts = []

        for t in range(len(terminals)):
            if terminals[t]:
                episode_end = t + 1
                if episode_end - episode_start > action_len:
                    valid_starts.append(torch.arange(episode_start, episode_end - action_len + 1))
                episode_start = episode_end

        self.valid_starts = torch.cat(valid_starts)
        self.size = len(self.valid_starts)
        self.terminal_locs = np.where(terminals == 1)[0]

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        return self.sample(idx)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, (batch_size,))
        starts = self.valid_starts[idxs]
        ends = starts + self.action_len

        obs = self.data['observations'][starts]
        next_obs = self.data['next_observations'][ends - 1]
        actions = torch.stack([self.data['actions'][start:end] for start, end in zip(starts, ends)], dim=0)
        rewards = torch.stack([self.data['rewards'][start:end] for start, end in zip(starts, ends)], dim=0)
        terminals = torch.stack([self.data['terminals'][start:end] for start, end in zip(starts, ends)], dim=0)

        # Use terminal states as goals
        terminal_locs = torch.from_numpy(self.terminal_locs).to(starts.device)
        goal_idxs = []
        for i, end in enumerate(ends):
            candidates = terminal_locs[terminal_locs >= end - 1]
            goal_idx = candidates[0] if len(candidates) > 0 else end - 1
            goal_idxs.append(goal_idx)
            if goal_idx == end:
                rewards[i][-1] = 0.0
        goal_idxs = torch.stack(goal_idxs)
        goals = self.data['observations'][goal_idxs]

        # Treat goals as part of the observation
        obs = torch.cat([obs, goals], dim=-1)
        next_obs = torch.cat([next_obs, goals], dim=-1)

        if torch.any(terminals[:, :-1] == 1.0):
            raise ValueError('Terminals should not be 1.0 in the middle of a chunk')
        terminals = terminals[:, -1:]
        discounts = (self.discount ** torch.arange(self.action_len, device=rewards.device))[None, :]
        R = (rewards * discounts[..., None]).sum(dim=1)

        sample = {
            'obs': obs,
            'actions': actions,
            'next_obs': next_obs,
            'rewards': R,
            'terminals': terminals,
        }

        return sample


class CubeDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], action_len: int, discount: float):
        self.data = data
        self.action_len = action_len
        self.discount = discount

        terminals = data['terminals']
        episode_start = 0
        valid_starts = []

        for t in range(len(terminals)):
            if terminals[t]:
                episode_end = t + 1
                if episode_end - episode_start > action_len:
                    valid_starts.append(torch.arange(episode_start, episode_end - action_len + 1))
                episode_start = episode_end

        self.valid_starts = torch.cat(valid_starts)
        self.size = len(self.valid_starts)
        self.terminal_locs = np.where(terminals == 1)[0]

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        return self.sample(idx)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, (batch_size,))
        starts = self.valid_starts[idxs]
        ends = starts + self.action_len

        obs = self.data['observations'][starts]
        next_obs = self.data['next_observations'][ends - 1]
        goals = self.data['oracle_reps'][starts]
        actions = torch.stack([self.data['actions'][start:end] for start, end in zip(starts, ends)], dim=0)
        rewards = torch.stack([self.data['rewards'][start:end] for start, end in zip(starts, ends)], dim=0)
        terminals = torch.stack([self.data['terminals'][start:end] for start, end in zip(starts, ends)], dim=0)

        if torch.any(terminals[:, :-1] == 1.0):
            raise ValueError('Terminals should not be 1.0 in the middle of a chunk')
        terminals = terminals[:, -1:]
        discounts = (self.discount ** torch.arange(self.action_len, device=rewards.device))[None, :]
        R = (rewards * discounts[..., None]).sum(dim=1)

        return {
            'obs': obs,
            'goals': goals,
            'actions': actions,
            'next_obs': next_obs,
            'rewards': R,
            'terminals': terminals,
        }


def load_data_dict(cfg: DictConfig, training: bool = True):
    """Load data dictionaries from dataset"""
    dataset_name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir

    log(f'Loading dataset: {dataset_dir}')

    if 'ogbench' in dataset_dir:
        env, train_dict, val_dict = ogbench.make_env_and_datasets(
            dataset_name,
            dataset_dir=dataset_dir,
            add_info=True,
        )
        # Ensure oracle_reps are computed (may be missing depending on ogbench version)
        if 'oracle_reps' not in train_dict:
            from ogbench.relabel_utils import add_oracle_reps
            env_name = '-'.join(dataset_name.split('-')[:-2] + dataset_name.split('-')[-1:])
            add_oracle_reps(env_name, env, train_dict)
            add_oracle_reps(env_name, env, val_dict)
    elif 'libero' in dataset_dir:
        train_dict, val_dict = get_libero_dataset(cfg.dataset, training=training, debug=cfg.debug)

    log(f'Loaded dataset: {dataset_name} with {len(train_dict["observations"])} observations')

    # if 'cube' in dataset_name:
    #     # Block x positions: robot has 14 DOF, each cube freejoint adds 7 (3 pos + 4 quat)
    #     block0_x = train_dict['qpos'][:, 14]
        
    #     mask = block0_x <= 0.4
    #     if 'double' in dataset_name:
    #         mask = block0_x <= 0.4
    #         block1_x = train_dict['qpos'][:, 21]
    #         mask = mask & (block1_x <= 0.4)

    #         # # Block positions: robot 14 DOF + 7 per freejoint (3 pos + 4 quat)
    #         # block0_z = train_dict['qpos'][:, 16]
    #         # block1_z = train_dict['qpos'][:, 23]
    #         # block0_xy = train_dict['qpos'][:, 14:16]
    #         # block1_xy = train_dict['qpos'][:, 21:23]
            
    #         # xy_dist = np.linalg.norm(block0_xy - block1_xy, axis=-1)
    #         # z_diff = np.abs(block0_z - block1_z)
            
    #         # # Stacked = close in xy AND one is significantly above the other
    #         # stacked = (xy_dist < 0.05) & (z_diff > 0.03)
    #         # mask = mask & (~stacked)

    #     keep_idxs = np.where(mask)[0]
    #     log(f'Filtering out {(~mask).sum()} trajectories where block_pos x > 0.4')
    #     for key in list(train_dict.keys()):
    #         process = psutil.Process(os.getpid())
    #         old = train_dict[key]
    #         train_dict[key] = np.take(old, keep_idxs, axis=0)
    #         del old
    #         gc.collect()

    if 'rewards' not in train_dict:
        train_dict['rewards'] = -np.ones((len(train_dict['observations']), 1), dtype=np.float32)
        val_dict['rewards'] = -np.ones((len(val_dict['observations']), 1), dtype=np.float32)

    log('Train data dictionary:')
    log('-' * 60)
    for k, v in train_dict.items():
        log(f'{k:12} | {tuple(v.shape)} | {str(v.dtype):>8} | {v.min():8.4f} | {v.max():8.4f}')
    log('-' * 60)

    log('Validation data dictionary:')
    log('-' * 60)
    for k, v in val_dict.items():
        log(f'{k:12} | {tuple(v.shape)} | {str(v.dtype):>8} | {v.min():8.4f} | {v.max():8.4f}')
    log('-' * 60)

    return train_dict, val_dict


def get_libero_dataset(cfg: DictConfig, training: bool = True, debug: bool = False):
    benchmark_name = cfg.name
    benchmark_dict = benchmark.get_benchmark_dict()

    if benchmark_name == 'libero_pretrain':
        benchmark_instance_10 = benchmark_dict['libero_10']()
        benchmark_instance_goal = benchmark_dict['libero_goal']()
        benchmark_instance_object = benchmark_dict['libero_object']()
        benchmark_instance_spatial = benchmark_dict['libero_spatial']()

        num_tasks = (
            benchmark_instance_10.get_num_tasks()
            + benchmark_instance_goal.get_num_tasks()
            + benchmark_instance_object.get_num_tasks()
            + benchmark_instance_spatial.get_num_tasks()
        )
        log(f'Number of tasks in the benchmark {benchmark_name}: {num_tasks}')

        datasets_default_path = cfg.dir
        demo_files = []
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_10.get_task_demonstration(i))
            for i in range(benchmark_instance_10.get_num_tasks())
        ]
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_goal.get_task_demonstration(i))
            for i in range(benchmark_instance_goal.get_num_tasks())
        ]
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_object.get_task_demonstration(i))
            for i in range(benchmark_instance_object.get_num_tasks())
        ]
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_spatial.get_task_demonstration(i))
            for i in range(benchmark_instance_spatial.get_num_tasks())
        ]

    elif benchmark_name == 'libero_100':
        benchmark_instance_10 = benchmark_dict['libero_10']()
        benchmark_instance_90 = benchmark_dict['libero_90']()

        num_tasks = benchmark_instance_10.get_num_tasks() + benchmark_instance_90.get_num_tasks()
        log(f'Number of tasks in the benchmark {benchmark_name}: {num_tasks}')

        datasets_default_path = cfg.dir
        demo_files = []
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_10.get_task_demonstration(i))
            for i in range(benchmark_instance_10.get_num_tasks())
        ]
        demo_files += [
            os.path.join(datasets_default_path, benchmark_instance_90.get_task_demonstration(i))
            for i in range(benchmark_instance_90.get_num_tasks())
        ]

    else:
        benchmark_instance = benchmark_dict[benchmark_name]()
        num_tasks = benchmark_instance.get_num_tasks()
        if debug:
            num_tasks = 2
        log(f'Number of tasks in the benchmark {benchmark_name}: {num_tasks}')

        datasets_default_path = cfg.dir
        demo_files = [
            os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i))
            for i in range(num_tasks)
        ]

    if debug:
        demo_files = demo_files[:2]

    if training:
        train_dict = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'states': [],
            'next_states': [],
            'rewards': [],
            'terminals': [],
            'lang_embeds': [],
        }
        val_dict = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'states': [],
            'next_states': [],
            'rewards': [],
            'terminals': [],
            'lang_embeds': [],
        }
    else:
        train_dict = {'observations': [], 'actions': [], 'states': []}
        val_dict = {'observations': [], 'actions': [], 'states': []}

    log(f'Loaded {len(demo_files)} demo files from {benchmark_name} dataset')

    for demo_file in tqdm(demo_files, desc='Loading demo files'):
        if not os.path.exists(demo_file):
            log(f'Demo file {demo_file} does not exist, skipping')
            continue

        try:
            with h5py.File(demo_file, 'r') as f:
                demos = f['data']
                problem_info = json.loads(demos.attrs['problem_info'])
                lang_instruction = ''.join(problem_info['language_instruction'])
                log(f'Demo language instruction: {lang_instruction}')

                num_demos = len(demos)
                split_idx = int(num_demos * 0.9)
                for i in tqdm(range(num_demos), desc='Loading demos'):
                    demo = demos[f'demo_{i}']

                    if training:
                        if cfg.use_hidden_obs:
                            log(f'Using hidden obs with shape {demo["obs/agentview_patch_tokens"].shape}')
                            obs = np.array(demo['obs/agentview_patch_tokens'])
                        else:
                            log(f'Using visible obs with shape {demo["obs/agentview_embed"].shape}')
                            obs = np.array(demo['obs/agentview_embed'])
                    else:
                        obs = np.flip(demo['obs/agentview_rgb'][:], axis=1)

                    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
                    actions = np.array(demo['actions'])
                    states = np.array(demo['robot_states'])
                    next_states = np.concatenate([states[1:], states[-1:]], axis=0)
                    rewards = np.array(demo['rewards'])
                    terminals = np.array(demo['dones'])
                    lang_embeds = np.array(demo['lang_embeds'])

                    data_dict = train_dict if i < split_idx else val_dict
                    data_dict['observations'].extend(obs)
                    data_dict['actions'].extend(actions)
                    data_dict['states'].extend(states)

                    if training:
                        data_dict['next_observations'].extend(next_obs)
                        data_dict['next_states'].extend(next_states)
                        data_dict['rewards'].extend(rewards)
                        data_dict['terminals'].extend(terminals)
                        data_dict['lang_embeds'].extend(lang_embeds)

                    torch.cuda.empty_cache()

        except (OSError, KeyError) as e:
            log(f'Skipping corrupted file: {demo_file} ({e})')
            continue

    for d in [train_dict, val_dict]:
        for key in d.keys():
            if key == 'langs':
                d[key] = np.array(d[key], dtype=str)
            else:
                d[key] = np.array(d[key], dtype=np.float32)

    return train_dict, val_dict
