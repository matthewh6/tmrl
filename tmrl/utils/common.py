import numpy as np
import torch
from pathlib import Path
import numpy as np
import torch
import yaml
from hydra.utils import instantiate
import random
from tmrl.utils.logging import cprint
import itertools
import os
import psutil
import time

log = lambda msg, color='bright_cyan': cprint(msg, color)


def set_seed(seed):
    log(f'Setting seed to: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_memory_kill_switch(avail_thresh=10.0):
    """Kills program if available memory is below threshold to avoid memory overflows."""
    try:
        if psutil.virtual_memory().available * 100 / psutil.virtual_memory().total < avail_thresh:
            print("Current memory usage of {}% surpasses threshold, killing program...".format(
                psutil.virtual_memory().percent
            ))
            time.sleep(10 * np.random.rand())   # avoid that all processes get killed at once
            if psutil.virtual_memory().available * 100 / psutil.virtual_memory().total < avail_thresh:
                exit(0)
    except FileNotFoundError:   # seems to happen infrequently
        pass


def to_tensor(
    data,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    if data is None:
        return None

    def ensure_float32(arr):
        if str(arr.dtype) == 'bfloat16' or arr.dtype == np.float64:
            return arr.astype(np.float32)
        return arr

    if isinstance(data, torch.Tensor):
        # Avoid bf16->f16 conversion on GPUs/CPUs that lack support; stay in fp32 instead.
        if str(data.dtype) == 'torch.bfloat16':
            data = data.float()
        return data.to(device=device)
    elif isinstance(data, np.ndarray):
        data = ensure_float32(data)
        return torch.as_tensor(data).to(device)
    elif isinstance(data, list):
        data = np.array(data)
        data = ensure_float32(data)
        return torch.as_tensor(data).to(device)
    else:
        data = np.array(data, copy=True)
        data = ensure_float32(data)
        return torch.as_tensor(data).to(device)


def to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if np.isscalar(x):
        x = np.array([x], dtype=dtype)
    else:
        x = np.asarray(x, dtype=dtype)
    return x


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def load_checkpoint(ckpt_file, device='cuda'):
    """Load pre-trained model from a checkpoint and freeze."""
    ckpt_dir = Path(ckpt_file)

    # Load config
    cfg_path = ckpt_dir / 'config.yaml'
    with open(cfg_path, 'r') as f:
        ckpt_cfg = yaml.safe_load(f)

    # Load checkpoint
    ckpt_path = ckpt_dir / 'last.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Instantiate and load model
    model = instantiate(ckpt_cfg['model'])
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    log(f'Successfully loaded model from {ckpt_path}')

    return model


def merge_batches(offline_batch, online_batch):
    if offline_batch is None:
        return online_batch
    if online_batch is None:
        return offline_batch

    merged = {}
    all_keys = set(online_batch.keys())  # | set(offline_batch.keys())

    for k in all_keys:
        v_off = offline_batch.get(k)
        v_on = online_batch.get(k)

        if isinstance(v_off, dict) and isinstance(v_on, dict):
            # Recursively merge nested dicts
            merged[k] = merge_batches(v_off, v_on)
        elif v_off is None:
            merged[k] = v_on
        elif v_on is None:
            merged[k] = v_off
        else:
            # Both exist and are tensors (or compatible)
            if torch.is_tensor(v_off) and torch.is_tensor(v_on):
                merged[k] = torch.cat([v_off, v_on], dim=0)
            else:
                # Fallback: prefer online value if not tensors
                merged[k] = v_on

    return merged


def subsequence_dtw(sequence, query):
    """
    Compute the minimum DTW distance between query and any contiguous subsequence of sequence.

    Args:
        sequence: np.ndarray of shape [T, D]
        query: np.ndarray of shape [L, D]

    Returns:
        (start_idx, end_idx): indices of that subsequence in sequence
        min_dist: float, the DTW distance
    """
    T, D = sequence.shape
    L, _ = query.shape

    # DTW matrix
    dtw = np.full((L + 1, T + 1), np.inf)
    dtw[0, :] = 0  # allow starting anywhere

    def euclidean(a, b):
        return np.linalg.norm(a - b)

    for i in range(1, L + 1):
        for j in range(1, T + 1):
            cost = euclidean(query[i - 1], sequence[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],  # insertion
                dtw[i, j - 1],  # deletion
                dtw[i - 1, j - 1],  # match
            )

    # find minimal endpoint
    min_dist = np.min(dtw[L, 1:])
    best_end = np.argmin(dtw[L, 1:]) + 1

    # backtrack to find start
    i, j = L, best_end
    path = [(i, j)]
    while i > 0:
        if j == 0:
            break
        choices = [dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]]
        move = np.argmin(choices)
        if move == 0:  # up
            i -= 1
        elif move == 1:  # left
            j -= 1
        else:  # diagonal
            i -= 1
            j -= 1
        path.append((i, j))
    path = np.array(path[::-1])

    start_idx = int(path[0, 1] - 1)
    end_idx = int(path[-1, 1])

    return float(min_dist), (start_idx, end_idx)


def load_pi0_model(cfg):
    """
    Load the pi0 policy model from the checkpoint directory.

    Args:
        cfg: Config

    Returns:
        The pi0 policy model.
    """
    from dsrl_openpi.policies import policy_config as _policy_config
    from dsrl_openpi.training import config as _config

    # If cfg.ckpt is already a full path, use it directly
    if os.path.isabs(cfg.ckpt):
        checkpoint = cfg.ckpt
    else:
        checkpoint = os.path.join(cfg.paths.ckpt_dir, cfg.ckpt)

    if 'libero' in checkpoint:
        if 'csp' in checkpoint:
            config = _config.get_config('tmpi0_libero')
        elif 'postbc' in checkpoint:
            config = _config.get_config('postbc_libero')
        else:
            config = _config.get_config('pi0_libero')
    else:
        if 'tmpi0' in checkpoint:
            if 'lora' in checkpoint:
                config = _config.get_config('tmpi0_lora_bridge_1_cam')
            elif 'droid' in checkpoint:
                config = _config.get_config('tmpi0_droid')
            else:
                config = _config.get_config('tmpi0_bridge_1_cam')
        else:
            if 'lora' in checkpoint:
                config = _config.get_config('pi0_lora_bridge_1_cam')
            elif 'droid' in checkpoint:
                if 'cspi' in checkpoint:
                    config = _config.get_config('tmpi0_droid')
                else:
                    config = _config.get_config('pi0_droid')
            else:
                config = _config.get_config('pi0_bridge_1_cam')

    policy = _policy_config.create_trained_policy(config, checkpoint)

    return policy


def create_optimizers(model, cfg):
    """Create optimizers for the SAC agent.

    Args:
        model: The OfflineRLAgent model
        cfg: Configuration object with optimizer settings

    Returns:
        dict: Dictionary of optimizers
    """

    optimizers = {
        'qf': torch.optim.Adam(itertools.chain(*[q.parameters() for q in model.critic_ensemble]), lr=cfg.optimizer.lr),
        'actor': torch.optim.Adam(model.actor.parameters(), lr=cfg.optimizer.lr),
    }

    if cfg.model.use_noise_critic and model.method in ['dsrl', 'tmrl', 'tmrl_cfg']:
        optimizers['qf_z'] = torch.optim.Adam(
            itertools.chain(*[q.parameters() for q in model.noise_critic_ensemble]), lr=cfg.optimizer.lr
        )

    if cfg.use_autotune:
        optimizers['alpha_a'] = torch.optim.Adam([model.log_alpha_a], lr=cfg.optimizer.lr)
        if model.method in ['tmrl', 'tmrl_cfg']:
            optimizers['alpha_t'] = torch.optim.Adam([model.log_alpha_t], lr=cfg.optimizer.lr)
            optimizers['alpha_c'] = torch.optim.Adam([model.log_alpha_c], lr=cfg.optimizer.lr)

    return optimizers


def infer_actions(cfg, model, raw_obs, action_noise, timesteps, context_noise):
    """Run diffusion policy inference to get chunked actions."""
    if cfg.method not in ['tmrl', 'dsrl']:
        return action_noise[:, :cfg.model.action_dim]

    if cfg.method == 'tmrl':
        if cfg.fixed_tcont_context is not None:
            assert 0.0 <= cfg.fixed_tcont_context <= 1.0, (
                f'fixed_tcont_context out of range: [{cfg.fixed_tcont_context}]'
            )
            tcont_context = np.full_like(timesteps, cfg.fixed_tcont_context)
        else:
            tcont_context = (timesteps * cfg.timestep_max / 2) + (cfg.timestep_max / 2)
        return model.dp.infer(obs=raw_obs, action_noise=action_noise, tcont_context=tcont_context, context_noise=context_noise)['actions']
    else:  # dsrl
        return model.dp.infer(obs=raw_obs, action_noise=action_noise)['actions']
