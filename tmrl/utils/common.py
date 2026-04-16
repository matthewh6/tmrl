from pathlib import Path
import os
import random
import itertools

import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from tmrl.utils.logging import cprint

log = lambda msg, color='bright_cyan': cprint(msg, color)


def set_seed(seed: int) -> None:
    log(f'Setting seed to: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(
    data: object,
    device: torch.device | str = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> torch.Tensor | None:
    if data is None:
        return None

    def ensure_float32(arr: np.ndarray) -> np.ndarray:
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


def to_numpy(x: object, dtype: object = None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if np.isscalar(x):
        x = np.array([x], dtype=dtype)
    else:
        x = np.asarray(x, dtype=dtype)
    return x


def to_device(x: object, device: torch.device | str) -> object:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def load_checkpoint(ckpt_dir: str | os.PathLike[str] | Path, device: str | torch.device = 'cuda') -> object:
    """Load pre-trained model from a checkpoint directory and freeze."""
    # Load config
    ckpt_dir = Path(ckpt_dir)
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


def merge_batches(offline_batch: dict[str, object] | None, online_batch: dict[str, object] | None) -> dict[str, object]:
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


def load_pi0_model(cfg: object) -> object:
    """
    Load the pi0 policy model from the checkpoint directory.

    Args:
        cfg: Config

    Returns:
        The pi0 policy model.
    """
    from tmrl_openpi.policies import policy_config as _policy_config
    from tmrl_openpi.training import config as _config

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


def create_optimizers(model: object, cfg: object) -> dict[str, torch.optim.Optimizer]:
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

    if cfg.use_autotune:
        optimizers['alpha_a'] = torch.optim.Adam([model.log_alpha_a], lr=cfg.optimizer.lr)
        if model.method == 'tmrl':
            optimizers['alpha_t'] = torch.optim.Adam([model.log_alpha_t], lr=cfg.optimizer.lr)
            optimizers['alpha_c'] = torch.optim.Adam([model.log_alpha_c], lr=cfg.optimizer.lr)

    return optimizers


def infer_actions(
    cfg: object,
    model: object,
    raw_obs: object,
    action_noise: np.ndarray | torch.Tensor,
    timesteps: np.ndarray | torch.Tensor | None,
    context_noise: np.ndarray | torch.Tensor | None,
) -> object:
    """Run diffusion policy inference to get chunked actions."""
    if cfg.method not in ['tmrl', 'dsrl']:
        return action_noise

    if cfg.method == 'tmrl':
        tcont_context = (timesteps * cfg.timestep_max / 2) + (cfg.timestep_max / 2)
        return model.dp.infer(
            obs=raw_obs, action_noise=action_noise, tcont_context=tcont_context, context_noise=context_noise
        )['actions']
    else:  # dsrl
        return model.dp.infer(obs=raw_obs, action_noise=action_noise)['actions']
