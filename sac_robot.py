from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import wandb
from tmrl.utils.buffer import TrajectoryReplayBuffer, flush_traj_to_buffer
from tmrl.utils.logging import cprint
from tmrl.utils.eval import evaluate_policy_robot
from tmrl.utils.remote_env import make_remote_env
from tmrl.utils.common import (
    set_seed,
    to_device,
    load_pi0_model,
    to_tensor,
    create_optimizers,
    infer_actions,
)
from tmrl.utils.obs import embed_images
from tmrl.models.common.vision import Dinov2withNorm


def log(msg: str, color: str = 'bright_green') -> None:
    cprint(msg, color)


def sample_random_actions(
    cfg: object, n_envs: int, action_dim: int, context_dim: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Sample random actions for replay buffer warm-up."""
    action_noise = np.random.uniform(low=-cfg.noise_bound, high=cfg.noise_bound, size=(n_envs, action_dim)).astype(
        np.float32
    )
    timesteps = context_noise = None
    if cfg.method == 'tmrl':
        timesteps = np.random.uniform(low=-1.0, high=1.0, size=(n_envs, 1)).astype(np.float32)
        context_noise = np.random.uniform(
            low=-cfg.context_noise_bound,
            high=cfg.context_noise_bound,
            size=(n_envs, context_dim),
        ).astype(np.float32)
    return action_noise, timesteps, context_noise


def _save_training_checkpoint(
    cfg: object,
    step: int,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    counters: dict[str, object],
) -> None:
    """Save model, optimizers, buffers, and counters to disk."""
    run_stamp = getattr(_save_training_checkpoint, '_run_stamp', None)
    if run_stamp is None:
        run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _save_training_checkpoint._run_stamp = run_stamp

    save_dir = Path(cfg.paths.results_dir).expanduser() / cfg.dataset.name / run_stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'step': step,
        'pi0_ckpt_dir': cfg.ckpt,
        'cfg': OmegaConf.to_container(cfg, resolve=True),
        'model': model.state_dict(),
        'optimizers': {k: opt.state_dict() for k, opt in optimizers.items()},
        'counters': counters,
    }

    step_path = save_dir / f'step_{step}.pt'
    latest_path = save_dir / 'latest.pt'
    torch.save(ckpt, step_path)
    torch.save(ckpt, latest_path)
    log(f'Saved checkpoint at step {step} to {step_path} and {latest_path}')


def _maybe_load_checkpoint(
    cfg: object, model: torch.nn.Module, optimizers: dict[str, torch.optim.Optimizer], device: torch.device
) -> tuple[int, dict[str, object]]:
    """Load checkpoint if resume_path is provided."""
    ckpt_cfg = getattr(cfg, 'checkpoint', None)
    resume_path = getattr(ckpt_cfg, 'resume_path', None)
    if resume_path is None:
        return 0, {}

    ckpt_path = Path(resume_path).expanduser()
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / 'latest.pt'
    if not ckpt_path.exists():
        log(f'Checkpoint path {ckpt_path} does not exist, starting fresh', color='bright_yellow')
        return 0, {}

    log(f'Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)

    pi0_ckpt_dir = ckpt.get('pi0_ckpt_dir', getattr(cfg, 'ckpt', None))
    if pi0_ckpt_dir:
        try:
            model.dp = load_pi0_model(cfg)
            log(f'Loaded Pi0 model from {pi0_ckpt_dir}')
        except Exception as exc:
            log(f'Failed to load Pi0 model from {pi0_ckpt_dir}: {exc}', color='bright_red')
            raise

    model.load_state_dict(ckpt['model'], strict=False)
    for name, opt in optimizers.items():
        if 'optimizers' in ckpt and name in ckpt['optimizers']:
            opt.load_state_dict(ckpt['optimizers'][name])
            opt.zero_grad()

    counters = ckpt.get('counters', {})
    step = ckpt.get('step', 0)
    log(f'Resumed from checkpoint {ckpt_path} at step {step}')
    return step, counters


@hydra.main(version_base=None, config_path='tmrl/cfg', config_name='sac_robot.yaml')
def main(cfg: object) -> None:
    if cfg.debug:
        cfg.learning_starts = 101
        cfg.action_exec_len = 1
        cfg.n_evals = 2

    # Wandb
    if cfg.use_wandb:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            **OmegaConf.to_container(cfg.wandb, resolve=True),
        )
        wandb.define_metric('global_step')
        wandb.define_metric('*', step_metric='global_step')

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    render_videos = (cfg.render_videos and cfg.use_wandb) or cfg.debug

    # Model
    model = instantiate(cfg.model)
    ckpt_cfg = getattr(cfg, 'checkpoint', None)
    resume_path = getattr(ckpt_cfg, 'resume_path', None)
    if cfg.method in ['dsrl', 'tmrl'] and resume_path is None:
        assert cfg.ckpt
        model.dp = load_pi0_model(cfg)
    model.initialize()
    model = model.train().to(device)
    optimizers = create_optimizers(model, cfg)

    # Image encoder
    image_encoder = Dinov2withNorm().to(device).eval()

    log('-' * 30 + ' Model Architecture ' + '-' * 30)
    log(model)
    log('-' * 80)

    # Environment
    remote_cfg = OmegaConf.select(cfg, 'remote_robot', default=None) or {}
    remote_host = remote_cfg.get('host') or 'localhost'
    remote_port = remote_cfg.get('port') or 6000
    log(f'Connecting to remote robot at {remote_host}:{remote_port}')
    env = make_remote_env(
        server_url=f'tcp://{remote_host}:{remote_port}',
        obs_format=cfg.dataset.name,
        connect_timeout=remote_cfg.get('connect_timeout', 300.0),
    )

    obs, _ = env.reset()
    instruction = obs['prompt']
    log(f'Instruction: {instruction}')
    log(f'Connected to remote robot.')

    n_envs = cfg.n_envs
    obs_dim = cfg.model.obs_dim
    state_dim = cfg.model.state_dim
    action_dim = cfg.model.action_dim
    action_len = cfg.model.action_len
    action_exec_len = cfg.action_exec_len
    timestep_max = cfg.timestep_max

    sep = '=' * 70
    log(f'{sep}\nTraining/Environment dimensions:\n{sep}')
    log(
        f'obs_dim: {obs_dim} | state_dim: {state_dim} | action_dim: {action_dim} | action_len: {action_len} | action_exec_len: {action_exec_len}'
    )
    log(
        f'noise_bound: {cfg.noise_bound} | timestep_max: {timestep_max} | context_noise_bound: {cfg.context_noise_bound}'
    )
    log(sep)

    # Replay buffer
    rb_kwargs = dict(
        method=cfg.method,
        dataset_name=cfg.dataset.name,
        capacity=cfg.capacity,
        n_envs=n_envs,
        obs_dim=obs_dim + state_dim,
        action_dim=action_dim,
        action_len=action_len,
        discount=cfg.dataset.discount,
        use_success_buffer=cfg.use_success_buffer,
        context_dim=getattr(model, 'context_dim', 0),
    )
    rb = TrajectoryReplayBuffer(**rb_kwargs)

    # Per-env state
    frames = []
    wrist_frames = []
    step_count = 0

    traj_data = [
        {'obs': [], 'action': [], 'timesteps': [], 'context_noises': [], 'reward': [], 'terminal': []}
        for _ in range(n_envs)
    ]

    cml_rollouts = 0
    cml_success = 0

    # Optionally resume from checkpoint
    start_step, ckpt_counters = _maybe_load_checkpoint(cfg, model, optimizers, device)
    if ckpt_counters:
        cml_rollouts = ckpt_counters.get('cml_rollouts', cml_rollouts)
        cml_success = ckpt_counters.get('cml_success', cml_success)

    log(f'Running {cfg.method} on dataset: {cfg.dataset.name} with task: {instruction}')

    for step in tqdm(
        range(start_step, cfg.train_steps + 1), total=cfg.train_steps, initial=start_step, dynamic_ncols=True
    ):
        log_data = {}

        # Checkpointing
        if ckpt_cfg is not None and step > 0 and (step - start_step) % ckpt_cfg.save_interval == 0:
            counters = {'cml_rollouts': cml_rollouts, 'cml_success': cml_success}
            _save_training_checkpoint(cfg, step, model, optimizers, counters)

        # Periodic eval
        if (step - start_step) % cfg.eval_interval == 0 and not (cfg.skip_first and step == start_step):
            model.eval()
            evaluate_policy_robot(
                cfg=cfg,
                eval_env=env,
                model=model,
                log_data=log_data,
                step=step,
                image_encoder=image_encoder,
            )
            model.train()

        # Encode image obs
        raw_obs = obs
        obs = embed_images(dict(obs), image_encoder)

        # Sample actions
        with torch.no_grad():
            if step - start_step < cfg.learning_starts:
                action_noise, timesteps, context_noise = sample_random_actions(
                    cfg, n_envs, action_dim, getattr(model, 'context_dim', 0)
                )
            else:
                inputs = torch.cat([to_tensor(obs['state'][None, :]), to_tensor(obs['observation/image_emb'])], dim=-1)
                (action_noise, timesteps, context_noise), _, _ = model.actor.get_action(inputs)
                action_noise = action_noise.cpu().numpy()
                if cfg.method == 'tmrl':
                    timesteps = timesteps.cpu().numpy()
                    context_noise = context_noise.cpu().numpy()

            actions = infer_actions(cfg, model, raw_obs, action_noise, timesteps, context_noise)
            actions = actions[:action_exec_len]
            actions = np.asarray(actions, dtype=np.float32)

        # Capture frame from pre-step obs
        frame = raw_obs.get('observation/exterior_image_1_left', raw_obs.get('observation/image'))
        wrist_frame = raw_obs.get('observation/wrist_image_left')

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(actions)
        if info.get('is_success', False):
            terminated = True

        step_count += action_exec_len
        done = bool(terminated or truncated)
        is_success = bool(terminated)
        cml_success += int(is_success)

        # Accumulate trajectory data
        state = obs['state']
        image_emb = obs['observation/image_emb'].squeeze(0).cpu().float().numpy()
        traj_data[0]['obs'].append(np.concatenate([state, image_emb], axis=-1))
        traj_data[0]['action'].append(np.copy(action_noise.squeeze(0)))
        if cfg.method == 'tmrl':
            traj_data[0]['timesteps'].append(np.copy(timesteps.squeeze(0)))
            traj_data[0]['context_noises'].append(np.copy(context_noise.squeeze(0)))
        traj_data[0]['reward'].append(float(reward))
        traj_data[0]['terminal'].append(float(terminated))

        # Flush trajectory to replay buffer
        about_to_eval = (step + 1 - start_step) % cfg.eval_interval == 0
        if done or about_to_eval:
            flush_traj_to_buffer(traj_data, 0, terminated, rb)

        # Frame logging for video
        if render_videos:
            frames.append(frame)
            if wrist_frame is not None:
                wrist_frames.append(wrist_frame)

        # Video upload and reset on episode end
        if done:
            cml_rollouts += 1

            if render_videos and frames:
                log_key = 'online/ep_success' if is_success else 'online/ep'
                log_data[log_key] = wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=15, format='mp4')
                if wrist_frames:
                    log_data[log_key + '_wrist'] = wandb.Video(
                        np.array(wrist_frames).transpose(0, 3, 1, 2), fps=15, format='mp4'
                    )

            frames.clear()
            wrist_frames.clear()

            obs, _ = env.reset()
        else:
            obs = next_obs

        # Actor/critic update
        if step - start_step >= cfg.learning_starts:
            if step - start_step == cfg.learning_starts:
                log(f'Learning starts at step {step}')
            batch = to_device(
                rb.sample(batch_size=cfg.dataset.batch_size * cfg.model.critic_updates_per_actor_update), device
            )
            log_data.update(model.update(**batch, optimizers=optimizers))

        # Metrics
        log_data.update(
            {
                'online/cml_success': int(cml_success),
                'online/cml_rollouts': cml_rollouts,
                'online/cml_success_rate': (cml_success / cml_rollouts) if cml_rollouts > 0 else 0.0,
                'online/action_noise_min': action_noise.min(),
                'online/action_noise_mean': action_noise.mean(),
                'online/action_noise_max': action_noise.max(),
            }
        )
        if cfg.method == 'tmrl':
            log_data.update(
                {
                    'online/timesteps_min': timesteps.min(),
                    'online/timesteps_mean': timesteps.mean(),
                    'online/timesteps_max': timesteps.max(),
                    'online/context_noises_min': context_noise.min(),
                    'online/context_noises_mean': context_noise.mean(),
                    'online/context_noises_max': context_noise.max(),
                }
            )

        if cfg.use_wandb and log_data:
            log_data['learning_step'] = step
            log_data['env_step'] = step_count
            wandb.log(log_data)

    env.close()


if __name__ == '__main__':
    main()
