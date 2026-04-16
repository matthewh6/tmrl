import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import wandb
from tmrl.utils.buffer import TrajectoryReplayBuffer, flush_traj_to_buffer
from tmrl.utils.logging import cprint
from tmrl.utils.eval import evaluate_policy_ogbench
from tmrl.utils.env import setup_envs
from tmrl.utils.common import (
    set_seed, load_checkpoint, merge_batches, to_device, create_optimizers, to_tensor
)


def log(msg: str, color: str = 'bright_green') -> None:
    cprint(msg, color)


def sample_random_actions(
    cfg: object, action_dim: int, context_dim: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Sample random actions for replay buffer warm-up."""

    action_noise = np.random.uniform(
        low=-cfg.noise_bound, high=cfg.noise_bound, size=(cfg.n_envs, action_dim)
    ).astype(np.float32)
    timesteps = context_noise = None
    if cfg.method == 'tmrl':
        timesteps = torch.empty((cfg.n_envs, 1), dtype=torch.float32).uniform_(-1.0, 1.0)
        context_noise = torch.empty((cfg.n_envs, context_dim), dtype=torch.float32).uniform_(-cfg.context_noise_bound, cfg.context_noise_bound)

    action_noise = to_tensor(action_noise)

    return action_noise, timesteps, context_noise


def infer_actions(
    cfg: object,
    model: object,
    obs: np.ndarray | torch.Tensor,
    info: dict[str, Any],
    action_noise: np.ndarray | torch.Tensor,
    timesteps: np.ndarray | torch.Tensor | None,
    context_noise: np.ndarray | torch.Tensor | None,
) -> torch.Tensor:
    obs = to_tensor(obs)
    goal = to_tensor(info['goal']) if 'goal' in info else None
    if goal is not None:
        # obs is [raw_obs, goal] concatenated; extract raw obs for flow model
        obs = obs[:, :cfg.dataset.obs_dim]

    method = cfg.method

    if method in ['tmrl', 'dsrl']:
        action_noise = to_tensor(action_noise)
        if method == 'tmrl':
            context_noise = to_tensor(context_noise)
            timesteps = to_tensor(timesteps)
            tcont_context = (timesteps * cfg.timestep_max / 2) + (cfg.timestep_max / 2)
        else:
            tcont_context = None

        return model.dp.sample(
            obs=obs,
            goals=goal,
            tcont_context=tcont_context,
            action_noise=action_noise,
            context_noise=context_noise
        )

    else:
        actions = to_tensor(action_noise)
        return actions.unsqueeze(1) if actions.ndim == 2 else actions


@hydra.main(
    version_base=None,
    config_path='tmrl/cfg',
    config_name='sac_ogbench.yaml',
)
def main(cfg: object) -> None:
    if cfg.debug:
        cfg.n_envs = 1
        cfg.n_evals = 2
        cfg.skip_first = True

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

    # For cube envs, actor/critic see [obs, goal] concatenated
    is_cube = 'cube' in cfg.dataset.name
    if is_cube:
        sac_obs_dim = cfg.dataset.obs_dim + cfg.dataset.goal_dim
        cfg.model.obs_dim = sac_obs_dim

    # Model
    model = instantiate(cfg.model)
    model.timestep_action_dim = 1 if cfg.method in ['tmrl'] else 0

    pretrained_model = None
    if cfg.method in ['dsrl', 'tmrl']:
        assert cfg.ckpt is not None
        pretrained_model = load_checkpoint(cfg.ckpt, device=device)
        model.dp = pretrained_model

    model.initialize()
    model = model.train().to(device)
    optimizers = create_optimizers(model, cfg)

    log('-' * 30 + ' Model Architecture ' + '-' * 30)
    log(model)
    log('-' * 80)

    # Environment
    env_name = dataset_name = cfg.dataset.name
    n_envs = cfg.n_envs
    env, eval_env = setup_envs(cfg, n_envs=n_envs, dataset_name=dataset_name)

    # Dims
    obs_dim = cfg.dataset.obs_dim
    goal_dim = cfg.dataset.goal_dim
    env_obs_dim = obs_dim + goal_dim if is_cube else obs_dim
    action_dim = cfg.dataset.action_dim
    context_dim = cfg.model.context_dim
    action_len = cfg.dataset.action_len
    uses_chunking = cfg.method in ['tmrl', 'dsrl', 'spirl']
    action_exec_len = cfg.action_exec_len if uses_chunking else 1
    state_dim = cfg.dataset.state_dim
    actor_action_dim = action_dim + 1 if cfg.method in ['tmrl'] else action_dim

    sep = '=' * 70
    log(f'{sep}\nTraining/Environment dimensions:\n{sep}')
    log(f'obs_dim: {obs_dim} | state_dim: {state_dim} | action_dim: {action_dim} | action_len: {action_len} | actor_action_dim: {actor_action_dim} | goal_dim: {goal_dim}')
    log(sep)

    # Offline dataset
    train_dataset = None
    if cfg.use_offline or cfg.method == 'rlpd':
        from tmrl.utils.data import load_data_dict, MazeDataset, CubeDataset

        old_dataset_name = cfg.dataset.name

        # For pointmaze experiment, we evaluate on the larger maze.
        if cfg.dataset.name == 'pointmaze-giant-navigate-v0':
            cfg.dataset.name = 'pointmaze-large-navigate-v0'

        log(f'Using offline dataset: {cfg.dataset.name}')
        train_dict, _ = load_data_dict(cfg=cfg)
        train_dict = {k: torch.from_numpy(v) for k, v in train_dict.items()}

        if 'maze' in cfg.dataset.name:
            train_dataset = MazeDataset(train_dict, cfg.dataset.action_len, cfg.dataset.discount)
        elif 'cube' in cfg.dataset.name:
            train_dataset = CubeDataset(train_dict, cfg.dataset.action_len, cfg.dataset.discount)

        log('Sample batch:')
        log('-' * 60)
        sample_batch = train_dataset.sample(batch_size=cfg.dataset.batch_size)
        for k, v in sample_batch.items():
            log(f'{k:12} | {tuple(v.shape)} | {str(v.dtype):>8} | {v.min():8.4f} | {v.max():8.4f}')
        log('-' * 60)

        cfg.dataset.name = old_dataset_name

    # Set up replay buffer
    rb = TrajectoryReplayBuffer(
        method=cfg.method,
        dataset_name=cfg.dataset.name,
        capacity=cfg.capacity,
        n_envs=n_envs,
        obs_dim=env_obs_dim,
        action_dim=action_dim,
        action_len=1,
        context_dim=context_dim,
        goal_dim=goal_dim,
        discount=cfg.dataset.discount,
        use_success_buffer=cfg.use_success_buffer,
    )

    # Initial env reset
    obs, info = env.reset()

    # Per-env state
    frames = [[] for _ in range(n_envs)]
    traj_data = [
        {'obs': [], 'action': [], 'timesteps': [], 'context_noises': [], 'reward': [], 'terminal': []}
        for _ in range(n_envs)
    ]

    cml_success = 0
    cml_rollouts = 0
    env_step = 0

    log(f'Running SAC on {env_name}')
    pbar = tqdm(total=cfg.train_steps, dynamic_ncols=True)
    while env_step <= cfg.train_steps:
        log_data = {}
        step_time = time.time()

        # Periodic eval
        if env_step % cfg.eval_interval == 0 and not (cfg.skip_first and env_step == 0):
            model.eval()
            evaluate_policy_ogbench(cfg=cfg, env=eval_env, model=model, log_data=log_data, step=env_step)
            model.train()

        # Sample actions
        with torch.no_grad():
            if env_step < cfg.learning_starts:
                action_noise, timesteps, context_noise = sample_random_actions(cfg, action_dim, context_dim)
            else:
                inputs = to_tensor(obs)
                (action_noise, timesteps, context_noise), _, _ = model.actor.get_action(inputs)
            actions = infer_actions(cfg, model, obs, info, action_noise, timesteps, context_noise)

        actions = actions.detach().cpu().numpy()
        action_noise = action_noise.detach().cpu().numpy()

        if cfg.method == 'tmrl':
            timesteps = timesteps.detach().cpu().numpy()
            context_noise = context_noise.detach().cpu().numpy()

        # Open-loop action execution
        done_mask = np.zeros(n_envs, dtype=bool)
        full_next_obs = np.zeros((n_envs, env_obs_dim), dtype=np.float32)
        full_reward = np.zeros(n_envs, dtype=np.float64)
        full_terminated = np.zeros(n_envs, dtype=bool)
        full_truncated = np.zeros(n_envs, dtype=bool)
        reset_obs = {}  # store reset obs for done envs

        for i in range(action_exec_len):
            if done_mask.all():
                break

            # Zero out actions for already-done envs
            step_actions = actions[:, i, :].copy()
            if cfg.dataset.name == 'pointmaze-giant-navigate-v0':
                step_actions = np.clip(step_actions, -1, 1)
            step_actions[done_mask] = 0.0

            sub_obs, sub_rew, sub_term, sub_trunc, info = env.step(step_actions)

            alive_mask = ~done_mask
            full_next_obs[alive_mask] = sub_obs[alive_mask]
            full_reward[alive_mask] += sub_rew[alive_mask]
            full_terminated[alive_mask] |= sub_term[alive_mask]
            full_truncated[alive_mask] |= sub_trunc[alive_mask]

            newly_done = np.where((sub_term | sub_trunc) & alive_mask)[0]
            done_mask |= (sub_term | sub_trunc)

            # Reset terminated envs so SyncVectorEnv allows stepping them next iteration
            for idx in newly_done:
                r_obs, r_info = env.envs[idx].reset()
                env._autoreset_envs[idx] = False
                reset_obs[idx] = (r_obs, r_info)

        next_obs = full_next_obs
        reward = full_reward
        terminated = full_terminated
        truncated = full_truncated
        dones = np.where(terminated | truncated)[0]
        cml_success += terminated.sum()
        cml_rollouts += len(dones)

        env_step += 1
        pbar.update(1)

        # Accumulate trajectory data
        for i in range(n_envs):
            traj_data[i]['obs'].append(obs[i])
            traj_data[i]['action'].append(np.copy(action_noise[i]))
            if cfg.method == 'tmrl':
                traj_data[i]['timesteps'].append(np.copy(timesteps[i]))
                traj_data[i]['context_noises'].append(np.copy(context_noise[i]))
            traj_data[i]['terminal'].append(float(terminated[i]))
            traj_data[i]['reward'].append(float(reward[i]))

        # Flush done trajectories to replay buffer
        for i in dones:
            flush_traj_to_buffer(traj_data, i, terminated, rb)

        # Also flush before eval to avoid losing data
        about_to_eval = (env_step + action_exec_len) % cfg.eval_interval < action_exec_len
        if about_to_eval:
            for i in range(n_envs):
                if traj_data[i]['obs']:
                    flush_traj_to_buffer(traj_data, i, terminated, rb)

        # Update obs — done envs were already reset in the inner loop
        if len(dones) > 0:
            for idx in dones:
                r_obs, r_info = reset_obs[idx]
                next_obs[idx] = r_obs
                for k in info:
                    if k in r_info:
                        info[k][idx] = r_info[k]
        obs = next_obs

        # Actor/critic updates
        if env_step > cfg.learning_starts:
            if env_step == cfg.learning_starts + 1:
                log(f'Learning starts at env_step {env_step}')

            total_batch_size = cfg.dataset.batch_size * cfg.model.critic_updates_per_actor_update

            if cfg.use_offline or cfg.method == 'rlpd':
                batch_size_offline = total_batch_size // 2
                batch_size_online = total_batch_size - batch_size_offline
                offline_batch = to_device(train_dataset.sample(batch_size=batch_size_offline), device)
                # For cube: concatenate [obs, goal] to match online buffer format
                if is_cube and 'goals' in offline_batch:
                    offline_batch['obs'] = torch.cat([offline_batch['obs'], offline_batch['goals']], dim=-1)
                    offline_batch['next_obs'] = torch.cat([offline_batch['next_obs'], offline_batch['goals']], dim=-1)
                    del offline_batch['goals']
            else:
                batch_size_online = total_batch_size
                offline_batch = None

            online_batch = to_device(rb.sample(batch_size=batch_size_online), device)

            batch = merge_batches(offline_batch=offline_batch, online_batch=online_batch)
            log_data.update(model.update(**batch, optimizers=optimizers))

        # Frame capture
        if render_videos:
            frame = env.render()
            for env_id in range(n_envs):
                frames[env_id].append(frame[env_id])

        # Video logging on episode end
        if render_videos:
            for env_id in dones:
                successful = bool(terminated[env_id])
                if frames[env_id]:
                    log_key = 'online/ep_success' if successful else 'online/ep'
                    log_data[log_key] = wandb.Video(
                        np.array(frames[env_id]).transpose(0, 3, 1, 2), fps=15, format='mp4'
                    )
                frames[env_id] = []

        # Metrics
        if cml_rollouts >= 10:
            log_data['online/cml_success_rate'] = cml_success / cml_rollouts
        log_data.update({
            'online/cml_success': int(cml_success),
            'online/cml_rollouts': cml_rollouts,
            'online/sps': time.time() - step_time,
        })
        if action_noise is not None:
            log_data.update({
                'online/action_noise_min': action_noise.min().item(),
                'online/action_noise_mean': action_noise.mean().item(),
                'online/action_noise_max': action_noise.max().item(),
            })
        if cfg.method == 'tmrl' and timesteps is not None:
            log_data['online/timesteps'] = timesteps.mean().item()

        if cfg.use_wandb and log_data:
            log_data['global_step'] = env_step
            wandb.log(log_data)


if __name__ == '__main__':
    main()
