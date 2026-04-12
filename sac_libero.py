from operator import ge
import sys
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
from tmrl.utils.eval import evaluate_policy_libero
from tmrl.utils.env import setup_libero_envs
from tmrl.utils.common import (
    set_seed, merge_batches, to_device, load_pi0_model, to_tensor, create_optimizers,
    check_memory_kill_switch,
)
from tmrl.models.common.vision import Dinov2withNorm


def log(msg, color='bright_green'):
    cprint(msg, color)


def infer_actions(cfg, model, obs, action_noise, timesteps, context_noise):
    """Run diffusion policy inference to get chunked actions."""

    if cfg.debug:
        return np.repeat(action_noise[:, None, :cfg.dataset.action_dim], cfg.action_exec_len, axis=1)

    uses_chunking = cfg.method in ['tmrl', 'dsrl']
    if not uses_chunking:
        return action_noise[:, None, :] if action_noise.ndim == 2 else action_noise

    if cfg.method == 'tmrl':
        if cfg.fixed_tcont_context is not None:
            assert cfg.fixed_tcont_context >= 0.0 and cfg.fixed_tcont_context <= 1.0, (
                f'fixed_tcont_context out of range: [{cfg.fixed_tcont_context}]'
            )
            tcont_context = np.full_like(timesteps, cfg.fixed_tcont_context)
        else:
            assert timesteps.min() >= -1.0 and timesteps.max() <= 1.0, (
                f'timesteps out of range: [{timesteps.min()}, {timesteps.max()}]'
            )
            tcont_context = (timesteps * cfg.timestep_max / 2) + (cfg.timestep_max / 2)

        return model.dp.infer(obs=obs, action_noise=action_noise, tcont_context=tcont_context, context_noise=context_noise)['actions']
    elif cfg.method == 'dsrl':
        return model.dp.infer(obs=obs, action_noise=action_noise)['actions']
    else:
        raise NotImplementedError(f'Unsupported method: {cfg.method}')


def sample_random_actions(cfg, n_envs, action_dim, context_dim):
    """Sample random actions for replay buffer warm-up."""
    action_noise = np.random.uniform(
        low=-cfg.noise_bound, high=cfg.noise_bound, size=(n_envs, action_dim)
    ).astype(np.float32)
    timesteps = context_noise = None
    if cfg.method == 'tmrl':
        timesteps = np.random.uniform(low=-1.0, high=1.0, size=(n_envs, 1)).astype(np.float32)
        context_noise = np.random.uniform(
            low=-cfg.context_noise_bound, high=cfg.context_noise_bound,
            size=(n_envs, context_dim),
        ).astype(np.float32)
    return action_noise, timesteps, context_noise


def get_env_timestep(cfg, timesteps, env_id):
    """Extract scalar timestep for a given env index."""
    if cfg.method != 'tmrl':
        return None
    ts = timesteps[env_id]
    return ts[0] if ts.shape[-1] == 2 else ts.squeeze(-1)


@hydra.main(version_base=None, config_path='tmrl/cfg', config_name='sac_libero.yaml')
def main(cfg):
    if cfg.debug:
        # cfg.n_envs = 1
        cfg.learning_starts = 101
        cfg.dataset.max_ep_steps = 100
        cfg.eval_interval = 5000

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
    if cfg.method in ['dsrl', 'tmrl'] and not cfg.debug:
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
    env = setup_libero_envs(
        cfg, n_envs=cfg.n_envs, dataset_name=cfg.dataset.name, height=256, width=256,
    )
    n_envs = cfg.n_envs
    obs_dim = cfg.model.obs_dim
    state_dim = cfg.model.state_dim
    action_dim = cfg.model.action_dim
    context_dim = cfg.model.context_dim
    action_len = cfg.model.action_len
    uses_chunking = cfg.method in ['tmrl', 'dsrl']
    action_exec_len = cfg.action_exec_len if uses_chunking else 1

    sep = '=' * 70
    log(f'{sep}\nTraining/Environment dimensions:\n{sep}')
    log(f'obs_dim: {obs_dim} | state_dim: {state_dim} | action_dim: {action_dim} | action_len: {action_len} | action_exec_len: {action_exec_len}')
    log(f'noise_bound: {cfg.noise_bound} | timestep_max: {cfg.timestep_max} | context_noise_bound: {cfg.context_noise_bound}')
    log(sep)

    # Offline data (optional)
    train_dict = None
    if cfg.use_offline:
        from tmrl.utils.data import load_data_dict
        old_name = cfg.dataset.name
        cfg.dataset.name = 'libero_pretrain'
        log(f'Using offline dataset: {cfg.dataset.name}')
        train_dict, _ = load_data_dict(cfg=cfg, training=False)
        train_dict = {k: torch.from_numpy(v) for k, v in train_dict.items()}
        cfg.dataset.name = old_name

    # Replay buffers
    rb = TrajectoryReplayBuffer(
        method=cfg.method,
        dataset_name=cfg.dataset.name,
        capacity=cfg.capacity,
        n_envs=n_envs,
        obs_dim=obs_dim + state_dim,
        action_dim=action_dim,
        action_len=action_len,
        context_dim=context_dim,
        discount=cfg.dataset.discount,
        use_success_buffer=cfg.use_success_buffer,
    )

    # Initial env reset + stabilization
    env.reset()
    for _ in range(50):
        env.step(np.random.uniform(-1, 1, (n_envs, 7,)))
    env.reset()
    dummy = np.zeros((n_envs, cfg.dataset.action_dim))
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _, _ = env.step(dummy)

    # Per-env state
    frames = [[] for _ in range(n_envs)]
    clean_frames = [[] for _ in range(n_envs)]
    step_counts = np.zeros(n_envs, dtype=int)
    action_queues = [[] for _ in range(n_envs)]
    curr_ep = [0] * n_envs

    traj_data = [
        {'obs': [], 'action': [], 'timesteps': [], 'context_noises': [], 'reward': [], 'terminal': []}
        for _ in range(n_envs)
    ]

    # Static viz frame
    viz_frame = None
    if render_videos:
        from tmrl.utils.env import make_libero_env
        viz_env = make_libero_env(cfg.dataset.name, task_id=cfg.dataset.task_id, height=5096, width=5096, wrap_env=False)
        for _ in range(cfg.num_steps_wait):
            viz_obs, _, _, _ = viz_env.step(np.zeros(cfg.dataset.action_dim))
        viz_obs = viz_env.reset()
        viz_frame = viz_obs['agentview_image'][::-1, ::-1]
        viz_frame = np.clip(viz_frame.astype(np.float32) * 1.3, 0, 255).astype(np.uint8) # brighten
        viz_env.close()

    cml_rollouts = 0
    cml_success = 0

    # Create eval env once and reuse across evals to avoid EGL context exhaustion
    n_evals = 2 if cfg.debug else cfg.n_evals
    eval_env = setup_libero_envs(cfg, n_envs=n_evals, dataset_name=cfg.dataset.name, height=256, width=256)

    log(f'Running SAC on {cfg.dataset.name} with max_ep_steps {cfg.dataset.max_ep_steps}')
    for step in tqdm(range(cfg.train_steps + 1), dynamic_ncols=True):
        log_data = {}
        step_time = time.time()
        render_videos = render_videos and step < 250_000
        check_memory_kill_switch()

        # Periodic eval
        if step % cfg.eval_interval == 0 and not (cfg.skip_first and step == 0):
            model.eval()
            evaluate_policy_libero(cfg=cfg, model=model, log_data=log_data, step=step, image_encoder=image_encoder, env=eval_env)
            model.train()

        # Encode image obs
        with torch.no_grad():
            image = to_tensor(obs['observation/image']).permute(0, 3, 1, 2)
            obs['observation/image_emb'] = to_tensor(image_encoder(image))

        # Sample actions
        with torch.no_grad():
            if step < cfg.learning_starts:
                action_noise, timesteps, context_noise = sample_random_actions(
                    cfg, n_envs, action_dim, getattr(model, 'context_dim', 0)
                )
            else:
                inputs = torch.cat(
                    [to_tensor(obs['observation/state']), to_tensor(obs['observation/image_emb'])], dim=-1
                )
                (action_noise, timesteps, context_noise), _, _ = model.actor.get_action(inputs)
                action_noise = action_noise.cpu().numpy()
                if cfg.method == 'tmrl':
                    timesteps = timesteps.cpu().numpy()
                    context_noise = context_noise.cpu().numpy()

            actions = infer_actions(cfg, model, obs, action_noise, timesteps, context_noise)

        # Open-loop action execution
        alive = list(range(n_envs))
        full_next_obs = {}
        full_reward = np.zeros(n_envs, dtype=np.float64)
        full_terminated = np.zeros(n_envs, dtype=bool)

        for i in range(action_exec_len):
            if not alive:
                break
            alive_ids = np.array(alive)
            sub_obs, sub_rew, sub_term, _, _ = env.step(actions[alive_ids, i, :], id=alive_ids)

            for k in sub_obs:
                if k not in full_next_obs:
                    full_next_obs[k] = (
                        np.empty(n_envs, dtype=sub_obs[k].dtype)
                        if sub_obs[k].dtype.kind in ('U', 'S', 'O')
                        else np.zeros((n_envs, *sub_obs[k].shape[1:]), dtype=sub_obs[k].dtype)
                    )
                full_next_obs[k][alive_ids] = sub_obs[k]

            full_reward[alive_ids] += sub_rew
            full_terminated[alive_ids] |= sub_term
            step_counts[alive_ids] += 1

            sub_trunc = step_counts[alive_ids] >= cfg.dataset.max_ep_steps
            newly_done = alive_ids[sub_term | sub_trunc]
            alive = [a for a in alive if a not in set(newly_done.tolist())]

        next_obs = full_next_obs
        reward = full_reward
        terminated = full_terminated
        dones = np.where(terminated | (step_counts >= cfg.dataset.max_ep_steps))[0]
        cml_success += terminated.sum()
        cml_rollouts += len(dones)

        # Brighten frame
        frame = next_obs['observation/image']
        frame = np.clip(frame.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)

        # Accumulate trajectory data
        for i in range(n_envs):
            state = obs['observation/state'][i]
            image_emb = obs['observation/image_emb'][i].cpu().float().numpy()
            traj_data[i]['obs'].append(np.concatenate([state, image_emb]))
            traj_data[i]['action'].append(action_noise[i])
            if cfg.method == 'tmrl':
                traj_data[i]['timesteps'].append(timesteps[i])
                traj_data[i]['context_noises'].append(context_noise[i])
            traj_data[i]['reward'].append(float(reward[i]))
            traj_data[i]['terminal'].append(float(terminated[i]))

        # Flush done rollouts to replay buffer
        flush_envs = set(dones.tolist())
        for i in flush_envs:
            flush_traj_to_buffer(traj_data, i, terminated, rb)

        # Reset done envs
        if len(dones) > 0:
            done_ids = np.array(dones)
            env.reset(done_ids)
            # Warmup only the reset envs so non-done envs are not disturbed
            warmup_obs = None
            for _ in range(cfg.num_steps_wait):
                warmup_obs, _, _, _, _ = env.step(
                    np.zeros((len(done_ids), cfg.dataset.action_dim)), id=done_ids
                )
            # Non-done envs keep next_obs; done envs get post-reset obs
            obs = next_obs
            if warmup_obs is not None:
                for k in warmup_obs:
                    obs[k][done_ids] = warmup_obs[k]

            for env_idx in dones:
                curr_ep[env_idx] += 1
                action_queues[env_idx] = []
                step_counts[env_idx] = 0

        else:
            obs = next_obs

        # Actor/critic update
        if step > cfg.learning_starts:
            if step == cfg.learning_starts + 1:
                log(f'Learning starts at step {step}')
            online_batch = to_device(rb.sample(batch_size=cfg.dataset.batch_size*cfg.model.critic_updates_per_actor_update), device)
            batch = merge_batches(offline_batch=None, online_batch=online_batch)
            log_data.update(model.update(**batch, optimizers=optimizers))

        # Frame logging for video
        if render_videos:
            for env_id in range(n_envs):
                frames[env_id].append(frame[env_id])
                clean_frames[env_id].append(frame[env_id])

        # Video upload on episode end
        if step < 100_000:
            for env_id in dones:
                successful = terminated[env_id]
                log_key = f'online/env_success_{env_id}' if successful else f'online/env_{env_id}'
                
                if render_videos and step % cfg.eval_interval == 0:
                    log_data[log_key] = wandb.Video(
                        np.array(frames[env_id]).transpose(0, 3, 1, 2), fps=15, format='mp4'
                    )
                    log_data[log_key + '_clean'] = wandb.Video(
                        np.array(clean_frames[env_id]).transpose(0, 3, 1, 2), fps=15, format='mp4'
                    )

                frames[env_id] = []
                clean_frames[env_id] = []
        
        # Metrics
        if cml_rollouts >= 20:
            log_data['online/cml_success_rate'] = cml_success / cml_rollouts
        log_data.update({
            'online/cml_success': int(cml_success),
            'online/cml_rollouts': cml_rollouts,
            'online/sps': time.time() - step_time,
            'online/action_noise_min': action_noise.min(),
            'online/action_noise_mean': action_noise.mean(),
            'online/action_noise_max': action_noise.max(),
        })
        if cfg.method == 'tmrl':
            log_data.update({
                'online/timesteps_min': timesteps.min(),
                'online/timesteps_mean': timesteps.mean(),
                'online/timesteps_max': timesteps.max(),
                'online/context_noises_min': context_noise.min(),
                'online/context_noises_mean': context_noise.mean(),
                'online/context_noises_max': context_noise.max(),
            })

        if cfg.use_wandb and log_data:
            log_data['global_step'] = step
            wandb.log(log_data)

    eval_env.close()


if __name__ == '__main__':
    main()
