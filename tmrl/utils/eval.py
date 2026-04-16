import time

import cv2
import numpy as np
import torch
import wandb
from tqdm import tqdm

from tmrl.utils.common import to_tensor, infer_actions
from tmrl.utils.env import create_env
from tmrl.utils.logging import cprint
from tmrl.utils.obs import embed_images

log = lambda msg, color='bright_cyan': cprint(msg, color)

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def evaluate_policy_ogbench(
    cfg: object,
    env: object,
    model: object,
    log_data: dict[str, object],
    step: int,
) -> None:
    """
    Evaluate policy performance across multiple environments.

    Args:
        cfg: Configuration object
        env: Environment to evaluate on
        model: Model to evaluate
        log_data: Dictionary to store logging data
        step: Current training step
    """
    from sac_ogbench import infer_actions

    dataset_name = cfg.dataset.name

    is_cube = 'cube' in dataset_name

    n_evals = cfg.n_evals
    n_viz = min(getattr(cfg, 'n_viz', 1), n_evals)

    action_exec_len = cfg.action_exec_len

    if wandb.run is not None or cfg.debug:
        viz_envs = [
            create_env(
                dataset_name=dataset_name,
                seed=cfg.seed + i,
                height=1024,
                width=1024,
            )
            for i in range(n_viz)
        ]

    action_maxs = []
    action_mins = []
    action_means = []
    action_stds = []

    obs, info = env.reset()

    if wandb.run is not None or cfg.debug:
        # synchronize visualization envs with the first n_viz eval envs
        for viz_idx in range(n_viz):
            if is_cube:
                qpos = env.envs[viz_idx].unwrapped._data.qpos.copy()
                qvel = env.envs[viz_idx].unwrapped._data.qvel.copy()
                task_id = env.envs[viz_idx].unwrapped.cur_task_id

                viz_envs[viz_idx].reset(options=dict(render_goal=True, task_id=task_id))
                viz_envs[viz_idx].unwrapped.set_state(qpos=qpos, qvel=qvel)
            else:
                xy = env.envs[viz_idx].unwrapped.get_xy()
                goal_xy = env.envs[viz_idx].unwrapped.cur_goal_xy
                viz_envs[viz_idx].unwrapped.set_goal(goal_xy=goal_xy)
                viz_envs[viz_idx].unwrapped.set_xy(xy)

    done = np.array([False] * n_evals)
    frames = [[] for _ in range(n_viz)] if (wandb.run is not None or cfg.debug) else None

    ep_successes = np.zeros(n_evals)
    ep_lengths = np.zeros(n_evals)
    env_step = 0

    while not np.all(done):

        with torch.no_grad():
            inputs = to_tensor(obs)
            (actor_actions, timesteps, context_noise), _, _ = model.actor.get_action(inputs)

            if cfg.method in ['dsrl', 'tmrl']:
                actions = infer_actions(cfg, model, obs, info, actor_actions, timesteps, context_noise)

            elif cfg.method == 'spirl':
                if actor_actions.ndim == 3:
                    actions = actor_actions
                else:
                    skill_latent = actor_actions.squeeze(1)
                    actions = model.spirl.decoder.get_action(skill_latent)

            else:
                if actor_actions.ndim == 2:
                    actor_actions = actor_actions.unsqueeze(1)
                actions = actor_actions

        assert actions.ndim == 3  # (n_envs, action_len, action_dim)
        actions = actions.cpu().numpy()

        actor_actions = actor_actions.cpu().numpy()
        action_maxs.append(actor_actions.max())
        action_mins.append(actor_actions.min())
        action_means.append(actor_actions.mean())
        action_stds.append(actor_actions.std())

        # Execute action chunk
        for i in range(action_exec_len):
            active_mask = ~done
            if not np.any(active_mask):
                break  # all envs are done

            action = actions[:, i, :]
            action = action * active_mask[:, None]  # mask actions for done environments

            obs, reward, terminated, truncated, info = env.step(action)

            done |= np.logical_or(terminated, truncated)

            for env_idx in range(n_evals):
                if not active_mask[env_idx]:
                    continue
                if info['success'][env_idx]:
                    ep_successes[env_idx] = 1
                    env.reset(options={'reset_mask': done})
                ep_lengths[env_idx] += 1

            if frames is not None and env_step % 5 == 0:
                for viz_idx in range(n_viz):
                    if done[viz_idx]:
                        continue

                    # sync viz env with current state of eval env
                    if is_cube:
                        qpos = env.envs[viz_idx].unwrapped._data.qpos.copy()
                        qvel = env.envs[viz_idx].unwrapped._data.qvel.copy()
                        task_id = env.envs[viz_idx].unwrapped.cur_task_id
                        viz_envs[viz_idx].reset(options=dict(render_goal=True, task_id=task_id))
                        viz_envs[viz_idx].unwrapped.set_state(qpos=qpos, qvel=qvel)
                    else:
                        xy = env.envs[viz_idx].unwrapped.get_xy()
                        goal_xy = env.envs[viz_idx].unwrapped.cur_goal_xy
                        viz_envs[viz_idx].unwrapped.set_goal(goal_xy=goal_xy)
                        viz_envs[viz_idx].unwrapped.set_xy(xy)

                    frame = viz_envs[viz_idx].render()
                    frames[viz_idx].append(frame)

            env_step += 1

    if frames is not None and wandb.run is not None:
        log_frames = frames
        if all(len(f) == 0 for f in log_frames):
            pass
        else:
            max_len = max(len(f) for f in log_frames)
            for i in range(n_viz):  # pad env frames to the same length
                if len(log_frames[i]) == 0:
                    continue
                if len(log_frames[i]) < max_len:
                    pad_frame = log_frames[i][-1]
                    log_frames[i].extend([pad_frame] * (max_len - len(log_frames[i])))

            # Drop any completely empty lists (if some envs never produced frames)
            log_frames = [f for f in log_frames if len(f) > 0]
            if len(log_frames) > 0:
                log_frames = np.concatenate(log_frames, axis=2)  # (F, H, viz_envs*W, C)
                log_frames = log_frames.transpose(0, 3, 1, 2)  # (F, C, H, viz_envs*W)

                log_data.update(
                    {
                        f'eval/eval': wandb.Video(
                            log_frames,
                            fps=10,
                            format='mp4',
                        )
                    }
                )

    log(
        f'eval at step {step} — success: {np.mean(ep_successes):.2f} ({np.sum(ep_successes)}/{len(ep_successes)}), len: {np.mean(ep_lengths):.1f}±{np.std(ep_lengths):.1f}'
    )

    # Log action statistics
    log_data.update(
        {
            'eval/actor_actions_min': np.mean(action_mins),
            'eval/actor_actions_max': np.mean(action_maxs),
            'eval/actor_actions_mean': np.mean(action_means),
            'eval/actor_actions_std': np.mean(action_stds),
            f'eval/success': np.mean(ep_successes),
            f'eval/ep_len': np.mean(ep_lengths),
        }
    )


def evaluate_policy_libero(
    cfg: object,
    model: object,
    log_data: dict[str, object],
    step: int,
    image_encoder: object | None = None,
    env: object | None = None,
) -> None:
    """
    Evaluate policy performance across multiple LIBERO environments.

    Args:
        cfg: Configuration object
        model: Model to evaluate
        log_data: Dictionary to store logging data
        step: Current training step
        image_encoder: Observation encoder
        env: Pre-created eval environment (reused across calls to avoid resource exhaustion).
             If None, a new environment is created and closed after eval.
    """
    from tmrl.utils.env import setup_libero_envs
    from sac_libero import infer_actions

    # Setup environments
    n_viz = cfg.n_viz
    n_evals = 2 if cfg.debug else cfg.n_evals
    owns_env = env is None
    if owns_env:
        env = setup_libero_envs(cfg, n_envs=n_evals, dataset_name=cfg.dataset.name, height=256, width=256)
    else:
        env.reset()

    dummy = np.zeros((n_evals, cfg.dataset.action_dim))
    for _ in range(cfg.num_steps_wait):
        obs, _, _, _, info = env.step(dummy)

    # Evaluation config
    max_ep_steps = cfg.dataset.max_ep_steps
    uses_chunking = cfg.method in ["tmrl", "dsrl", "spirl"]
    uses_timesteps = cfg.method == "tmrl"
    action_exec_len = cfg.action_exec_len if uses_chunking else 1

    # Logging state
    n_viz = min(n_viz, n_evals)
    action_dists = [[] for _ in range(n_evals)]
    timesteps_list = [[] for _ in range(n_evals)] if uses_timesteps else None
    dones = np.zeros(n_evals, dtype=bool)
    rewards = np.zeros(n_evals)
    ep_lengths = np.zeros(n_evals)
    frames = [[] for _ in range(n_viz)]

    alive = list(range(n_evals))
    steps = 0
    start = time.time()

    while alive and steps < max_ep_steps:
        # Inference
        with torch.no_grad():
            image = to_tensor(obs["observation/image"]).permute(0, 3, 1, 2)
            obs["observation/image_emb"] = to_tensor(image_encoder(image))

            inputs = torch.cat(
                [to_tensor(obs["observation/state"]), obs["observation/image_emb"]], dim=-1
            )
            _, _, (action_noise, timesteps, context_noise) = model.actor.get_action(inputs)
            action_noise = action_noise.cpu().numpy()

            if cfg.method == 'tmrl':
                timesteps = timesteps.cpu().numpy()
                context_noise = context_noise.cpu().numpy()

            actions = infer_actions(cfg, model, obs, action_noise, timesteps, context_noise)

        # action_noise / timesteps are shaped (len(alive), ...) — use local idx to index them
        if uses_timesteps:
            ts = timesteps.squeeze(-1) if timesteps.shape[-1] == 1 else timesteps

        # Record per-env stats
        for idx, e in enumerate(alive):
            action_dists[e].append(np.linalg.norm(action_noise[idx]))
            if uses_timesteps:
                timesteps_list[e].append(ts[idx])

        # Open-loop action execution
        for i in range(action_exec_len):
            obs_step, reward_step, done_step, _, info_step = env.step(actions[:, i, :])

            # Only update stats for alive envs
            for e in alive:
                if not dones[e]:
                    ep_lengths[e] += 1
                    rewards[e] += reward_step[e]

                    if e < n_viz and steps % 3 == 0:
                        current_frame = obs_step["observation/image"][e]
                        current_frame = np.clip(current_frame.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
                        frames[e].append(current_frame)

            newly_done = [e for e in alive if done_step[e] and not dones[e]]
            if newly_done:
                for e in newly_done:
                    log(f"[SUCCESSFUL ENV]: {info_step[alive.index(e)]['language_instruction']}")
                    dones[e] = True
                env.reset(newly_done) # reset newly done envs, but treat them as dead envs

            steps += 1
            if steps >= max_ep_steps:
                break

        alive = [e for e in alive if not dones[e]]
        obs = obs_step

    # Logging
    if wandb.run is not None:
        valid = [f for f in frames if f]
        if valid:
            max_len = max(len(f) for f in valid)
            padded = []
            for env_frames in valid:
                env_frames = list(env_frames)
                env_frames += [env_frames[-1]] * (max_len - len(env_frames))
                padded.append(env_frames)

            # (n_viz, F, H, W, C) -> (F, C, H, n_viz*W)
            arr = np.stack(padded, axis=0).transpose(1, 0, 2, 3, 4)
            F, n, H, W, C = arr.shape
            arr = arr.transpose(0, 2, 1, 3, 4).reshape(F, H, n * W, C).transpose(0, 3, 1, 2)
            log_data["eval/viz"] = wandb.Video(
                arr, fps=10, format="mp4", caption=info[0]["language_instruction"]
            )

    log(
        f"eval at step {step} — success: {np.mean(dones):.2f} ({np.sum(dones)}/{len(dones)}), "
        f"len: {np.mean(ep_lengths):.1f}±{np.std(ep_lengths):.1f}"
    )

    mean_action_dist = np.mean([np.mean(d) for d in action_dists])
    log_data["eval/action_noise_dist"] = mean_action_dist

    if dones.any():
        success_idxs = np.where(dones)[0]
        log_data["eval/success_action_noise_dist"] = np.mean([np.mean(action_dists[e]) for e in success_idxs])

    log_data.update({
        "eval/success": np.mean(dones),
        "eval/ep_len": np.mean(ep_lengths),
        "eval/time": time.time() - start,
    })

    if owns_env:
        env.close()


def evaluate_policy_robot(
    cfg: object,
    eval_env: object,
    model: object,
    log_data: dict[str, object],
    step: int,
    image_encoder: object | None = None,
) -> None:
    """
    Evaluate policy performance across multiple robot embodiments.

    Args:
        cfg: Configuration object
        eval_env: Robot environment to evaluate on
        model: Model to evaluate
        log_data: Dictionary to store logging data
        step: Current training step
        image_encoder: Observation image encoder
    """
    n_evals = cfg.n_evals
    n_viz = min(getattr(cfg, 'n_viz', 1), n_evals)
    action_exec_len = cfg.action_exec_len


    all_rewards = []
    all_steps = []
    all_success = []
    all_logp_a = []
    all_logp_t = []
    all_logp_p = []
    all_timesteps = []  # per-step timestep values across all episodes

    frames = [[] for _ in range(n_viz)] if cfg.render_videos else None
    for episode_idx in tqdm(range(n_evals), dynamic_ncols=True):
        sum_reward = 0
        is_done = False
        obs, _ = eval_env.reset()
        step_count = 0
        ep_logp_a = []
        ep_logp_t = []
        ep_logp_p = []
        final_is_success = False

        ep_timesteps = []
        while not is_done:
            raw_obs = obs
            obs = embed_images(obs, image_encoder)
            inputs = torch.cat([to_tensor(obs['state'][None, :]), to_tensor(obs['observation/image_emb'])], dim=-1)
            with torch.no_grad():
                (action_noise, timesteps, context_noise), (log_prob_a, log_prob_t, log_prob_p), _ = (model.actor.get_action(inputs))

            action_noise = action_noise.cpu().detach().numpy()
            if cfg.method == 'tmrl':
                timesteps = timesteps.cpu().detach().numpy()
                context_noise = context_noise.cpu().detach().numpy()
                ep_timesteps.append(float(timesteps.mean()))

            # Infer actions
            if cfg.debug:
                actions = np.ones((action_exec_len, cfg.model.action_dim), dtype=np.float32) * 0.001
            else:
                actions = infer_actions(cfg, model, raw_obs, action_noise, timesteps, context_noise)[:action_exec_len]
            actions = np.asarray(actions)

            obs, reward, terminated, truncated, info = eval_env.step(actions)
            sum_reward += float(reward)
            is_done = bool(terminated or truncated)
            step_count += action_exec_len

            if is_done:
                final_is_success = bool(info.get('is_success', bool(terminated)))

            # Track log-probs
            if log_prob_a is not None:
                ep_logp_a.append(float(log_prob_a.detach().cpu().mean()))
            if cfg.method == 'tmrl':
                if log_prob_t is not None:
                    ep_logp_t.append(float(log_prob_t.detach().cpu().mean()))
                if log_prob_p is not None:
                    ep_logp_p.append(float(log_prob_p.detach().cpu().mean()))

            # Video frames
            frame = obs.get('observation/exterior_image_1_left', obs.get('observation/image'))
            if frame is not None and episode_idx < n_viz:
                frames[episode_idx].append(frame)

        all_rewards.append(sum_reward)
        all_steps.append(step_count)
        all_success.append(final_is_success)
        if ep_logp_a:
            all_logp_a.append(np.mean(ep_logp_a))
        if ep_logp_t:
            all_logp_t.append(np.mean(ep_logp_t))
        if ep_logp_p:
            all_logp_p.append(np.mean(ep_logp_p))
        if ep_timesteps:
            all_timesteps.append(ep_timesteps)  # keep per-episode sequences

    # Compute and log statistics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = np.mean(all_success)
    avg_steps = np.mean(all_steps)

    log(
        f'eval at step {step} — reward: {avg_reward:.3f} ± {std_reward:.3f} | '
        f'success: {success_rate:.1%} ({sum(all_success)}/{n_evals}) | steps: {avg_steps:.1f}'
    )
    if all_logp_a:
        log(f'  Avg logp_action: {np.mean(all_logp_a):.3f}')
    if all_logp_t:
        log(f'  Avg logp_timestep: {np.mean(all_logp_t):.3f}')
    if all_logp_p:
        log(f'  Avg logp_context: {np.mean(all_logp_p):.3f}')
    flat_timesteps = [t for ep in all_timesteps for t in ep]
    if flat_timesteps:
        log(f'  Timesteps — mean: {np.mean(flat_timesteps):.3f}  std: {np.std(flat_timesteps):.3f}  min: {np.min(flat_timesteps):.3f}  max: {np.max(flat_timesteps):.3f}')
        for i, ep_ts in enumerate(all_timesteps):
            log(f'    ep {i}: {[f"{t:.3f}" for t in ep_ts]}')

    log_data.update({
        'eval/avg_reward': avg_reward,
        'eval/success_rate': success_rate,
        'eval/avg_steps': avg_steps,
        **({'eval/logp_action': float(np.mean(all_logp_a))} if all_logp_a else {}),
        **({'eval/logp_timestep': float(np.mean(all_logp_t))} if all_logp_t else {}),
        **({'eval/logp_context': float(np.mean(all_logp_p))} if all_logp_p else {}),
        **({'eval/timestep_mean': float(np.mean(flat_timesteps)),
            'eval/timestep_std': float(np.std(flat_timesteps)),
            'eval/timestep_values': flat_timesteps,
            'eval/timestep_sequences': all_timesteps} if flat_timesteps else {}),
    })

    if frames is not None:
        max_len = max((len(f) for f in frames), default=0)
        for i in range(n_viz):
            if frames[i] and len(frames[i]) < max_len:
                frames[i].extend([frames[i][-1]] * (max_len - len(frames[i])))
        valid = [f for f in frames if f]
        if valid:
            ref_shape = valid[0][0].shape[:2]  # (H, W) from first frame
            def _resize_frames(
                episode_frames: list[np.ndarray], shape: tuple[int, int]
            ) -> list[np.ndarray]:
                return [cv2.resize(fr, (shape[1], shape[0])) if fr.shape[:2] != shape else fr
                        for fr in episode_frames]
            valid = [_resize_frames(f, ref_shape) for f in valid]
            arr = np.concatenate([np.stack(f) for f in valid], axis=2)
            log_data['eval/video'] = wandb.Video(arr.transpose(0, 3, 1, 2), fps=10, format='mp4')
