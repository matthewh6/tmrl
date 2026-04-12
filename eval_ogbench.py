"""
Evaluate OGBench flow policies: pass@K measurement for 3 modes.

1. Normal flow policy (dsrl): standard sampling with random noise
2. Post-BC (postbc): same as dsrl (different checkpoint, no tcont_context)
3. TMRL at fixed timestep: ContextSmoothedFlowPolicy with fixed tcont_context (swept over multiple values)

Procedure:
  - For each of num_seeds initial states (fixed via env.reset(seed=...)):
    - Run K_max rollouts with different random action noise (parallelized across n_parallel envs)
    - Record success/failure for each rollout
  - Compute pass@K for all K <= K_max in post-processing:
    pass@K = (# states where at least 1 of first K rollouts succeeded) / num_seeds
  - Same initial states are used across all methods for fair comparison.

Usage:
    python eval_ogbench.py --ckpt_dsrl /path --ckpt_postbc /path --ckpt_tmrl /path \
        --dataset pointmaze-giant-navigate-v0 --num_seeds 50 --k_max 20 --n_parallel 10
"""

import argparse
import glob
import json
import os
import time
from functools import partial

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tmrl.utils.common import load_checkpoint, to_tensor, set_seed
from tmrl.utils.env import create_env, MazeInfoWrapper, CubeInfoWrapper
from tmrl.utils.logging import cprint

log = lambda msg, color='bright_cyan': cprint(msg, color)


def save_video(frames, path, fps=15):
    """Save a list of (H, W, 3) uint8 frames as an mp4 video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_single_env(dataset_name, goal_dim=0):
    """Create a single (non-vectorized) env for sequential evaluation."""
    is_cube = 'cube' in dataset_name
    is_maze = 'pointmaze' in dataset_name

    wrappers = []
    if is_cube:
        wrappers.append(partial(CubeInfoWrapper, goal_dim=goal_dim))
    elif is_maze:
        wrappers.append(MazeInfoWrapper)

    env = create_env(
        dataset_name=dataset_name,
        seed=0,
        wrappers=wrappers,
        height=128,
        width=128,
    )
    return env


def run_single_episode(model, env, action_exec_len, max_ep_steps, dataset_name,
                       mode, tcont_context_value, obs_dim, goal_dim, seed,
                       record=False):
    """Run a single episode. Returns (success, frames) where frames is [] if not recording."""
    is_cube = 'cube' in dataset_name
    has_context = hasattr(model, 'context_dim')

    obs, info = env.reset(seed=seed)
    done = False
    steps = 0
    frames = []

    if record:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while not done and steps < max_ep_steps:
        with torch.no_grad():
            obs_t = to_tensor(obs[None])

            if is_cube and obs_dim is not None:
                raw_obs = obs_t[:, :obs_dim]
                goal = to_tensor(info['goal'][None]) if 'goal' in info else None
            else:
                raw_obs = obs_t
                goal = None

            action_noise = torch.randn(
                (1, model.action_len, model.action_dim), device=raw_obs.device
            )

            if mode in ['dsrl', 'postbc'] or not has_context:
                actions = model.sample(obs=raw_obs, goals=goal, action_noise=action_noise)
            elif mode == 'tmrl':
                context_noise = torch.randn(
                    (1, model.context_dim), device=raw_obs.device
                )
                tcont = torch.full((1,), tcont_context_value, device=raw_obs.device)
                actions = model.sample(
                    obs=raw_obs, goals=goal, tcont_context=tcont,
                    action_noise=action_noise, context_noise=context_noise,
                )

        actions_np = actions.cpu().numpy()[0]  # (action_len, action_dim)

        for i in range(action_exec_len):
            if done:
                break
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            done = terminated or truncated

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if info.get('success', False):
                return True, frames
            steps += 1

    return False, frames


def run_parallel_rollouts(model, envs, n_rollouts, action_exec_len, max_ep_steps,
                          dataset_name, mode, tcont_context_value, obs_dim, goal_dim,
                          seed, n_parallel, record_n=0):
    """
    Run n_rollouts episodes in parallel using a list of envs.
    All envs reset to the same seed for the same initial state.
    Returns (successes, recorded_frames) where:
      - successes: boolean array of shape (n_rollouts,)
      - recorded_frames: dict mapping rollout_idx -> list of frames (only for first record_n rollouts)
    """
    is_cube = 'cube' in dataset_name
    has_context = hasattr(model, 'context_dim')
    successes = np.zeros(n_rollouts, dtype=bool)
    recorded_frames = {}

    for batch_start in range(0, n_rollouts, n_parallel):
        batch_size = min(n_parallel, n_rollouts - batch_start)

        # Reset all envs in this batch to the same seed
        obs_list = []
        info_list = []
        for e in range(batch_size):
            o, inf = envs[e].reset(seed=seed)
            obs_list.append(o)
            info_list.append(inf)

        done = np.zeros(batch_size, dtype=bool)
        steps = 0

        # Set up frame recording
        recording = {}
        if record_n > 0:
            for e in range(batch_size):
                rollout_idx = batch_start + e
                if rollout_idx < record_n:
                    recording[e] = []
                    frame = envs[e].render()
                    if frame is not None:
                        recording[e].append(frame)

        step_pbar = tqdm(total=max_ep_steps, desc=f'  batch {batch_start // n_parallel}',
                         leave=False, dynamic_ncols=True)
        while not all(done) and steps < max_ep_steps:
            # Batch observations into tensors
            with torch.no_grad():
                obs_batch = to_tensor(np.stack(obs_list))  # (batch_size, obs_dim)

                if is_cube and obs_dim is not None:
                    raw_obs = obs_batch[:, :obs_dim]
                    goals = []
                    for e in range(batch_size):
                        if 'goal' in info_list[e]:
                            goals.append(info_list[e]['goal'])
                    goal = to_tensor(np.stack(goals)) if goals else None
                else:
                    raw_obs = obs_batch
                    goal = None

                action_noise = torch.randn(
                    (batch_size, model.action_len, model.action_dim), device=raw_obs.device
                )

                if mode in ['dsrl', 'postbc'] or not has_context:
                    actions = model.sample(obs=raw_obs, goals=goal, action_noise=action_noise)
                elif mode == 'tmrl':
                    context_noise = torch.randn(
                        (batch_size, model.context_dim), device=raw_obs.device
                    )
                    tcont = torch.full((batch_size,), tcont_context_value, device=raw_obs.device)
                    actions = model.sample(
                        obs=raw_obs, goals=goal, tcont_context=tcont,
                        action_noise=action_noise, context_noise=context_noise,
                    )

            actions_np = actions.cpu().numpy()  # (batch_size, action_len, action_dim)

            for i in range(action_exec_len):
                if all(done):
                    break

                for e in range(batch_size):
                    if done[e]:
                        continue
                    obs_e, reward, terminated, truncated, info_e = envs[e].step(actions_np[e, i])
                    obs_list[e] = obs_e
                    info_list[e] = info_e

                    if e in recording:
                        frame = envs[e].render()
                        if frame is not None:
                            recording[e].append(frame)

                    if terminated or truncated or info_e.get('success', False):
                        done[e] = True
                        if info_e.get('success', False):
                            successes[batch_start + e] = True

                steps += 1
                step_pbar.update(1)
                n_done = int(done.sum())
                step_pbar.set_postfix(done=f'{n_done}/{batch_size}')
                if steps >= max_ep_steps:
                    break
        step_pbar.close()

        # Move recorded frames to output dict
        for e, frames in recording.items():
            recorded_frames[batch_start + e] = frames

    return successes, recorded_frames


def evaluate_mode(model, envs, args, mode_name, mode_type, tcont_val,
                  obs_dim, goal_dim, max_ep_steps, n_parallel,
                  video_dir=None, n_videos=3):
    """Evaluate a single mode across all seeds with parallel rollouts."""
    successes = np.zeros((args.num_seeds, args.k_max), dtype=bool)
    t0 = time.time()

    log(f'\n{"=" * 60}')
    log(f'Mode: {mode_name}' + (f' (tcont_context={tcont_val})' if tcont_val is not None else ''))
    log(f'{"=" * 60}')

    pbar = tqdm(range(args.num_seeds), desc=f'{mode_name}', dynamic_ncols=True)
    for state_idx in pbar:
        seed = args.seed + state_idx

        # Record videos for the first n_videos seeds
        record = video_dir is not None and state_idx < n_videos

        seed_successes, recorded_frames = run_parallel_rollouts(
            model=model, envs=envs, n_rollouts=args.k_max,
            action_exec_len=args.action_exec_len,
            max_ep_steps=max_ep_steps,
            dataset_name=args.dataset,
            mode=mode_type,
            tcont_context_value=tcont_val,
            obs_dim=obs_dim, goal_dim=goal_dim,
            seed=seed, n_parallel=n_parallel,
            record_n=3 if record else 0,
        )
        successes[state_idx] = seed_successes

        # Save videos
        if record and recorded_frames:
            for k_idx, frames in recorded_frames.items():
                status = 'success' if seed_successes[k_idx] else 'fail'
                # Remove any existing video for this seed/k (status may differ)
                for old in glob.glob(os.path.join(video_dir, f'seed{state_idx}_k{k_idx}_*.mp4')):
                    os.remove(old)
                vid_path = os.path.join(video_dir, f'seed{state_idx}_k{k_idx}_{status}.mp4')
                save_video(frames, vid_path)

        # Update progress bar with running stats
        completed = state_idx + 1
        s = successes[:completed]
        parts = []
        for K in [1, 5, 10, 20]:
            if K <= args.k_max:
                parts.append(f'p@{K}={s[:, :K].any(axis=1).mean():.2f}')
        pbar.set_postfix_str(', '.join(parts))

    elapsed = time.time() - t0

    # Compute pass@K for all K <= k_max
    success_at_k = {}
    for K in range(1, args.k_max + 1):
        any_success = successes[:, :K].any(axis=1)
        success_at_k[K] = float(any_success.mean())

    raw_success = float(successes.mean())
    result = {
        'success_at_k': success_at_k,
        'raw_success': raw_success,
        'successes': successes.tolist(),
        'time': elapsed,
    }

    log(f'  Raw success rate: {raw_success:.4f}')
    for K in [1, 5, 10, 20]:
        if K in success_at_k:
            log(f'  pass@{K}: {success_at_k[K]:.4f}')
    log(f'  Time: {elapsed:.1f}s')

    return result


def main():
    parser = argparse.ArgumentParser(description='OGBench pass@K evaluation')
    parser.add_argument('--ckpt_dsrl', type=str, default=None,
                        help='Checkpoint for normal flow policy (dsrl mode)')
    parser.add_argument('--ckpt_postbc', type=str, default=None,
                        help='Checkpoint for post-BC flow policy')
    parser.add_argument('--ckpt_tmrl', type=str, default=None,
                        help='Checkpoint for tmrl (ContextSmoothedFlowPolicy)')
    parser.add_argument('--dataset', type=str, default='pointmaze-giant-navigate-v0')
    parser.add_argument('--num_seeds', type=int, default=50, help='Number of initial states')
    parser.add_argument('--k_max', type=int, default=20, help='Max K (rollouts per initial state)')
    parser.add_argument('--n_parallel', type=int, default=20,
                        help='Number of parallel envs')
    parser.add_argument('--action_exec_len', type=int, default=5, help='Open-loop execution length')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for initial states')
    parser.add_argument('--tmrl_tcont_values', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='List of tcont_context values to sweep for tmrl mode')
    parser.add_argument('--n_videos', type=int, default=3,
                        help='Number of seeds to record videos for (0 to disable)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset properties
    is_cube = 'cube' in args.dataset
    is_maze = 'pointmaze' in args.dataset

    if is_maze:
        obs_dim, goal_dim, max_ep_steps = 4, 0, 1000
    elif is_cube:
        obs_dim, goal_dim, max_ep_steps = 28, 3, 200
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # Build modes: (mode_name, ckpt_path, mode_type, tcont_val)
    modes = []
    if args.ckpt_dsrl:
        modes.append(('dsrl', args.ckpt_dsrl, 'dsrl', None))
    if args.ckpt_postbc:
        modes.append(('postbc', args.ckpt_postbc, 'postbc', None))
    if args.ckpt_tmrl:
        for tcont_val in args.tmrl_tcont_values:
            modes.append((f'tmrl_t{tcont_val:.2f}', args.ckpt_tmrl, 'tmrl', tcont_val))

    if not modes:
        log('ERROR: Provide at least one checkpoint (--ckpt_dsrl, --ckpt_postbc, or --ckpt_tmrl)')
        return

    n_parallel = min(args.n_parallel, args.k_max)
    log(f'Evaluating {args.num_seeds} seeds x {args.k_max} rollouts | {n_parallel} parallel envs')

    # Video output dir
    out_base = f'eval_ogbench_{args.dataset}_seeds{args.num_seeds}_kmax{args.k_max}'
    video_base = out_base + '_videos'
    if args.n_videos > 0:
        os.makedirs(video_base, exist_ok=True)

    # Create parallel envs (list of independent gymnasium envs, reused across modes)
    log(f'Creating {n_parallel} parallel envs...')
    envs = [make_single_env(args.dataset, goal_dim=goal_dim) for _ in range(n_parallel)]

    results = {}
    prev_ckpt = None
    model = None

    for mode_name, ckpt_path, mode_type, tcont_val in modes:
        # Only reload model when checkpoint changes
        if ckpt_path != prev_ckpt:
            log(f'Loading checkpoint from {ckpt_path}')
            model = load_checkpoint(ckpt_path, device=device)
            has_context = hasattr(model, 'context_dim')
            log(f'Model type: {"ContextSmoothedFlowPolicy" if has_context else "FlowPolicy"}')
            prev_ckpt = ckpt_path

        # Per-mode video dir
        video_dir = None
        if args.n_videos > 0:
            video_dir = os.path.join(video_base, mode_name)
            os.makedirs(video_dir, exist_ok=True)

        result = evaluate_mode(model, envs, args, mode_name, mode_type, tcont_val,
                               obs_dim, goal_dim, max_ep_steps, n_parallel,
                               video_dir=video_dir, n_videos=args.n_videos)
        result['ckpt'] = ckpt_path
        results[mode_name] = result

    for env in envs:
        env.close()

    # Summary
    log(f'\n{"=" * 60}')
    log('SUMMARY')
    log(f'{"=" * 60}')
    for mode_name, mode_results in results.items():
        log(f'\n{mode_name} ({mode_results.get("ckpt", "")}):')
        log(f'  Raw success: {mode_results["raw_success"]:.4f}')
        for K in [1, 5, 10, 20]:
            if K in mode_results['success_at_k']:
                log(f'  pass@{K}: {mode_results["success_at_k"][K]:.4f}')

    # Save results (merge with existing)
    json_path = f'{out_base}.json'
    npz_path = f'{out_base}.npz'

    # Load existing results if present
    json_results = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            json_results = json.load(f)
    npz_data = {}
    if os.path.exists(npz_path):
        npz_data = dict(np.load(npz_path))

    # Update with new results (overwrite existing modes, add new ones)
    for mode, data in results.items():
        json_results[mode] = {
            'success_at_k': {str(k): v for k, v in data['success_at_k'].items()},
            'raw_success': data['raw_success'],
            'time': data['time'],
            'ckpt': data.get('ckpt', ''),
        }
        npz_data[f'{mode}_successes'] = np.array(data['successes'])

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    np.savez(npz_path, **npz_data)
    log(f'\nResults saved to {json_path} and {npz_path}')


if __name__ == '__main__':
    main()
