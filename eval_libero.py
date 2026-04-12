"""
Evaluate LIBERO pi0 policies: pass@K measurement for 3 modes.

1. Normal flow policy (dsrl): standard pi0 sampling with random action noise
2. Post-BC (postbc): pi0 sampling (same as dsrl, different checkpoint)
3. TMRL at fixed timestep: pi0 sampling with fixed tcont_context + context_noise

Procedure:
  - For each of num_seeds initial states (fixed via env np_random seeding):
    - Run K_max rollouts with different random action noise (parallelized across n_parallel envs)
    - Record success/failure for each rollout
  - Compute pass@K for all K <= K_max in post-processing:
    pass@K = (# states where at least 1 of first K rollouts succeeded) / num_seeds
  - Same initial states are used across all methods for fair comparison.

Usage:
    python eval_libero.py --ckpt_dsrl /path/to/pi0 --ckpt_postbc /path/to/postbc --ckpt_tmrl /path/to/tmrl \
        --dataset libero_spatial --task_id 0 --num_seeds 50 --k_max 20 --n_parallel 20
"""

import argparse
import glob
import json
import os
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cv2
import numpy as np
import torch
from tqdm import tqdm

from libero.libero.envs import SubprocVectorEnv
from tmrl.utils.common import load_pi0_model, set_seed
from tmrl.utils.env import make_libero_env
from tmrl.utils.logging import cprint

log = lambda msg, color='bright_cyan': cprint(msg, color)

ENV_ACTION_DIM = 7
POLICY_ACTION_DIM = 32


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


class FakeCfg:
    """Minimal config object to satisfy load_pi0_model."""

    def __init__(self, ckpt, dataset_name, task_id):
        self.ckpt = ckpt
        self.dataset = type('D', (), {
            'name': dataset_name,
            'task_id': task_id,
            'action_dim': 7,
            'max_ep_steps': 300,
        })()
        self.paths = type('P', (), {'ckpt_dir': ''})()


def create_parallel_envs(dataset, task_id, n_parallel):
    env_fns = [
        lambda i=i: make_libero_env(dataset, task_id=task_id, height=256, width=256)
        for i in range(n_parallel)
    ]
    envs = SubprocVectorEnv(env_fns)
    envs.reset()
    return envs


def run_parallel_rollouts(policy, envs, n_rollouts, action_exec_len, max_ep_steps,
                          mode, tcont_context_value, num_steps_wait, n_parallel,
                          record_env_id=None):
    """
    Run n_rollouts episodes in parallel using envs (SubprocVectorEnv).
    All envs should already be seeded to the same initial state before calling this.
    Returns (successes, recorded_frames) where:
      - successes: boolean array of shape (n_rollouts,)
      - recorded_frames: dict mapping rollout_idx -> list of frames (only for record_env_id)
    """
    successes = np.zeros(n_rollouts, dtype=bool)
    recorded_frames = {}

    for batch_start in range(0, n_rollouts, n_parallel):
        batch_size = min(n_parallel, n_rollouts - batch_start)

        envs.reset(list(range(batch_size)))

        # Stabilization
        dummy = np.zeros((n_parallel, ENV_ACTION_DIM))
        for _ in range(num_steps_wait):
            obs, _, _, _, _ = envs.step(dummy)

        done = np.zeros(n_parallel, dtype=bool)
        steps = 0

        # Set up frame recording for envs in this batch
        recording = {}
        if record_env_id is not None:
            for e in range(batch_size):
                recording[e] = []
                # Capture initial frame
                frame = obs['observation/image'][e]
                frame = np.clip(frame.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
                recording[e].append(frame)

        step_pbar = tqdm(total=max_ep_steps, desc=f'  batch {batch_start//n_parallel}',
                         leave=False, dynamic_ncols=True)
        while not all(done[:batch_size]) and steps < max_ep_steps:
            action_noise = np.random.randn(n_parallel, POLICY_ACTION_DIM).astype(np.float32)

            if mode in ['dsrl', 'postbc']:
                result = policy.infer(obs=obs, action_noise=action_noise)
            elif mode == 'tmrl':
                tcont = np.full((n_parallel, 1), tcont_context_value, dtype=np.float32)
                ctx_noise = np.random.randn(n_parallel, 2048).astype(np.float32)
                result = policy.infer(
                    obs=obs, action_noise=action_noise,
                    tcont_context=tcont, context_noise=ctx_noise,
                )

            actions = result['actions']  # (n_parallel, action_horizon, 7)

            for i in range(action_exec_len):
                if all(done[:batch_size]):
                    break

                step_actions = np.array(actions[:, i, :])
                step_actions[done] = 0.0

                obs, reward, done_step, _, info = envs.step(step_actions)

                # Record frames
                if recording:
                    for e in range(batch_size):
                        if not done[e] and e in recording:
                            frame = obs['observation/image'][e]
                            frame = np.clip(frame.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
                            recording[e].append(frame)

                newly_done = []
                for e in range(batch_size):
                    if not done[e] and done_step[e]:
                        done[e] = True
                        successes[batch_start + e] = True
                        newly_done.append(e)

                if newly_done:
                    envs.reset(newly_done)

                steps += 1
                step_pbar.update(1)
                n_done = int(done[:batch_size].sum())
                step_pbar.set_postfix(done=f'{n_done}/{batch_size}')
                if steps >= max_ep_steps:
                    break
        step_pbar.close()

        # Move recorded frames to output dict
        for e, frames in recording.items():
            recorded_frames[batch_start + e] = frames

    return successes, recorded_frames


def evaluate_mode(policy, envs, args, mode, tcont_context_value, n_parallel,
                   video_dir=None, n_videos=3):
    """Evaluate a single mode across all seeds with parallel rollouts."""
    successes = np.zeros((args.num_seeds, args.k_max), dtype=bool)
    t0 = time.time()

    pbar = tqdm(range(args.num_seeds), desc=f'{mode}', dynamic_ncols=True)
    for state_idx in pbar:
        state_seed = args.seed + state_idx

        # Record videos for the first n_videos seeds
        record = video_dir is not None and state_idx < n_videos

        seed_successes, recorded_frames = run_parallel_rollouts(
            policy=policy, envs=envs, n_rollouts=args.k_max,
            action_exec_len=args.action_exec_len,
            max_ep_steps=args.max_ep_steps,
            mode=mode,
            tcont_context_value=tcont_context_value,
            num_steps_wait=args.num_steps_wait,
            n_parallel=n_parallel,
            record_env_id=0 if record else None,
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
    return {
        'success_at_k': success_at_k,
        'raw_success': raw_success,
        'successes': successes.tolist(),
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='LIBERO pass@K evaluation')
    parser.add_argument('--ckpt_dsrl', type=str, default=None,
                        help='Checkpoint for normal pi0 (dsrl mode)')
    parser.add_argument('--ckpt_postbc', type=str, default=None,
                        help='Checkpoint for post-BC pi0')
    parser.add_argument('--ckpt_tmrl', type=str, default=None,
                        help='Checkpoint for tmrl pi0')
    parser.add_argument('--dataset', type=str, default='libero_spatial')
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=50, help='Number of initial states')
    parser.add_argument('--k_max', type=int, default=20, help='Max K (rollouts per initial state)')
    parser.add_argument('--n_parallel', type=int, default=10,
                        help='Number of parallel envs (should divide k_max evenly for best efficiency)')
    parser.add_argument('--action_exec_len', type=int, default=5)
    parser.add_argument('--max_ep_steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42, help='Base seed for initial states')
    parser.add_argument('--tmrl_tcont_values', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='List of tcont_context values to sweep for tmrl mode')
    parser.add_argument('--num_steps_wait', type=int, default=25)
    parser.add_argument('--n_videos', type=int, default=3,
                        help='Number of seeds to record videos for (0 to disable)')
    parser.add_argument('--perturbation', type=str, default=None,
                        choices=['env', 'swap', 'object', 'lan', 'task'],
                        help='Perturbation type to append to dataset name (e.g., libero_goal + swap -> libero_goal_swap)')
    args = parser.parse_args()

    # Apply perturbation to dataset name
    if args.perturbation:
        args.dataset = f'{args.dataset}_{args.perturbation}'

    set_seed(args.seed)

    # Build list of (mode_name, ckpt_path, mode_type, tcont_value) to evaluate
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
    out_base = f'eval_libero_{args.dataset}_task{args.task_id}_seeds{args.num_seeds}_kmax{args.k_max}'
    video_base = os.path.join(out_base + '_videos')
    if args.n_videos > 0:
        os.makedirs(video_base, exist_ok=True)

    # Create parallel envs (reused across all modes)
    envs = create_parallel_envs(args.dataset, args.task_id, n_parallel)

    # Save results (merge with existing) -- paths defined before loop for incremental saving
    json_path = f'{out_base}.json'
    npz_path = f'{out_base}.npz'

    results = {}
    prev_ckpt = None
    policy = None

    for mode_name, ckpt_path, mode_type, tcont_val in modes:
        log(f'\n{"=" * 60}')
        log(f'Mode: {mode_name} | Checkpoint: {ckpt_path}')
        if tcont_val is not None:
            log(f'  tcont_context={tcont_val}')
        log(f'{"=" * 60}')

        # Only reload if checkpoint changes
        if ckpt_path != prev_ckpt:
            fake_cfg = FakeCfg(ckpt=ckpt_path, dataset_name=args.dataset, task_id=args.task_id)
            policy = load_pi0_model(fake_cfg)
            prev_ckpt = ckpt_path

        # Per-mode video dir
        video_dir = None
        if args.n_videos > 0:
            video_dir = os.path.join(video_base, mode_name)
            os.makedirs(video_dir, exist_ok=True)

        mode_results = evaluate_mode(policy, envs, args, mode_type, tcont_val, n_parallel,
                                     video_dir=video_dir, n_videos=args.n_videos)
        mode_results['ckpt'] = ckpt_path
        results[mode_name] = mode_results

        log(f'  Raw success rate: {mode_results["raw_success"]:.4f}')
        for K in [1, 3, 5, 10, 20]:
            if K in mode_results['success_at_k']:
                log(f'  pass@{K}: {mode_results["success_at_k"][K]:.4f}')
        log(f'  Time: {mode_results["time"]:.1f}s')

        # Incremental save after each mode
        json_results = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                json_results = json.load(f)
        npz_data = {}
        if os.path.exists(npz_path):
            npz_data = dict(np.load(npz_path))
        json_results[mode_name] = {
            'success_at_k': {str(k): v for k, v in mode_results['success_at_k'].items()},
            'raw_success': mode_results['raw_success'],
            'time': mode_results['time'],
            'ckpt': mode_results.get('ckpt', ''),
        }
        npz_data[f'{mode_name}_successes'] = np.array(mode_results['successes'])
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        np.savez(npz_path, **npz_data)
        log(f'  Results saved to {json_path}')

    envs.close()

    # Summary
    log(f'\n{"=" * 60}')
    log('SUMMARY')
    log(f'{"=" * 60}')
    for mode_name, mode_results in results.items():
        log(f'\n{mode_name} ({mode_results.get("ckpt", "")}):')
        log(f'  Raw success: {mode_results["raw_success"]:.4f}')
        for K in [1, 3, 5, 10, 20]:
            if K in mode_results['success_at_k']:
                log(f'  pass@{K}: {mode_results["success_at_k"][K]:.4f}')

    log(f'\nAll results saved to {json_path} and {npz_path}')


if __name__ == '__main__':
    main()
