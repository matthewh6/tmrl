import gymnasium
import gym
from gym.utils import seeding
from typing import Callable

import math
import numpy as np
import ogbench
import os
from functools import partial
from tmrl.utils.logging import cprint
from gymnasium.spaces import Box

# Need to set the multiprocessing start method to 'spawn' to avoid the error: https://github.com/Lifelong-Robot-Learning/LIBERO/issues/3#issuecomment-1868387638
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


log = lambda msg, color='bright_cyan': cprint(msg, color)


class MazeInfoWrapper(gymnasium.Wrapper):
    """
    Wrap a pointmaze env to return a structured obs dict.
    - observation: agent (x, y) position
    - goal: oracle goal (x, y) representation
    """

    def __init__(self, env: object) -> None:
        super().__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def _make_obs(self) -> np.ndarray:
        obs = self.env.unwrapped.get_xy().astype(np.float32)
        goal = self.env.unwrapped.get_oracle_rep().astype(np.float32)
        return np.concatenate([obs, goal])

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        _, info = self.env.reset(**kwargs)
        return self._make_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._make_obs(), reward, terminated, truncated, info


class CubeInfoWrapper(gymnasium.Wrapper):
    """
    Wrap a cube env to return a flat obs vector [observation, goal].
    Goal is also stored in info['goal'] for separate access by the flow policy.
    """

    def __init__(self, env: object, goal_dim: int) -> None:
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.goal_dim = goal_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim + goal_dim,), dtype=np.float32
        )

    def _get_goal(self) -> np.ndarray:
        return self.env.unwrapped.cur_task_info['goal_xyzs'].flatten().astype(np.float32)

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        obs, info = self.env.reset(**kwargs)
        goal = self._get_goal()
        info['goal'] = goal
        return np.concatenate([obs.astype(np.float32), goal]), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal = self._get_goal()
        info['goal'] = goal
        return np.concatenate([obs.astype(np.float32), goal]), reward, terminated, truncated, info


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LIBEROInfoWrapper:
    """
    Wrap a Gym env to add a concatenated state vector from info dict.
    Returns a dict with 'state' key added to observations (or combined with existing dict).
    """

    def __init__(self, env: object) -> None:
        # Manually set self.env instead of using super().__init__() to avoid
        # gymnasium.Wrapper's check that requires env to be a gymnasium.Env
        # OffScreenRenderEnv inherits from ControlEnv, not gymnasium.Env
        self.env = env
        self.metadata = {'render_modes': ['rgb_array']}

        if not hasattr(self.env, 'action_space'):
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Overwrite the observation space
        self.observation_space = gym.spaces.Dict(
            {
                'observation/state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
                'observation/image': gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                'observation/wrist_image': gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            }
        )

        self._completed_goal_states = set()

        # metadata
        self.task_id = None
        self.task_suite = None

        self.np_random, _ = seeding.np_random(None)  # ensures we get unique random numbers for each worker

    def get_robot_state_vector(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [obs_dict['robot0_eef_pos'], _quat2axisangle(obs_dict['robot0_eef_quat']), obs_dict['robot0_gripper_qpos']]
        )

    def get_data_robot_state_vector(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [obs_dict['robot0_gripper_qpos'], obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']]
        )

    def reset(self, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        self._completed_goal_states = set()  # reset completed goal states

        self.env.seed(0)  # IMPORTANT: object locations are different if we don't seed (even for the same initial state)
        self.env.reset()

        init_states = self.task_suite.get_task_init_states(self.task_id)
        init_state = self.np_random.choice(init_states)

        obs_dict = self.env.set_init_state(init_state)

        # Get preprocessed image
        # IMPORTANT: rotate both h/v 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs_dict['agentview_image'][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs_dict['robot0_eye_in_hand_image'][::-1, ::-1])
        from openpi_client import image_tools

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        # Get preprocessed state
        state = self.get_robot_state_vector(obs_dict)
        data_state = self.get_data_robot_state_vector(obs_dict)

        return {
            'observation/state': state,
            'observation/image': img,
            'observation/wrist_image': wrist_img,
            'prompt': self.env.language_instruction,
        }, {'language_instruction': self.env.language_instruction, 'data_state': data_state}

    def step(self, action: np.ndarray) -> tuple[dict[str, object], np.float64, bool, bool, dict[str, object]]:
        obs_dict, reward, done, _ = self.env.step(action)

        # Get preprocessed image
        # IMPORTANT: rotate both h/v 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs_dict['agentview_image'][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs_dict['robot0_eye_in_hand_image'][::-1, ::-1])
        from openpi_client import image_tools

        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))

        # Get preprocessed state
        state = self.get_robot_state_vector(obs_dict)
        data_state = self.get_data_robot_state_vector(obs_dict)

        return (
            {
                'observation/state': state,
                'observation/image': img,
                'observation/wrist_image': wrist_img,
                'prompt': self.env.language_instruction,
            },
            np.float64(reward),
            done,
            False,
            {'language_instruction': self.env.language_instruction, 'data_state': data_state},
        )

    def close(self) -> None:
        self.env.close()


def make_libero_env(
    dataset_name: str,
    task_id: int = 0,
    height: int = 256,
    width: int = 256,
    wrap_env: bool = True,
    **kwargs: object,
) -> object:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = dataset_name
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

    # https://github.com/Physical-Intelligence/openpi/blob/175f89c31d1b2631a8ff3b678768f17489c5ead4/examples/libero/main.py#L85
    env_args = {
        'bddl_file_name': task_bddl_file,
        'camera_heights': height,
        'camera_widths': width,
        'has_renderer': False,           # disable on-screen renderer (required for SubprocVectorEnv / headless)
        'has_offscreen_renderer': True,  # enable GPU offscreen rendering via EGL
        'use_camera_obs': True,
    }
    env = OffScreenRenderEnv(**env_args)
    if wrap_env:
        env = LIBEROInfoWrapper(env)
    task_description = env.env.language_instruction if wrap_env else env.language_instruction

    log(
        f'Making LIBEROInfoWrapper: task_id: {task_id}, suite: {task_suite_name}, language instruction: {task_description}, task_bddl_file: {task_bddl_file}'
    )

    # Attach metadata
    env.task_id = task_id
    env.task_suite = task_suite

    return env


def create_env(
    dataset_name: str,
    seed: int = 42,
    wrappers: list[Callable[[object], object]] | None = None,
    task_id: int | None = None,
    height: int = 256,
    width: int = 256,
    **kwargs: object,
) -> object:
    """Create a single environment instance."""
    if 'pointmaze' in dataset_name:
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True, height=height, width=width, **kwargs)
        env.reset(seed=seed)
    elif 'cube' in dataset_name:
        env = ogbench.make_env_and_datasets(
            dataset_name, env_only=True, permute_blocks=False, height=height, width=width, **kwargs
        )  # permute_blocks off for fixed goals
        env.reset(seed=seed)
    elif 'libero' in dataset_name:
        env = make_libero_env(dataset_name, task_id=task_id, height=height, width=width, **kwargs)
    elif 'two' in dataset_name:  # dexmg
        env = make_dexmimicgen_env(dataset_name, seed, **kwargs)
    else:
        raise ValueError(f'Unsupported environment: {dataset_name}')

    if wrappers is not None:
        for wrapper in wrappers:
            env = wrapper(env)

    # env.reset()
    # initialize the renderer
    if 'two' in dataset_name:
        env.sim.render(height=84, width=84)
    elif 'libero' in dataset_name:
        pass  # libero does not define env.render()
    else:
        env.render()

    return env


def setup_envs(
    cfg: object,
    n_envs: int,
    dataset_name: str,
    task_id: int | None = None,
    height: int = 256,
    width: int = 256,
    **kwargs: object,
) -> object:
    log(f'Setting up environments for {dataset_name}')

    is_libero = 'libero' in dataset_name
    is_cube = 'cube' in dataset_name
    is_pointmaze = 'pointmaze' in dataset_name

    if is_libero:
        from libero.libero.envs import SubprocVectorEnv

        # Create online envs
        log(f'Creating {n_envs} online envs')
        env_fns = []
        for i in range(n_envs):
            env_kwargs = dict()
            env_fns.append(
                lambda i=i, env_kwargs=env_kwargs: create_env(
                    dataset_name=dataset_name,
                    height=height,
                    width=width,
                    task_id=task_id,
                    **env_kwargs,
                )
            )

        # env = DummyVectorEnv(env_fns)
        env = SubprocVectorEnv(env_fns)
        env.reset()

        return env

    else:  # ogbench
        wrappers = []
        if is_cube:
            wrappers.append(partial(CubeInfoWrapper, goal_dim=cfg.dataset.goal_dim))
        elif is_pointmaze:
            wrappers.append(MazeInfoWrapper)

        # Create online envs
        env_fns = [
            lambda i=i: create_env(
                dataset_name=dataset_name,
                seed=cfg.seed + i,
                wrappers=wrappers,
                height=128,
                width=128,
                **kwargs,
            )
            for i in range(cfg.n_envs)
        ]
        env = gymnasium.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED
        )  # https://farama.org/Vector-Autoreset-Mode

        # Create eval envs
        env_fns = [
            lambda i=i: create_env(
                dataset_name=dataset_name,
                seed=cfg.seed + i,
                wrappers=wrappers,
                height=128,
                width=128,
                **kwargs,
            )
            for i in range(cfg.n_evals)
        ]
        eval_env = gymnasium.vector.SyncVectorEnv(
            env_fns, autoreset_mode=gymnasium.vector.AutoresetMode.DISABLED
        )

    return env, eval_env


def setup_libero_envs(
    cfg: object,
    n_envs: int,
    dataset_name: str,
    height: int = 128,
    width: int = 128,
    **kwargs: object,
) -> object:
    """
    Inputs:
        cfg: Configuration object
        n_envs: Number of online environments
        dataset_name: Name of the dataset
        kwargs: Additional keyword arguments

    Outputs:
        env: Vectorized environment
    """
    log(f'Creating {n_envs} envs for {dataset_name} with task_id: {cfg.dataset.task_id}...')

    from libero.libero.envs import DummyVectorEnv, SubprocVectorEnv

    # Either a perturbation is provided or the dataset name is already changed
    if getattr(cfg, 'perturbation', None):
        dataset_name = dataset_name + '_' + cfg.perturbation

    env_fns = [
        lambda i=i: create_env(
            dataset_name=dataset_name,
            height=height,
            width=width,
            task_id=cfg.dataset.task_id,
            **kwargs,
        )
        for i in range(n_envs)
    ]

    if n_envs > 1:
        env = SubprocVectorEnv(env_fns)
    else:
        env = DummyVectorEnv(env_fns)

    env.reset()
    return env
