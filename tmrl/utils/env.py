import gymnasium
from gymnasium import spaces
import gym
from gym.utils import seeding

import math
import numpy as np
import ogbench
import os
from functools import partial
from tmrl.utils.logging import cprint
from gymnasium.spaces import Box

# LIBERO and openpi_client are imported lazily inside make_libero_env / setup_libero_envs /
# LIBEROInfoWrapper so that importing this module (e.g. via tmrl.utils.eval) does not require
# those packages unless a LIBERO env is actually created.

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

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def _make_obs(self):
        obs = self.env.unwrapped.get_xy().astype(np.float32)
        goal = self.env.unwrapped.get_oracle_rep().astype(np.float32)
        return np.concatenate([obs, goal])

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._make_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._make_obs(), reward, terminated, truncated, info


class CubeInfoWrapper(gymnasium.Wrapper):
    """
    Wrap a cube env to return a flat obs vector [observation, goal].
    Goal is also stored in info['goal'] for separate access by the flow policy.
    """

    def __init__(self, env, goal_dim: int):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.goal_dim = goal_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim + goal_dim,), dtype=np.float32
        )

    def _get_goal(self):
        return self.env.unwrapped.cur_task_info['goal_xyzs'].flatten().astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal = self._get_goal()
        info['goal'] = goal
        return np.concatenate([obs.astype(np.float32), goal]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        goal = self._get_goal()
        info['goal'] = goal
        return np.concatenate([obs.astype(np.float32), goal]), reward, terminated, truncated, info


def _quat2axisangle(quat):
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

    def __init__(self, env):
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
        self.lang_embeds_dict = None
        self.init_states = None
        self.task_id = None
        self.task_suite = None

        self.np_random, _ = seeding.np_random(None)  # ensures we get unique random numbers for each worker

    def get_robot_state_vector(self, obs_dict):
        return np.concatenate(
            [obs_dict['robot0_eef_pos'], _quat2axisangle(obs_dict['robot0_eef_quat']), obs_dict['robot0_gripper_qpos']]
        )

    def get_data_robot_state_vector(self, obs_dict):
        return np.concatenate(
            [obs_dict['robot0_gripper_qpos'], obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_quat']]
        )

    def reset(self, **kwargs):
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

    def step(self, action):
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

    def close(self):
        self.env.close()


def make_libero_env(dataset_name, task_id=0, height=256, width=256, wrap_env=True,**kwargs):
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
    env.init_states = task_suite.get_task_init_states(task_id)

    return env


def create_env(
    dataset_name,
    seed=42,
    wrappers=None,
    task_id=None,
    height=256,
    width=256,
    **kwargs,
):
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
    cfg, n_envs, dataset_name, task_id=None, height=256, width=256, **kwargs
):
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


def setup_libero_envs(cfg, n_envs, dataset_name, height=128, width=128, **kwargs):
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


def project_world_to_image(p_world, dataset_name, env=None, cam_name=None, img_shape=(128, 128)):
    """
    Project a 3D world point to pixel coords using env camera intrinsics/extrinsics.

    If env is not provided and dataset is libero, falls back to hardcoded agentview params.
    """
    import mujoco

    # Choose default camera
    if cam_name is None:
        cam_name = 'agentview' if 'libero' in dataset_name else 'front_pixels'

    # If we have an env with sim/model, use it
    # Fallback hardcoded params for libero if no env provided
    if 'libero' in dataset_name:
        cam_pos = np.array([0.60657737, 0.0, 0.96])
        cam_mat = np.array(
            [
                [-1.72339050e-06, -5.28769744e-01, 8.48765314e-01],
                [1.00000000e00, -7.82314965e-07, 1.54309560e-06],
                [-1.51940455e-07, 8.48765314e-01, 5.28769744e-01],
            ]
        )
        fovy = 45.0
    else:
        model = env.model
        data = env.data

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        cam_pos = data.cam_xpos[cam_id]
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3)  # camera → world
        fovy = model.cam_fovy[cam_id]

    p_cam = cam_mat.T @ (p_world - cam_pos)
    if p_cam[2] >= 0:
        return None, None

    H, W = img_shape
    f = 0.5 * H / np.tan(0.5 * np.deg2rad(fovy))

    depth = -p_cam[2]
    if 'libero' in dataset_name:
        u = (-f * p_cam[0] / depth) + W / 2
        v = (-f * p_cam[1] / depth) + H / 2
    else: # ogbench
        u = (f * p_cam[0] / depth) + W / 2
        v = (-f * p_cam[1] / depth) + H / 2

    return u, v
