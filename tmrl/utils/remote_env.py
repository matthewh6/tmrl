"""
Remote environment wrapper for connecting to remote robot servers.
Works with both DROID sim server and WidowX real server.
"""

import socket
import pickle
import struct
import time
from urllib.parse import urlparse

import gymnasium as gym
import numpy as np

STEP_REWARD = -1
SUCCESS_REWARD = 0


def send_msg(sock: socket.socket, msg: object) -> None:
    """Send a message with length prefix over socket."""
    msg_bytes = pickle.dumps(msg)
    msg_len = struct.pack('>I', len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock: socket.socket) -> object | None:
    """Receive a length-prefixed message from socket."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return pickle.loads(recvall(sock, msglen))


def recvall(sock: socket.socket, n: int) -> bytes | None:
    """Helper to receive n bytes or return None if EOF is hit."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


DROID_IMAGE_RESOLUTION = 224
WIDOWX_IMAGE_RESOLUTION = 224


class RemoteEnv(gym.Env):
    """
    Gymnasium environment that connects to a remote robot server.

    Works with both:
    - droid_remote_server.py (DROID sim)
    - widowx_remote_server.py (WidowX real robot)

    Usage:
        env = RemoteEnv("tcp://localhost:6000")
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        server_url: str,
        obs_format: str = 'droid',  # "droid" or "widowx"
        socket_timeout: float = 30.0,
        connect_timeout: float = float('inf'),
        retry_interval: float = 5.0,
    ) -> None:
        """
        Args:
            server_url: URL of remote server (e.g., "tcp://localhost:6000")
            obs_format: Observation format - "droid" or "widowx"
            socket_timeout: Timeout for socket recv/send operations in seconds (default: 30.0)
            connect_timeout: Total timeout for initial connection in seconds (default: 300.0 = 5 min)
            retry_interval: Time between connection retry attempts in seconds (default: 5.0)
        """
        super().__init__()

        self.server_url = server_url
        self.obs_format = obs_format
        self.socket_timeout = socket_timeout
        self.connect_timeout = connect_timeout
        self.retry_interval = retry_interval

        # Attribute to control whether step() should request full observation
        # Can be set externally by rollout workers to optimize camera captures
        self.need_obs = True

        # Parse URL
        parsed = urlparse(server_url)
        self.host = parsed.hostname
        self.port = parsed.port

        if obs_format == 'droid':
            self.image_resolution = DROID_IMAGE_RESOLUTION
        elif obs_format == 'widowx':
            self.image_resolution = WIDOWX_IMAGE_RESOLUTION
        else:
            raise ValueError(f'Unknown obs_format: {obs_format}')

        if not self.host or not self.port:
            raise ValueError(f'Invalid server URL: {server_url}. Expected format: tcp://host:port')

        # Socket connection
        self.sock = None
        self._connect()

        # Do initial reset to ensure server is ready and determine observation structure
        # This blocks until server is available and ready
        obs, _ = self.reset()

        # Define observation and action spaces
        if obs_format == 'droid':
            self.dsrl_key_mapping = {
                'image': ['observation/wrist_image_left', 'observation/exterior_image_1_left'],
                # "image": ["observation/wrist_image", "observation/image"],
                'state': 'state',
                'language': 'language',
            }
            self.observation_space = gym.spaces.Dict(
                {
                    'observation/exterior_image_1_left': gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            obs['observation/exterior_image_1_left'].shape[0],
                            obs['observation/exterior_image_1_left'].shape[1],
                            3,
                        ),
                        dtype=np.uint8,
                    ),
                    # "observation/image": gym.spaces.Box(
                    #     low=0,
                    #     high=255,
                    #     shape=(
                    #         obs["observation/image"].shape[0],
                    #         obs["observation/image"].shape[1],
                    #         3,
                    #     ),
                    #     dtype=np.uint8,
                    # ),
                    'observation/wrist_image_left': gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            obs['observation/wrist_image_left'].shape[0],
                            obs['observation/wrist_image_left'].shape[1],
                            3,
                        ),
                        dtype=np.uint8,
                    ),
                    # "observation/wrist_image": gym.spaces.Box(
                    #     low=0,
                    #     high=255,
                    #     shape=(
                    #         obs["observation/wrist_image"].shape[0],
                    #         obs["observation/wrist_image"].shape[1],
                    #         3,
                    #     ),
                    #     dtype=np.uint8,
                    # ),
                    'observation/joint_position': gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                    ),
                    'observation/gripper_position': gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                    ),
                    'state': gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(8,),
                        dtype=np.float32,  # for DSRL policy
                    ),
                    'prompt': gym.spaces.Text(  # make sure this is included or SyncVectorEnv will drop it
                        min_length=1, max_length=512
                    ),
                }
            )
            self.action_dim = 8
        elif obs_format == 'widowx':
            self.dsrl_key_mapping = {
                'image': ['observation_images_image_0'],
                'state': 'state',
                'language': 'language',
            }
            self.observation_space = gym.spaces.Dict(
                {
                    'state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                    'observation_images_image_0': gym.spaces.Box(  # needed because "." not supported in pytorch moduledict for DSRL policy/critic
                        low=0, high=255, shape=(self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                    ),
                    'observation.images.image_0': gym.spaces.Box(
                        low=0, high=255, shape=(self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                    ),
                    'prompt': gym.spaces.Text(  # make sure this is included or SyncVectorEnv will drop it
                        min_length=1, max_length=512
                    ),
                }
            )
            self.action_dim = 7
        else:
            raise ValueError(f'Unknown obs_format: {obs_format}')

        prompt_space = gym.spaces.Text(min_length=1, max_length=512)
        spaces_dict = dict(self.observation_space.spaces)
        spaces_dict['prompt'] = prompt_space
        # Only add 'language' to the observation space if we actually have an embedding
        spaces_dict['language'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(384,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(spaces_dict)

        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)

    def _connect(self) -> None:
        """Connect to remote server with retries until connect_timeout."""
        import time

        start_time = time.time()

        while True:
            try:
                print(f'Connecting to robot server at {self.host}:{self.port}...')
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.socket_timeout)  # Socket timeout for recv/send operations
                self.sock.connect((self.host, self.port))
                print(f'✓ Connected to robot server at {self.host}:{self.port}')
                return
            except Exception as e:
                # Catch ALL exceptions (ConnectionRefusedError, timeout, gaierror, etc.)
                elapsed = time.time() - start_time
                if elapsed > self.connect_timeout:
                    raise ConnectionError(
                        f'Failed to connect to robot server at {self.host}:{self.port} '
                        f'after {self.connect_timeout:.0f}s: {e}'
                    )
                print(
                    f'Connection failed: {type(e).__name__}: {e}. Retrying in {self.retry_interval}s... '
                    f'(elapsed: {elapsed:.1f}s / {self.connect_timeout:.0f}s)'
                )
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                time.sleep(self.retry_interval)

    def reset(
        self, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Reset environment and return initial observation.

        This method will keep trying to reconnect until successful.
        This allows training to survive robot server restarts.

        Note: Reset uses no timeout because the robot server may wait for
        operator input (e.g., pressing Enter to start episode).
        """
        super().reset(seed=seed)

        attempt = 0
        while True:
            attempt += 1
            try:
                # Ensure we have a connection (will block until connected)
                self._ensure_connected()

                # Temporarily disable socket timeout for reset
                # The server may wait for operator input (e.g., "Press Enter to start")
                # which can take arbitrarily long
                self.sock.settimeout(None)  # Blocking mode

                # Send reset command
                send_msg(self.sock, {'type': 'RESET'})

                # Receive observation (blocks until server responds)
                obs = recv_msg(self.sock)

                # Restore socket timeout for subsequent operations
                self.sock.settimeout(self.socket_timeout)

                if obs is None or 'prompt' not in obs:
                    raise ConnectionError('Server disconnected during reset')

                # Extract prompt/instruction if present (servers may use either key)
                prompt = obs['prompt']

                # Format observation
                formatted_obs = self._format_observation(obs)

                if attempt > 1:
                    print(f'[RemoteEnv] ✓ Successfully reconnected after {attempt} attempts')

                # Extract server capabilities from response (may contain supports_action_chunking)
                server_info = obs.get('info', {})
                info = {'prompt': prompt, **server_info}
                return formatted_obs, info

            except (ConnectionError, socket.error, EOFError, OSError) as e:
                print(f'[RemoteEnv] Connection error during reset (attempt {attempt}): {e}')
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                self.sock = None

                print(f'[RemoteEnv] Waiting {self.retry_interval}s before retry...')
                print('[RemoteEnv] (Restart the robot server when ready)')
                time.sleep(self.retry_interval)

    def _ensure_connected(self) -> None:
        """Ensure we have an active connection, reconnect if needed."""
        if self.sock is None:
            self._connect()

    def send_success_check_done(self) -> tuple[bool, dict[str, object] | None, dict[str, object]]:
        """Send success signal to server, and then wait for server to confirm if this is the last step or if there are more steps to come.

        If done=False (more stages remaining), automatically sends RESET to continue to next stage.
        Returns (done: bool, obs: dict, info: dict) if reset was triggered, else (done: bool, None, None).
        """
        send_msg(self.sock, {'type': 'SUCCESS_CHECK'})
        response = recv_msg(self.sock)
        if response is None:
            print('Server disconnected during success check.')
            done = False
        else:
            done = response.get('done', False)

        if not done:
            # this is a continuation of the task, expect the response to have new formatted obs and info
            # Extract components
            reward = response.get('reward', 0.0)
            done = response.get('done', False)
            truncated = response.get('truncated', False)
            success = response.get('success', False)
            info = response.get('info', {})

            # Add success to info
            info['success'] = success
            info['is_success'] = success
            info['new_reward'] = SUCCESS_REWARD

            # Add num_steps if present (from chunked execution)
            if 'num_steps' in response:
                info['num_steps'] = response['num_steps']

            # Remove non-observation fields before formatting
            obs_dict = {
                k: v for k, v in response.items() if k not in ['reward', 'done', 'truncated', 'success', 'info']
            }

            formatted_obs = self._format_observation(obs_dict)
            return done, formatted_obs, info
        else:
            info = {'new_reward': SUCCESS_REWARD}
        return done, None, info

    def step(self, action: np.ndarray) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Execute action and return next observation.

        If connection is lost, returns a truncated episode with disconnected flag
        instead of raising an exception. This allows training to continue after
        the robot server reconnects.
        """
        self._ensure_connected()

        try:
            # Ensure action is numpy array
            action = np.array(action, dtype=np.float32)

            # Send step command
            # Use self.need_obs attribute (can be set externally by rollout workers)
            send_msg(
                self.sock,
                {
                    'type': 'STEP',
                    'action': action.tolist(),
                    'need_obs': self.need_obs,
                },
            )

            # Receive response
            response = recv_msg(self.sock)

            if response is None or 'prompt' not in response:
                raise ConnectionError('Server disconnected during step or missing prompt key')

            # Extract components
            reward = response.get('reward', 0.0)
            done = response.get('done', False)
            truncated = response.get('truncated', False)
            success = response.get('success', False)
            info = response.get('info', {})

            # Add success to info
            info['success'] = success
            info['is_success'] = success

            # Add num_steps if present (from chunked execution)
            if 'num_steps' in response:
                info['num_steps'] = response['num_steps']

            # Remove non-observation fields before formatting
            obs_dict = {
                k: v for k, v in response.items() if k not in ['reward', 'done', 'truncated', 'success', 'info']
            }

            formatted_obs = self._format_observation(obs_dict)

            # Log episode end
            if done or truncated:
                status = 'SUCCESS ✓' if success else 'FAILURE ✗'
                print(f'[RemoteEnv] Episode ended: {status} (reward={reward:.1f})')

            # Check if this is a multi-stage transition (stage succeeded but episode continues)
            # is_stage_transition = info.get('stage_complete', False) and success and not done

            if success:
                reward = SUCCESS_REWARD
            else:
                reward = STEP_REWARD
                assert not done
            return formatted_obs, reward, bool(done), bool(truncated), info

        except (ConnectionError, socket.error, EOFError, OSError) as e:
            print(f'\n[RemoteEnv] ⚠️ Connection lost during step: {e}')
            print('[RemoteEnv] Marking episode as truncated (disconnected)')
            print('[RemoteEnv] Will attempt to reconnect on next reset...\n')

            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            self.sock = None

            # Return a "disconnected" transition instead of raising
            # This allows the rollout to end gracefully and trigger a reset
            dummy_obs = self._get_dummy_observation()
            info = {
                'disconnected': True,
                'error': str(e),
                'success': False,
                'is_success': False,
            }

            # Return truncated=True to signal episode end
            return dummy_obs, STEP_REWARD, False, True, info

    def _get_dummy_observation(self) -> dict[str, object]:
        """Return a safe dummy observation when connection is lost."""
        if self.obs_format == 'droid':
            return {
                'observation/exterior_image_1_left': np.zeros(
                    (self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                ),
                'observation/wrist_image_left': np.zeros(
                    (self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                ),
                'observation/joint_position': np.zeros(7, dtype=np.float32),
                'observation/gripper_position': np.zeros(1, dtype=np.float32),
                'state': np.zeros(8, dtype=np.float32),
                'prompt': '',
            }
        else:  # widowx
            return {
                'state': np.zeros(7, dtype=np.float32),
                'observation.images.image_0': np.zeros(
                    (self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                ),
                'observation_images_image_0': np.zeros(
                    (self.image_resolution, self.image_resolution, 3), dtype=np.uint8
                ),
                'prompt': '',
            }

    def _format_observation(self, obs: dict[str, object]) -> dict[str, object]:
        """Format observation based on obs_format."""
        if self.obs_format == 'droid':
            # Already in DROID format from server
            # state is used for DSRL:
            state = np.concatenate([obs.get('observation/joint_position'), obs.get('observation/gripper_position')])
            return {
                'observation/exterior_image_1_left': np.array(obs.get('observation/exterior_image_1_left')),
                # "observation/image": np.array(obs.get("observation/exterior_image_1_left")),
                'observation/wrist_image_left': np.array(obs.get('observation/wrist_image_left')),
                # "observation/wrist_image": np.array(obs.get("observation/wrist_image_left")),
                'observation/joint_position': np.array(obs.get('observation/joint_position')),
                'observation/gripper_position': np.array(obs.get('observation/gripper_position')),
                'state': state,
                'prompt': obs['prompt'],
            }
        elif self.obs_format == 'widowx':
            # Convert to WidowX format (use base image if we have to choose 1 image input)
            return {
                'state': np.array(obs['state']),
                'observation.images.image_0': np.array(obs['observation.images.image_0']),
                'observation_images_image_0': np.array(obs['observation.images.image_0']),
                'prompt': obs['prompt'],
            }
        else:
            return obs

    def close(self) -> None:
        """Close connection to server."""
        if self.sock:
            try:
                send_msg(self.sock, {'type': 'CLOSE'})
            except:
                pass
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
            print('✓ Disconnected from remote server')

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


# Convenience function for registration
def make_remote_env(server_url: str, **kwargs: object) -> RemoteEnv:
    """Create a remote environment."""
    return RemoteEnv(server_url, **kwargs)
