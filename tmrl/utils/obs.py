import numpy as np
from tmrl.utils.common import to_tensor
from PIL import Image
import torch


def process_libero_obs(obs):
    """
    Prepare observations for the model.

    Args:
        obs: list of dicts with keys like eye/agentview images and proprio.
    Returns:
        dict with stacked images and state vectors.
    """
    obs_arr = []
    states_arr = []

    for o in obs:
        # Flip images
        eye = np.flipud(o['robot0_eye_in_hand_image'])
        agentview = np.flipud(o['agentview_image'])
        imgs = np.stack([eye, agentview], axis=0)

        # Compose state
        gripper_qpos = o['robot0_gripper_qpos']
        eef_pos = o['robot0_eef_pos']
        eef_quat = o['robot0_eef_quat']
        state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1)

        obs_arr.append(imgs)
        states_arr.append(state)

    obs_arr = np.stack(obs_arr, axis=0)
    states_arr = np.stack(states_arr, axis=0)

    # Return in the format model expects
    return {'obs': obs_arr, 'states': states_arr}


def process_ogbench_obs(ob_dict, model, dataset_name, env):
    """
    Prepares observations as model inputs for ogbench.

    Expects ob_dict to have 'observation' and 'goal' keys (from MazeInfoWrapper / CubeInfoWrapper).
    Returns (ob, state, goal, goal_emb).
    """
    ob = to_tensor(ob_dict['observation'])
    goal = to_tensor(ob_dict['goal']) if 'goal' in ob_dict else None
    state = None
    goal_emb = None

    return ob, state, goal, goal_emb


# TODO UNUSED NOW
def process_pi0_obs(obs, n_envs, obs_encoder=None):
    """Ensure observation has all required keys for Pi0's get_prefix_rep."""
    obs = dict(obs)  # Make a copy to avoid modifying original

    if 'observation/wrist_image' not in obs:
        obs['observation/wrist_image'] = np.zeros((n_envs, 224, 224, 3), dtype=np.uint8)

    if 'observation/image' not in obs:
        obs['observation/image'] = np.zeros((n_envs, 224, 224, 3), dtype=np.uint8)

    # if obs_encoder is not None:
    #     obs['observation/image'] = obs_encoder(obs['observation/image'])

    obs_dict = {
        'observation/image': obs['observation/image'],
        'observation/wrist_image': obs['observation/wrist_image'],
        'observation/state': obs['observation/state'],
        'prompt': np.array(obs.get('prompt', obs.get('language_instruction', 'unknown task'))),
    }

    if obs_encoder is not None:
        image = obs['observation/image']
        image = to_tensor(image).permute(0, 3, 1, 2)
        with torch.no_grad():
            image = obs_encoder(image)
            image = image.cpu().numpy()

        obs_dict['observation/image_emb'] = image

    return obs_dict


def process_pi0_obs_simpler(obs, n_envs, instruction, obs_encoder=None):
    """Ensure observation has all required keys for Pi0's get_prefix_rep in SimplerEnv."""
    from simpler_env.policies.pi0.geometry import quat2mat, mat2euler

    obs = dict(obs)  # Make a copy to avoid modifying original

    image = Image.fromarray(obs['image']['3rd_view_camera']['rgb'])
    image = image.resize((224, 224))
    image = np.array(image)[None, :]

    if 'observation/wrist_image' in obs:
        wrist_image = obs['observation/wrist_image']
    else:
        wrist_image = np.zeros((n_envs, 224, 224, 3), dtype=np.uint8)

    # Build state like SimplerEnv's Pi0 preprocessing (widowx_bridge):
    # [xyz, rpy (from quat), gripper_openness] -> 7 dims
    if 'observation/state' in obs:
        state = obs['observation/state']
    else:

        def preprocess_widowx_proprio(eef_pos) -> np.array:
            """Convert ee rotation to the frame of top-down."""
            default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
            proprio = eef_pos
            rm_bridge = quat2mat(proprio[3:7])
            rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
            gripper_openness = proprio[7]  # 0 closed, 1 open
            return np.concatenate([proprio[:3], rpy_bridge_converted, [gripper_openness]])

        eef_pos = obs['agent']['eef_pos']
        state = preprocess_widowx_proprio(eef_pos)
        if state.ndim == 1:
            state = state[None, :]

    obs_dict = {
        'observation/image': image,
        'observation/wrist_image': wrist_image,
        'observation/state': state,
        'prompt': np.array([instruction]),
    }

    if obs_encoder is not None:
        image = to_tensor(image).permute(0, 3, 1, 2)
        with torch.no_grad():
            image = obs_encoder(image)
            image = image.cpu().numpy()

        obs_dict['observation/image_emb'] = image

    return obs_dict


@torch.no_grad()
def embed_images(obs, obs_encoder):
    # image = obs['observation/image']
    image = obs['observation/exterior_image_1_left']
    if image.ndim == 3:
        image = image[None, :]
    image = to_tensor(image).permute(0, 3, 1, 2)  # (bs, C, H, W)
    bs = image.shape[0]

    if 'observation/wrist_image_left' in obs:
        # wrist_image = obs['observation/wrist_image']
        wrist_image = obs['observation/wrist_image_left']
        if wrist_image.ndim == 3:
            wrist_image = wrist_image[None, :]
        wrist_image = to_tensor(wrist_image).permute(0, 3, 1, 2)

        # stack along batch dimension
        both = torch.cat([image, wrist_image], dim=0)  # (2*bs, C, H, W)

        # single forward pass
        emb = obs_encoder(both)  # (2*bs, D)

        # split + combine
        emb = emb.view(2, bs, -1)  # (2, bs, D)
        combined_emb = torch.cat([emb[0], emb[1]], dim=-1)  # (bs, 2D)
    else:
        # only main image
        combined_emb = obs_encoder(image)  # (bs, D)

    obs['observation/image_emb'] = combined_emb
    return obs
