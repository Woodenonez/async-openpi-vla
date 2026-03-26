from typing import Optional
import os
import xml.etree.ElementTree as ET

import numpy as np

import robosuite
from robosuite.utils.mjcf_utils import find_elements
from robosuite.utils.transform_utils import quat2axisangle

from libero import get_libero_path
from libero.envs import OffScreenRenderEnv
from libero.envs.env_wrapper import ControlEnv

DIR = os.path.dirname(__file__)


TASK_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


def postprocess_model_xml(xml_str, cameras_dict={}):
    """
    This function postprocesses the model.xml collected from a MuJoCo demonstration
    in order to make sure that the STL files can be found.

    Args:
        xml_str (str): Mujoco sim demonstration XML file as string

    Returns:
        str: Post-processed xml file as string
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # replace mesh and texture file paths
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" not in old_path_split:
            continue
        ind = max(
            loc for loc, val in enumerate(old_path_split) if val == "robosuite"
        )  # last occurrence index
        new_path_split = path_split + old_path_split[ind + 1 :]
        new_path = "/".join(new_path_split)
        elem.set("file", new_path)

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")
    return ET.tostring(root, encoding="utf8").decode("utf8")


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.0


def reconstruct_image_output(img_array):
    # return (img_array + 1.) / 2. * 255.
    return img_array * 255.0


def update_env_kwargs(env_kwargs, **kwargs):
    for (k, v) in kwargs.items():
        env_kwargs[k] = v

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

def get_libero_obsv_image(obs, view: str) -> np.ndarray:
    """Extracts image from observations and preprocesses it.
    
    Args:
        view: supported views: ['agentview_image', 'robot0_eye_in_hand_image']
    """
    view_list = ['agentview_image', 'robot0_eye_in_hand_image']
    assert view in view_list, f"View {view} not in supported views: {view_list}"
    img = obs[view][::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

def prepare_libero_observation(obsv, wrist_image=False):
    """Prepare observation for policy input."""
    img = get_libero_obsv_image(obsv, view="agentview_image")
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (obsv["robot0_eef_pos"], quat2axisangle(obsv["robot0_eef_quat"]), obsv["robot0_gripper_qpos"])
        ),
    }
    if wrist_image:
        wrist_img = get_libero_obsv_image(obsv, view="robot0_eye_in_hand_image")
        observation["wrist_image"] = wrist_img
    return observation, img

def get_libero_env(task, resolution=256, render=False, horizon:Optional[int]=None, control_freq=20):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if render:
        env_args.update(
            {
                "use_camera_obs": True,
                "has_renderer": True,
                "has_offscreen_renderer": True,
                "render_camera": "agentview",
                "control_freq": control_freq,
                "horizon": 200 if horizon is None else horizon,
            }
        )
        env = ControlEnv(**env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.set_seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

