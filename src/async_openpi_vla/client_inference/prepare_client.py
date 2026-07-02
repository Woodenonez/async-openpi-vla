from typing import Deque, Literal
import time
from concurrent.futures import Future
from dataclasses import dataclass
import pathlib

import imageio
import numpy as np

from openpi_client import image_tools # type: ignore
from openpi_client_extend import WebsocketClientPolicyExtend

from libero import benchmark # type: ignore
from robosuite.utils.transform_utils import quat2axisangle # type: ignore
from libero.utils.utils import get_libero_env # type: ignore


@dataclass
class StampedAction:
    action: np.ndarray
    init_step: int
    init_time: float
    sche_step: int
    sche_time: float
    exec: bool

    @classmethod
    def from_action(cls, action: np.ndarray, init_step: int, init_time: float, offset: int, current_step: int, control_freq: int):
        """If the scheduled execution step (init_step + offset) is after the current step, mark exec as True."""
        return cls(
            action=action,
            init_step=init_step,
            init_time=init_time,
            sche_step=init_step + offset,
            sche_time=init_time + offset * (1.0 / control_freq),
            exec=True if init_step + offset >= current_step else False, 
        )

@dataclass
class PendingResult:
    future: Future | None
    _submit_step: int = 0
    _submit_time: float = 0.0

    @property
    def pending(self) -> bool:
        return self.future is not None and not self.future.done()

    @property
    def submit_step(self) -> int:
        return self._submit_step
    
    @property
    def submit_time(self) -> float:
        return self._submit_time
    
    def reset(self):
        self.future = None
        self._submit_step = 0
        self._submit_time = 0.0
    
    def set(self, future: Future, submit_step: int, submit_time: float):
        self.future = future
        self._submit_step = submit_step
        self._submit_time = submit_time


@dataclass
class Args:
    # C/S Parameters
    host: str = "localhost"
    port: int = 8888
    # Task parameters
    task_suite_name: str = "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_id: int = 0
    init_state_id: int = 0
    num_steps: int = 600
    control_freq: int = 20 # Hz, 
    # Inference parameters
    rt: bool = False # real-time inference
    infer_by: str = "queue"  # "time" or "queue"
    infer_period_s: float = 0.20
    infer_when_queue_below: int = 8
    max_queue_step: int = 10
    video_out_path: str = "outputs/videos"
    video_format: Literal["mp4", "gif"] = "mp4"


class PieceTimer:
    """A timer to measure a piece of code's execution.

    Funcs:
        __call__: Return the passed time in [sec].
        reset: Reset the timer to the current time.
    """
    def __init__(self) -> None:
        self._instant = time.perf_counter()

    def __call__(self, round_decimals:int=4, ms=False, reset=False) -> float:
        if ms:
            res = round((time.perf_counter()-self._instant)*1000, round_decimals)
        else:
            res = round(time.perf_counter()-self._instant, round_decimals)
        if reset:
            self.reset()
        return res

    def reset(self):
        self._instant = time.perf_counter()

class RealTimePacer:
    def __init__(self, control_freq: int, autoset=True) -> None:
        self.hz = control_freq
        self.dt = 1.0 / control_freq
        self._t_next: float | None = None
        if autoset:
            self.reset()

    def reset(self):
        self._t_next = time.perf_counter() + self.dt

    def sleep_until_next(self):
        over_time = 0.0
        if self._t_next is None:
            self.reset()
            return over_time
        now = time.perf_counter()
        if now < self._t_next:
            time_to_sleep = self._t_next - now
            time.sleep(time_to_sleep)
        else:
            over_time = now - self._t_next
            self.reset() # avoid accumulating delay
        self._t_next += self.dt
        return over_time


def get_task_env(task_suite_name: str, task_id=0, control_freq=20, render=True):
    task_suite = benchmark.get_task_suite(task_suite_name)
    try:
        task = task_suite.get_task(task_id)
    except IndexError as e:
        raise ValueError(f"Invalid task_id {task_id} for {task_suite_name} (max={task_suite.n_tasks})") from e
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(
        task, 
        resolution=224, 
        render=render,
        horizon=1000,
        control_freq=control_freq,
    )
    print(f"Task suite: {task_suite_name}; Task ID: {task_id}")
    return env, task, task_description, initial_states


def capture_render_frame(obsv) -> np.ndarray:
    """Return agentview frame for video export (180° rotated to match training)."""
    return np.ascontiguousarray(obsv["agentview_image"][::-1, ::-1])


def build_video_path(args: Args) -> pathlib.Path:
    filename = f"{args.task_suite_name}_task{args.task_id}_init{args.init_state_id}.{args.video_format}"
    return pathlib.Path(args.video_out_path) / filename


def save_render_video(frames: list[np.ndarray], path: pathlib.Path, *, fps: int, video_format: Literal["mp4", "gif"]) -> None:
    if not frames:
        print("No frames captured; skipping video export.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if video_format == "gif":
        imageio.mimwrite(path, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimwrite(path, frames, fps=fps)
    print(f"Saved {len(frames)} frames to {path}")


def make_policy_obs(obsv, task_instruction: str):
    img = obsv["agentview_image"][::-1, ::-1]
    wrist_img = obsv["robot0_eye_in_hand_image"][::-1, ::-1]
    state = np.concatenate(
        (
            obsv["robot0_eef_pos"],
            quat2axisangle(obsv["robot0_eef_quat"]),
            obsv["robot0_gripper_qpos"],
        )
    ).astype(np.float32)

    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

def infer(client: WebsocketClientPolicyExtend, observation: dict, nbatch = 1):
    out = client.infer_batch(observation, nbatch=nbatch)
    action_chunks = out["actions"]  # batch of action_chunks, each shape: (action_horizon, action_dim)
    # vlm_fea = out["vlm_fea"]
    return action_chunks

def evolve_action_queue(action_queue: Deque[StampedAction], new_chunk: list[StampedAction]):
    if not new_chunk:
        return

    # Replanning: drop queued actions scheduled after the incoming chunk start.
    min_new_sche_step = min(action.sche_step for action in new_chunk)
    kept_actions = [action for action in action_queue if action.sche_step <= min_new_sche_step]
    if len(kept_actions) != len(action_queue):
        action_queue.clear()
        action_queue.extend(kept_actions)

    assert action_queue.maxlen is not None
    free = action_queue.maxlen - len(action_queue)
    if free <= 0:
        return
    action_queue.extend(new_chunk[:free])

def select_action_chunk(candidate_chunks: list[np.ndarray], scheduled_step: int, current_queue: Deque[StampedAction]) -> np.ndarray:
    _queue = [a.action for a in current_queue if a.sche_step >= scheduled_step]
    if not _queue:
        return candidate_chunks[0]
    
    _trunc_candidate = [c[:len(_queue)] for c in candidate_chunks]
    dists = [np.linalg.norm(c[:,:3] - np.array(_queue)[:, :3], axis=1).sum() for c in _trunc_candidate]
    selected_chunk = candidate_chunks[np.argmin(dists)]
    return selected_chunk


