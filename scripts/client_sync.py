from typing import Deque
from collections import deque

import matplotlib.pyplot as plt

from openpi_client import websocket_client_policy # type: ignore
from libero.utils.utils import get_libero_dummy_action # type: ignore

from async_openpi_vla.client_inference.prepare_client import Args, StampedAction
from async_openpi_vla.client_inference.prepare_client import get_task_env, make_policy_obs, infer
from async_openpi_vla.client_inference.prepare_client import capture_render_frame, build_video_path, save_render_video


USE_QUEUE = True

def main(args: Args):
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    action_queue: Deque[StampedAction] = deque(maxlen=args.max_queue_step)
    
    env, task, task_instruction, initial_states = get_task_env(args.task_suite_name, args.task_id, control_freq=args.control_freq, render=False)
    frames: list[np.ndarray] = []
    env.reset()

    try:
        obsv = env.set_init_state(initial_states[args.init_state_id])
    except IndexError as e:
        raise ValueError(f"Invalid init_state_id {args.init_state_id} for task_id {args.task_id} (max={len(initial_states)})") from e
    frames.append(capture_render_frame(obsv))
    for _ in range(10):
        obsv, _, _, _ = env.step(get_libero_dummy_action()) # settle environment
        frames.append(capture_render_frame(obsv))
    try:
        observation = make_policy_obs(obsv, task_instruction)
        infer(client, observation)
    except Exception as e:
        raise ValueError(f"Error during initial inference: {e}")

    print("Starting main loop...")
    print("=" * 50)
    for step in range(args.num_steps):
        print(f"Step: {step}")

        observation = make_policy_obs(obsv, task_instruction)

        if not USE_QUEUE or len(action_queue) == 0:
            out = client.infer(observation, nbatch=10)
            action_chunks = out["actions"] # one action_chunk shape: (action_horizon, action_dim)
            vlm_fea = out["vlm_fea"] # 10 x 1004, 10 is the action chunk length
            action_chunk = action_chunks[0]

        if USE_QUEUE:
            action = action_queue.popleft()
        else:
            action = action_chunk[0]

        # Execute the actions in the environment.
        obsv, reward, done, info = env.step(action)
        frames.append(capture_render_frame(obsv))

        step += 1
        if done:
            print(f"Episode success after {step + 1} timesteps")
            break

    save_render_video(
        frames,
        build_video_path(args),
        fps=args.control_freq,
        video_format=args.video_format,
    )

    plt.show()


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main(args)