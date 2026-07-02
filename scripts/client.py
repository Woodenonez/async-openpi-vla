from typing import Deque
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from openpi_client_extend import WebsocketClientPolicyExtend
from libero.utils.utils import get_libero_dummy_action # type: ignore

from async_openpi_vla.client_inference.prepare_client import Args, StampedAction
from async_openpi_vla.client_inference.prepare_client import PendingResult, PieceTimer, RealTimePacer # for real-time inference
from async_openpi_vla.client_inference.prepare_client import get_task_env, make_policy_obs, infer, evolve_action_queue
from async_openpi_vla.client_inference.prepare_client import capture_render_frame, build_video_path, save_render_video


overall_latencies: list[float] = []
ol_timer = PieceTimer()

step_time_list: list[float] = []
step_timer = PieceTimer()


def main(args: Args):
    client = WebsocketClientPolicyExtend(host=args.host, port=args.port)
    action_queue: Deque[StampedAction] = deque(maxlen=args.max_queue_step)

    if args.rt:
        real_time_pacer = RealTimePacer(control_freq=args.control_freq)
        next_infer_time = 0.0
        executor = ThreadPoolExecutor(max_workers=1)
        pending_result = PendingResult(future=None)

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

    if args.rt:
        last_action = None
        real_time_pacer.reset()

    print("Starting main loop...")
    print(f"Task: {task_instruction}")
    print("=" * 50)
    for step in range(args.num_steps):
        print(f"Step: {step}")

        if args.rt:
            step_timer.reset()
            now = time.time()
            if pending_result.future is None:
                if args.infer_by == "time" and now >= next_infer_time:
                    should_infer = True
                elif args.infer_by == "queue" and len(action_queue) < args.infer_when_queue_below:
                    should_infer = True
                else:
                    should_infer = False
                if should_infer:
                    print("Submitting inference request...")
                    observation = make_policy_obs(obsv, task_instruction)
                    ol_timer.reset()
                    _pending = executor.submit(infer, client, observation, nbatch=1)
                    pending_result.set(_pending, submit_step=step, submit_time=now)
                    next_infer_time = now + args.infer_period_s

            elif pending_result.future.done():
                try:
                    delay_steps = int(step - pending_result.submit_step)
                    delay_time = time.time() - pending_result.submit_time
                    print(f"Inference completed (delay: {delay_steps} steps, {int(delay_time*1000)} ms)")
                    action_chunk = pending_result.future.result()[0]
                    overall_latencies.append(ol_timer(ms=True))
                    stamped_action_chunk = [
                        StampedAction.from_action(
                            action=action,
                            init_step=pending_result.submit_step,
                            init_time=pending_result.submit_time,
                            offset=i,
                            current_step=step,
                            control_freq=args.control_freq,
                        ) for i, action in enumerate(action_chunk)
                    ]
                    evolve_action_queue(action_queue, [action for action in stamped_action_chunk if action.exec])
                except Exception:
                    print("Error in pending inference result.")
                finally:
                    pending_result.reset()

            if len(action_queue) > 0:
                action:list[float] = action_queue.popleft().action.tolist() # type: ignore
                last_action = action
            else: # no action queue, use dummy action with last action's gripper state
                if last_action is not None:
                    action = get_libero_dummy_action()
                    action[-1] = last_action[-1] # keep gripper state
                else:
                    action = get_libero_dummy_action()

        else:
            observation = make_policy_obs(obsv, task_instruction)

            out = client.infer_batch(observation, nbatch=10)
            action_chunks = out["actions"] # one action_chunk shape: (action_horizon, action_dim)
            vlm_fea = out["vlm_fea"] # 10 x 1004, 10 is the action chunk length
            action_chunk = action_chunks[0]
            if args.max_queue_step > 1:
                stamped_action_chunk = [
                    StampedAction.from_action(
                        action=action,
                        init_step=step,
                        init_time=time.time(),
                        offset=i,
                        current_step=step,
                        control_freq=args.control_freq,
                    ) for i, action in enumerate(action_chunk)
                ]
                evolve_action_queue(action_queue, [action for action in stamped_action_chunk])
                action = action_queue.popleft().action.tolist()
            else:
                action = action_chunk[0]

        print(f"Executing action: {action}")

        obsv, reward, done, info = env.step(action)
        frames.append(capture_render_frame(obsv))

        if args.rt:
            over_time = real_time_pacer.sleep_until_next()
            if over_time > 0.0:
                print("Warning: control loop is over time!")

        step += 1
        step_time_list.append(step_timer(ms=True))

        if done:
            print(f"Episode success after {step + 1} timesteps")
            break

        print("-" * 50)

    if args.rt:
        executor.shutdown()

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

    if args.rt:
        print(f"Average overall latency: {np.mean(overall_latencies):.2f} ms")
        print(f"Average step time: {np.mean(step_time_list):.2f} ms, {1000/np.mean(step_time_list):.2f} Hz")