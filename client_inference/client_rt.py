from typing import Deque
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from openpi_client.websocket_client_policy import WebsocketClientPolicy # type: ignore
from libero.utils.utils import get_libero_dummy_action # type: ignore

from prepare_client import Args, StampedAction, PendingResult, PieceTimer, RealTimePacer
from prepare_client import get_task_env, make_policy_obs, infer, evolve_action_queue


overall_latencies: list[float] = []
ol_timer = PieceTimer()

step_time_list: list[float] = []
step_timer = PieceTimer()


def main(args: Args):
    real_time_pacer = RealTimePacer(control_freq=args.control_freq)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    action_queue: Deque[StampedAction] = deque(maxlen=args.max_queue_step)

    next_infer_time = 0.0
    executor = ThreadPoolExecutor(max_workers=1)
    pending_result = PendingResult(future=None)

    env, task, task_instruction, initial_states = get_task_env(args.task_suite_name, args.task_id, control_freq=args.control_freq)
    env.reset()
    env.render()
    try:
        obsv = env.set_init_state(initial_states[args.init_state_id])
    except IndexError as e:
        raise ValueError(f"Invalid init_state_id {args.init_state_id} for task_id {args.task_id} (max={len(initial_states)})") from e
    for _ in range(10):
        env.step(get_libero_dummy_action()) # settle environment
    try:
        observation = make_policy_obs(obsv, task_instruction)
        infer(client, observation)
    except Exception as e:
        raise ValueError(f"Error during initial inference: {e}")

    last_action = None
    real_time_pacer.reset()
    print("Starting main loop...")
    print("=" * 50)
    for step in range(args.num_steps):
        print(f"Step: {step}")
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
        else:
            if last_action is not None:
                action = get_libero_dummy_action()
                action[-1] = last_action[-1] # keep gripper state
            else:
                action = get_libero_dummy_action()
        print(f"Executing action: {action}")

        obsv, reward, done, info = env.step(action)
        env.render()
        over_time = real_time_pacer.sleep_until_next()
        if over_time > 0.0:
            print("Warning: control loop is over time!")

        step += 1
        step_time_list.append(step_timer(ms=True))

        if done:
            print(f"Episode success after {step + 1} timesteps")
            break

        print("-" * 50)

    executor.shutdown()
    plt.show()


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main(args)

    print(f"Average overall latency: {np.mean(overall_latencies):.2f} ms")
    print(f"Average step time: {np.mean(step_time_list):.2f} ms, {1000/np.mean(step_time_list):.2f} Hz")