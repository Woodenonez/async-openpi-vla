from typing import Deque
from collections import deque

import matplotlib.pyplot as plt

from openpi_client import websocket_client_policy # type: ignore
from libero.utils.utils import get_libero_dummy_action # type: ignore

from prepare_client import Args, StampedAction
from prepare_client import get_task_env, make_policy_obs, infer


USE_QUEUE = True

def main(args: Args):
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    action_queue: Deque[StampedAction] = deque(maxlen=args.max_queue_step)
    
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
        env.render()

        step += 1
        if done:
            print(f"Episode success after {step + 1} timesteps")
            break

    plt.show()


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main(args)