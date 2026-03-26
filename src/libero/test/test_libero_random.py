import os
import numpy as np

from libero.envs.env_wrapper import ControlEnv


def test_one_task(bddl_root, task_type="libero_goal", task_file="open_the_middle_drawer_of_the_cabinet.bddl", n_steps:int=100) -> None:
    task_fpath = os.path.join(bddl_root, task_type, task_file)

    env = ControlEnv(
        bddl_file_name=task_fpath,
        use_camera_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        control_freq=20,
        horizon=200,
    )

    env.reset()
    env.render()

    for _ in range(n_steps):
        low, high = env.env.action_spec
        random_action = np.random.uniform(low, high)

        observation, reward, done, info = env.step(random_action)

        env.render()

        if done:
            env.reset()
            env.render()

    env.close()


if __name__ == "__main__":
    from libero import get_libero_path # type: ignore
    test_one_task(bddl_root=get_libero_path("bddl_files"))
