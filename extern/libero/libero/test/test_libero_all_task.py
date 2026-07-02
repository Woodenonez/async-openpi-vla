import os

from libero import get_libero_path
from .test_libero_random import test_one_task


def test_all_tasks(test_one=False, auto=False) -> None:
    bddl_root = get_libero_path("bddl_files")
    if test_one:
        test_one_task(bddl_root=bddl_root)
        return

    task_group_list = os.listdir(bddl_root)
    print(f"Found \033[33m{len(task_group_list)}\033[0m task groups: {task_group_list}")

    for task_group in task_group_list:
        task_group_path = os.path.join(bddl_root, task_group)
        task_file_list = os.listdir(task_group_path)
        task_file_list = [f for f in task_file_list if f.endswith(".bddl")]

        print(f"--Found \033[33m{len(task_file_list)}\033[0m tasks in group '{task_group}'.")

        for i, task_file in enumerate(task_file_list):
            task_file_path = os.path.join(task_group_path, task_file)
            print(f"\n\033[34mTesting task file: {task_file_path}\033[0m")
            test_one_task(bddl_root=bddl_root, task_type=task_group, task_file=task_file, n_steps=50)
            if auto:
                print(f"{i+1}/{len(task_file_list)}")
            else:
                input(f"{i+1}/{len(task_file_list)} Press Enter to continue to the next task ...")

    print("\n\033[32mAll tasks tested successfully!\033[0m")


if __name__ == "__main__":
    test_all_tasks()