
from dataclasses import dataclass
import statistics

import numpy as np


@dataclass
class TrialMetrics:
    # mean, max, std
    infer_latency: list[float]
    extra_latency: list[float]
    diff_action_pos: list[float]
    diff_eef_pos: list[float]
    diff_joint: list[float]
    finish_time_step: int
    success: bool


class Metrics:
    def __init__(self):
        self.trial_list: list[TrialMetrics] = [] # type: ignore
        self.success_rate = 0.0

    def get_average(self, round_digit:int=4) -> dict:
        self.metric_average = {}
        all_infer_latencies = []
        all_extra_latencies = []
        all_diff_action_pos = []
        all_diff_eef_pos = []
        all_diff_joint = []
        all_finish_time = []
        for trial in self.trial_list:
            all_infer_latencies.append(trial.infer_latency)
            all_extra_latencies.append(trial.extra_latency)
            all_diff_action_pos.append(trial.diff_action_pos)
            all_diff_eef_pos.append(trial.diff_eef_pos)
            all_diff_joint.append(trial.diff_joint)
            if trial.success:
                all_finish_time.append(trial.finish_time_step)
        if len(all_infer_latencies) > 20:
            all_infer_latencies = all_infer_latencies[10:] # remove the first 10 trials for more stable results
            all_extra_latencies = all_extra_latencies[10:]
            all_diff_action_pos = all_diff_action_pos[10:]
            all_diff_eef_pos = all_diff_eef_pos[10:]
            all_diff_joint = all_diff_joint[10:]
        if not all_finish_time:
            all_finish_time = [-1]
        self.metric_average["infer_latency"] = [round(statistics.mean([x[0] for x in all_infer_latencies]), round_digit),
                                                round(statistics.mean([x[1] for x in all_infer_latencies]), round_digit),
                                                round(statistics.mean([x[2] for x in all_infer_latencies]), round_digit)]
        self.metric_average["extra_latency"] = [round(statistics.mean([x[0] for x in all_extra_latencies]), round_digit),
                                                round(statistics.mean([x[1] for x in all_extra_latencies]), round_digit),
                                                round(statistics.mean([x[2] for x in all_extra_latencies]), round_digit)]
        self.metric_average["diff_action_pos"] = [round(statistics.mean([x[0] for x in all_diff_action_pos]), round_digit),
                                                  round(statistics.mean([x[1] for x in all_diff_action_pos]), round_digit),
                                                  round(statistics.mean([x[2] for x in all_diff_action_pos]), round_digit)]
        self.metric_average["diff_eef_pos"] = [round(statistics.mean([x[0] for x in all_diff_eef_pos]), round_digit),
                                               round(statistics.mean([x[1] for x in all_diff_eef_pos]), round_digit),
                                               round(statistics.mean([x[2] for x in all_diff_eef_pos]), round_digit)]
        self.metric_average["diff_joint"] = [round(statistics.mean([x[0] for x in all_diff_joint]), round_digit),
                                             round(statistics.mean([x[1] for x in all_diff_joint]), round_digit),
                                             round(statistics.mean([x[2] for x in all_diff_joint]), round_digit)]
        self.metric_average["finish_time"] = round(statistics.mean(all_finish_time), round_digit) # type: ignore
        self.metric_average["success_rate"] = self.success_rate
        return self.metric_average

    def add_trial_result(
            self, 
            infer_latency_list: list[float],
            extra_latency_list: list[float],
            action_list: list[np.ndarray],
            eef_traj: list[np.ndarray],
            joint_traj: list[np.ndarray],
            succeed: bool,
        ):
        trial_res = TrialMetrics(
            infer_latency=self._get_computation_time(infer_latency_list),
            extra_latency=self._get_computation_time(extra_latency_list),
            diff_action_pos=self._get_diff(action_list, lambda x: x[:3]),
            diff_eef_pos=self._get_diff(eef_traj, lambda x: x[:3]),
            diff_joint=self._get_diff(joint_traj, lambda x: x),
            finish_time_step=len(action_list) if succeed else -1,
            success=succeed,
        )
        self.trial_list.append(trial_res)
        self._get_success_rate()

    def _get_computation_time(self, computation_time_list):
        return [statistics.mean(computation_time_list), max(computation_time_list), statistics.stdev(computation_time_list)]
    
    def _get_diff(self, traj_list: list[np.ndarray], extract_fn) -> list[float]:
        diffs = []
        for i in range(1, len(traj_list)):
            prev = extract_fn(traj_list[i-1])
            curr = extract_fn(traj_list[i])
            diff = float(np.linalg.norm(curr - prev))
            diffs.append(diff)
        return [statistics.mean(diffs), max(diffs), statistics.stdev(diffs)]
    
    def _get_success_rate(self):
        self.success_rate = sum([trial.success for trial in self.trial_list])/len(self.trial_list)