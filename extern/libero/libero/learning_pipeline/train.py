import os
import yaml

import torch
import numpy as np
from tqdm import trange
from easydict import EasyDict

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from libero.benchmark import get_benchmark
from libero import benchmark, get_libero_path
from libero_lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)
from libero_lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero_lifelong.metric import evaluate_loss, evaluate_success

from . import LifelongAlgo


def load_cfg(cfg_path: str):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=cfg_path)
    hydra_cfg = compose(config_name="config")
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    return cfg

def set_cfg(cfg: EasyDict):
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.eval.num_procs = 1
    cfg.eval.n_eval = 5
    cfg.train.n_epochs = 25
    return cfg

def _prepare_datasets(cfg: EasyDict, benchmark_name:str=None):
    if benchmark_name is None:
        benchmark_name = "libero_object"
    task_order = cfg.data.task_order_index # can be from {0 .. 21}, default to 0, which is [task 0, 1, 2 ...]
    cfg.benchmark_name = benchmark_name
    benchmark = get_benchmark(cfg.benchmark_name)(task_order)

    datasets = []
    descriptions = []
    shape_meta = None # dict of observation and action space shapes
    n_tasks = benchmark.n_tasks

    for i in range(n_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i==0),
                seq_len=cfg.data.seq_len,
        )
        # add language to the vision dataset, hence we call vl_dataset
        descriptions.append(benchmark.get_task(i).language)
        datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    datasets_vl = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
    return datasets_vl, benchmark, shape_meta

def train(cfg: dict):
    datasets_vl, benchmark, shape_meta = _prepare_datasets(cfg)
    assert len(datasets_vl) == benchmark.n_tasks
    n_tasks = benchmark.n_tasks

    cfg.policy.policy_type = "TransformerPolicy"
    cfg.lifelong.algo = "LifelongAlgo"

    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    print("Experiment directory is: ", cfg.experiment_dir)
    algo = safe_device(LifelongAlgo(n_tasks, cfg), cfg.device)

    result_summary = {
        'L_conf_mat': np.zeros((n_tasks, n_tasks)),   # loss confusion matrix
        'S_conf_mat': np.zeros((n_tasks, n_tasks)),   # success confusion matrix
        'L_fwd'     : np.zeros((n_tasks,)),           # loss AUC, how fast the agent learns
        'S_fwd'     : np.zeros((n_tasks,)),           # success AUC, how fast the agent succeeds
    }

    if (cfg.train.n_epochs < 50):
        print("NOTE: the number of epochs used in this example is intentionally reduced to 30 for simplicity.")
    if (cfg.eval.n_eval < 20):
        print("NOTE: the number of evaluation episodes used in this example is intentionally reduced to 5 for simplicity.")

    for i in trange(n_tasks):
        algo.train()
        s_fwd, l_fwd = algo.learn_one_task(datasets_vl[i], i, benchmark, result_summary)
        # s_fwd is success rate AUC, when the agent learns the {0, e, 2e, ...} epochs
        # l_fwd is BC loss AUC, similar to s_fwd
        result_summary["S_fwd"][i] = s_fwd
        result_summary["L_fwd"][i] = l_fwd

        if cfg.eval.eval:
            algo.eval()
            # we only evaluate on the past tasks: 0 .. i
            L = evaluate_loss(cfg, algo, benchmark, datasets_vl[:i+1]) # (i+1,)
            S = evaluate_success(cfg, algo, benchmark, list(range((i+1)*cfg.data.task_group_size))) # (i+1,)
            result_summary["L_conf_mat"][i][:i+1] = L
            result_summary["S_conf_mat"][i][:i+1] = S

            torch.save(result_summary, os.path.join(cfg.experiment_dir, f'result.pt'))


if __name__ == "__main__":
    cfg = load_cfg("../libero/configs")
    cfg = set_cfg(cfg)
    train(cfg)