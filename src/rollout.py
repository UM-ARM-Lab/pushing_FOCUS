import copy
import logging
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

N_SUB_STEPS = 125

logger = logging.getLogger(__name__)


def rollout(model, data, controls, get_result_func=None):
    # run for the initial data, so that the current state is returned in the output
    results_lists = None
    if get_result_func is not None:
        result_tuple = get_result_tuple(data, get_result_func, model)

        if results_lists is None:
            results_lists = tuple([] for _ in result_tuple)

        for result_list, result in zip(results_lists, result_tuple):
            result_list.append(result)

    for t in range(controls.shape[0]):
        qvel_target = controls[t]

        control_step(model, data, qvel_target)

        if get_result_func is not None:
            result_tuple = get_result_tuple(data, get_result_func, model)

            if results_lists is None:
                results_lists = tuple([] for _ in result_tuple)

            for result_list, result in zip(results_lists, result_tuple):
                result_list.append(result)

    if results_lists is None:
        return None

    if len(results_lists) == 1:
        return results_lists[0]
    return results_lists


def get_result_tuple(data, get_result_func, model):
    result_tuple = get_result_func(model, data)
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)

    # make sure we copy otherwise the data gets overwritten
    result_tuple = tuple(np.copy(result) for result in result_tuple)

    return result_tuple


def control_step(model, data, qvel_target):
    if qvel_target is not None:
        np.copyto(data.ctrl, qvel_target)
    else:
        logger.warning("control is None!!!")
    mujoco.mj_step(model, data, nstep=N_SUB_STEPS)


def parallel_rollout(pool: ThreadPoolExecutor, model, data, controls_samples, get_result_func=None):
    args_sets = [(model, copy.copy(data), controls, get_result_func) for controls in controls_samples]
    futures = [pool.submit(rollout, *args) for args in args_sets]

    results = [f.result() for f in futures]

    result0 = results[0]
    if isinstance(result0, tuple):
        results = tuple(np.array(result_i) for result_i in zip(*results))
    else:
        results = np.array(results)

    return results
