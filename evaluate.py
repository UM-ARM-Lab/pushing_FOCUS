import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

import wandb


def random_color_like(color, color_rng, blend = 0.8):
    color = np.array(color)
    perturbation = color_rng.uniform(0, 1, 3)
    return blend * color + (1 - blend) * perturbation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')

    args = parser.parse_args()

    # get the wandb run
    api = wandb.Api()
    run = api.run(f"armlab/pushing_focus/{args.run_id}")

    color_rng = np.random.default_rng(0)

    # animate the weight & error for each example
    # over the course of training
    color_map = {}
    for log_i in run.history(pandas=False):
        if 'similar-dataset_uuids' in log_i:
            for uuid in log_i['similar-dataset_uuids']:
                color_map[uuid] = color_rng.uniform(0, 1, 3)
        if 'dissimilar-dataset_uuids' in log_i:
            for uuid in log_i['dissimilar-dataset_uuids']:
                color_map[uuid] = color_rng.uniform(0, 1, 3)

    errors = []
    weights = []
    colors = []
    # Collect the errors and weights for both similar and dissimilar datasets
    # with matching global_step
    rows = []
    colors_by_uuid = {}
    for log_i in run.history(pandas=False):
        global_step = log_i['global_step']
        if 'dissimilar-dataset_weights' in log_i:
            for e, w, uuid in zip(log_i['dissimilar-dataset_error'], log_i['dissimilar-dataset_weights'], log_i['dissimilar-dataset_uuids']):
                if uuid not in colors_by_uuid:
                    colors_by_uuid[uuid] = random_color_like([1, 0, 0], color_rng)
                color = colors_by_uuid[uuid]
                rows.append([global_step, e, w, uuid, color])
        if 'similar-dataset_weights' in log_i:
            for e, w, uuid in zip(log_i['similar-dataset_error'], log_i['similar-dataset_weights'], log_i['similar-dataset_uuids']):
                if uuid not in colors_by_uuid:
                    colors_by_uuid[uuid] = random_color_like([0, 1, 0], color_rng)
                color = colors_by_uuid[uuid]
                rows.append([global_step, e, w, uuid, color])

    df = pd.DataFrame(rows, columns=['global_step', 'error', 'weight', 'uuid', 'color'])

    fig, ax = plt.subplots()
    global_steps = df['global_step'].unique()
    data0 = df[df['global_step'] == 0]
    scatt = ax.scatter(np.zeros(len(data0)), np.zeros(len(data0)), s=50, alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(1e-6, 1e-4)
    ax.set_xlabel('Error')
    ax.set_ylabel('Weight')
    vline = ax.axvline(0, color='k', linestyle='--')

    def func(i):
        global_step = global_steps[i]
        data = df[df['global_step'] == global_step]
        scatt.set_offsets(data[['error', 'weight']].to_numpy())
        scatt.set_color(data['color'].to_numpy())
        # set the mean line to be the mean of the errors
        vline.set_xdata([data['error'].quantile(0.25)])

    anim = FuncAnimation(fig, func, frames=len(global_steps), interval=500)
    plt.show()


if __name__ == '__main__':
    main()
