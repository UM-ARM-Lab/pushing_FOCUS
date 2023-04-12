from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader

import wandb
from dataset import DynamicsDataset
from model import DynamicsNetwork


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    np.set_printoptions(precision=3, suppress=True)

    method = 'FOCUS'

    dataset = DynamicsDataset('target-dataset')
    similar_dataset = DynamicsDataset('similar-dataset')
    dissimilar_dataset = DynamicsDataset('dissimilar-dataset')

    train_loader = DataLoader(dataset, batch_size=999)

    similar_loader = DataLoader(similar_dataset, batch_size=999)
    dissimilar_loader = DataLoader(dissimilar_dataset, batch_size=999)

    # Load from source model
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt',
                                                 lr=1e-4,
                                                 method=method,
                                                 gamma=1e-3,
                                                 global_k=10000)

    n_steps = 100
    similar_errors = []
    dissimilar_errors = []
    train_errors = []
    similar_weights = []
    dissimilar_weights = []
    train_weights = []
    train_uuids = None
    similar_uuids = None
    dissimilar_uuids = None
    opt = model.configure_optimizers()
    for global_step in range(n_steps):
        for batch in train_loader:
            inputs, targets, train_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            train_errors.append(log_dict['error'])
            if method == 'FOCUS':
                train_weights.append(log_dict['weights'])
        for batch in similar_loader:
            inputs, targets, similar_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            similar_errors.append(log_dict['error'])
            if method == 'FOCUS':
                similar_weights.append(log_dict['weights'])
        for batch in dissimilar_loader:
            inputs, targets, dissimilar_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            dissimilar_errors.append(log_dict['error'])
            if method == 'FOCUS':
                dissimilar_weights.append(log_dict['weights'])

        # now run an update of gradient descent on the model
        # for each example in the training set (mixed data from target env)
        # NOTE: we're using full-batches not mini-batches
        for batch in train_loader:
            model.zero_grad()
            inputs, targets, _ = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            loss = log_dict["loss"]
            loss.backward()
            opt.step()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    min_error = 1e-5
    max_error = 0.1
    error_bins = np.geomspace(min_error, max_error, 50)
    more_error_bins = np.geomspace(min_error, max_error, 500)
    ax.axvline(model.gamma, label='gamma', color='k', linestyle='--')
    _, _, train_bars = ax.hist([], error_bins, color='b', label='all target data', alpha=1)
    _, _, similar_bars = ax.hist([], error_bins, color='g', label='similar', alpha=0.2)
    _, _, dissimilar_bars = ax.hist([], error_bins, color='r', label='dissimilar', alpha=0.2)
    text = ax.text(1e-2, 15, "")

    ax.set_xlim(min_error, max_error)
    ax.set_xscale('log')
    ax.set_ylim(0, 25)
    ax.set_xlabel("Prediction Error (log)")
    ax.set_ylabel("Count")

    if method == 'FOCUS':
        wline = ax2.plot(more_error_bins, np.zeros_like(more_error_bins), c='m', label='weight')[0]
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Weight")

    ax.legend(loc=1)
    ax2.legend(loc=2)

    def func(global_step):
        text.set_text(f"Step {global_step}")
        train_counts, _ = np.histogram(train_errors[global_step], error_bins)
        for count, rect in zip(train_counts, train_bars.patches):
            rect.set_height(count)
        similar_counts, _ = np.histogram(similar_errors[global_step], error_bins)
        for count, rect in zip(similar_counts, similar_bars.patches):
            rect.set_height(count)
        dissimilar_counts, _ = np.histogram(dissimilar_errors[global_step], error_bins)
        for count, rect in zip(dissimilar_counts, dissimilar_bars.patches):
            rect.set_height(count)
        if method == 'FOCUS':
            weights = 1 - sigmoid(model.global_k * global_step * (more_error_bins - model.gamma))
            wline.set_ydata(weights)

    anim = FuncAnimation(fig, func, frames=n_steps, interval=1)
    plt.show(block=True)
    anim.save('adaptation.mp4', writer='imagemagick', dpi=100)


if __name__ == '__main__':
    main()
