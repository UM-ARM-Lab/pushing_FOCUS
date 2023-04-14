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

    n_steps = 100

    # FOCUS
    method = 'FOCUS'
    model = get_model(method)
    model, train_errors, similar_errors, dissimilar_errors = generate_data(model, n_steps)
    # only train data
    animate(f'{method}_train.mp4',
            model,
            train_errors,
            similar_errors=None,
            dissimilar_errors=None,
            n_steps=n_steps,
            show_weighting_function=True)
    # similar and dissimilar
    animate(f'{method}_test.mp4',
            model,
            train_errors=None,
            similar_errors=similar_errors,
            dissimilar_errors=dissimilar_errors,
            n_steps=n_steps,
            show_weighting_function=False)

    # All Data
    method = 'all_data'
    model = get_model(method)
    model, train_errors, similar_errors, dissimilar_errors = generate_data(model, n_steps)
    # only train data
    animate(f'{method}_train.mp4',
            model,
            train_errors,
            similar_errors=None,
            dissimilar_errors=None,
            n_steps=n_steps,
            show_weighting_function=True)
    # similar and dissimilar
    animate(f'{method}_test.mp4',
            model,
            train_errors=None,
            similar_errors=similar_errors,
            dissimilar_errors=dissimilar_errors,
            n_steps=n_steps,
            show_weighting_function=False)

    # CL
    method = 'CL'
    model = get_model(method)
    print(f'training {method}...')
    model, train_errors, similar_errors, dissimilar_errors = generate_data(model, n_steps)
    # only train data
    animate(f'{method}_train.mp4',
            model,
            train_errors,
            similar_errors=None,
            dissimilar_errors=None,
            n_steps=n_steps,
            show_weighting_function=True)
    # similar and dissimilar
    animate(f'{method}_test.mp4',
            model,
            train_errors=None,
            similar_errors=similar_errors,
            dissimilar_errors=dissimilar_errors,
            n_steps=n_steps,
            show_weighting_function=False)


def animate(filename, model, train_errors, similar_errors, dissimilar_errors, n_steps, show_weighting_function=True):
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    ax2 = ax.twinx()
    min_error = 2e-5
    max_error = 0.1
    error_bins = np.geomspace(min_error, max_error, 50)
    more_error_bins = np.geomspace(min_error, max_error, 500)
    ax.axvline(model.gamma, label=r'$\gamma$', color='k', linestyle='--', linewidth=2)
    if train_errors:
        _, _, train_bars = ax.hist([], error_bins, color='b', label='all target data')
    if similar_errors:
        _, _, similar_bars = ax.hist([], error_bins, color='g', label='similar')
    if dissimilar_errors:
        _, _, dissimilar_bars = ax.hist([], error_bins, color='r', label='dissimilar')
    text = ax.text(2e-2, 18, "")
    ax.set_xlim(min_error, max_error)
    ax.set_xscale('log')
    ax.set_ylim(0, 25)
    ax.set_xlabel("Prediction Error (log)")
    ax.set_ylabel("Count")
    if model.method == 'FOCUS' and show_weighting_function:
        wline = ax2.plot(more_error_bins, np.zeros_like(more_error_bins), c='m', label='weight', linewidth=2)[0]
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Weight")
    ax.legend(loc=2)

    if show_weighting_function:
        ax2.legend(loc=1)

    def func(global_step):
        text.set_text(f"Step {global_step}")
        if train_errors:
            train_counts, _ = np.histogram(train_errors[global_step], error_bins)
            for count, rect in zip(train_counts, train_bars.patches):
                rect.set_height(count)
        if similar_errors:
            similar_counts, _ = np.histogram(similar_errors[global_step], error_bins)
            for count, rect in zip(similar_counts, similar_bars.patches):
                rect.set_height(count)
        if dissimilar_errors:
            dissimilar_counts, _ = np.histogram(dissimilar_errors[global_step], error_bins)
            for count, rect in zip(dissimilar_counts, dissimilar_bars.patches):
                rect.set_height(count)
        if show_weighting_function:
            if model.method == 'FOCUS':
                weights = 1 - sigmoid(model.global_k * global_step * (more_error_bins - model.gamma))
                wline.set_ydata(weights)

    anim = FuncAnimation(fig, func, frames=n_steps, interval=30)

    root = Path("animations")
    root.mkdir(exist_ok=True)
    anim.save((root / filename).as_posix(), writer='ffmpeg', dpi=200)
    print(f'Saved {filename}')


def get_model(method):
    # Load from source model
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    return DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt',
                                                lr=1e-5,
                                                method=method,
                                                gamma=3e-3,
                                                global_k=100)


def generate_data(model, n_steps):
    dataset = DynamicsDataset('target-dataset')
    similar_dataset = DynamicsDataset('similar-dataset')
    dissimilar_dataset = DynamicsDataset('dissimilar-dataset')
    train_loader = DataLoader(dataset, batch_size=999)
    similar_loader = DataLoader(similar_dataset, batch_size=999)
    dissimilar_loader = DataLoader(dissimilar_dataset, batch_size=999)
    similar_errors = []
    dissimilar_errors = []
    train_errors = []
    similar_weights = []
    dissimilar_weights = []
    train_weights = []
    opt = model.configure_optimizers()
    for global_step in range(n_steps):
        for batch in train_loader:
            inputs, targets, train_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            train_errors.append(log_dict['error'])
            if model.method == 'FOCUS':
                train_weights.append(log_dict['weights'])
        for batch in similar_loader:
            inputs, targets, similar_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            similar_errors.append(log_dict['error'])
            if model.method == 'FOCUS':
                similar_weights.append(log_dict['weights'])
        for batch in dissimilar_loader:
            inputs, targets, dissimilar_uuids = batch
            outputs = model.forward(inputs)
            log_dict = model.compute_errors(outputs, targets, global_step)
            dissimilar_errors.append(log_dict['error'])
            if model.method == 'FOCUS':
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
    return model, train_errors, similar_errors, dissimilar_errors


if __name__ == '__main__':
    main()
