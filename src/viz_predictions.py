"""
This script trains a simple 2-layer MLP on the source dynamics dataset
"""
import argparse
from pathlib import Path

import numpy as np
import rerun as rr
import torch

import rrr
import wandb
from dataset import DynamicsDataset
from model import DynamicsNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')

    args = parser.parse_args()

    # Load the dataset
    dataset = DynamicsDataset(args.dataset_name)

    # Load the network
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt', method='FOCUS')

    # visualize predictions
    rrr.init()

    # run the network on each example in the dataset
    for example_idx, (inputs, targets, uuids) in enumerate(dataset):
        rr.set_time_sequence('examples', example_idx)

        outputs = model(inputs[None])
        prediction = outputs[0]

        # compute error
        log_dict = model.compute_errors(outputs, targets[None])
        loss = log_dict['loss']
        rr.log_scalar('loss', loss.detach().cpu().numpy())

        # grab the initial state from inputs
        robot_pos0 = inputs[0:3]
        object_pos0 = inputs[3:6]
        pred_traj = prediction.reshape(-1, 3)  # [robot x,y,z, object x,y,z]
        gt_traj = targets.reshape(-1, 3)
        pred_obj_positions = pred_traj[:, 0:3]

        gt_obj_positions = gt_traj[:, 0:3]

        pred_obj_positions = torch.cat([object_pos0[None], pred_obj_positions], 0)

        gt_obj_positions = torch.cat([object_pos0[None], gt_obj_positions], 0)

        gt_obj_positions = gt_obj_positions.detach().numpy()
        pred_obj_positions = pred_obj_positions.detach().numpy()

        rr.log_line_strip('object/gt', gt_obj_positions, color=(0, 0, 0.2), stroke_width=0.01)

        rr.log_line_strip('object/pred', pred_obj_positions + np.array([0, 0, 0.001]), color=(0, 0, 1.), stroke_width=0.01)


if __name__ == '__main__':
    main()
