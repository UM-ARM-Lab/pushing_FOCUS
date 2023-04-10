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

    args = parser.parse_args()

    # Load the dataset
    dataset = DynamicsDataset('source-dataset')

    # Load the network
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')

    # visualize predictions
    rrr.init()

    # run the network on each example in the dataset
    for example_idx, (inputs, outputs) in enumerate(dataset):
        rr.set_time_sequence('examples', example_idx)

        predictions = model(inputs[None])[0]
        # grab the initial state from inputs
        robot_pos0 = inputs[0:3]
        object_pos0 = inputs[3:6]
        pred_traj = predictions.reshape(-1, 6)  # [robot x,y,z, object x,y,z]
        gt_traj = outputs.reshape(-1, 6)
        pred_robot_positions = pred_traj[:, 0:3]
        pred_obj_positions = pred_traj[:, 3:6]

        gt_robot_positions = gt_traj[:, 0:3]
        gt_obj_positions = gt_traj[:, 3:6]

        pred_robot_positions = torch.cat([robot_pos0[None], pred_robot_positions], 0)
        pred_obj_positions = torch.cat([object_pos0[None], pred_obj_positions], 0)

        gt_robot_positions = torch.cat([robot_pos0[None], gt_robot_positions], 0)
        gt_obj_positions = torch.cat([object_pos0[None], gt_obj_positions], 0)

        pred_robot_positions = pred_robot_positions.detach().numpy()
        pred_obj_positions = pred_obj_positions.detach().numpy()

        rr.log_line_strip('object/gt', gt_obj_positions, color=(0, 0, 0.2), stroke_width=0.01)
        rr.log_line_strip('robot/gt', gt_robot_positions, color=(0.2, 0, 0), stroke_width=0.01)

        rr.log_line_strip('object/pred', pred_obj_positions + np.array([0, 0, 0.001]), color=(0, 0, 1.), stroke_width=0.01)
        rr.log_line_strip('robot/pred', pred_robot_positions + np.array([0, 0, 0.001]), color=(1., 0, 0), stroke_width=0.01)


if __name__ == '__main__':
    main()
