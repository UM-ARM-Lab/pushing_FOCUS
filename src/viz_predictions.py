"""
This script trains a simple 2-layer MLP on the source dynamics dataset
"""
import argparse
from pathlib import Path

import rerun as rr
import torch
import wandb

import rrr
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
    for example_idx, (context, actions, target_obj_positions, target_robot_positions, uuids) in enumerate(dataset):
        rr.set_time_sequence('examples', example_idx)

        outputs = model(context[None], actions[None])
        prediction = outputs[0]

        # compute error
        log_dict = model.compute_errors(outputs, target_obj_positions[None])
        loss = log_dict['loss']
        rr.log_scalar('loss', loss.detach().cpu().numpy())

        # grab the initial state from inputs
        pred_obj_positions = prediction.reshape(-1, 3)  # [object x,y,z]
        target_obj_positions = target_obj_positions.reshape(-1, 3)
        target_robot_positions = target_robot_positions.reshape(-1, 3)

        context_traj = context.reshape(-1, 9)
        context_obj_positions = context_traj[:, 3:6]
        context_robot_positions = context_traj[:, 0:3]

        gt_robot_positions = torch.cat([context_robot_positions, target_robot_positions], 0)

        context_obj_positions = torch.cat([context_obj_positions, target_obj_positions[0:1]], 0)
        context_obj_positions = context_obj_positions.detach().numpy()
        target_obj_positions = target_obj_positions.detach().numpy()
        pred_obj_positions = pred_obj_positions.detach().numpy()

        rr.log_line_strip('object/context', context_obj_positions, color=(125, 125, 125), stroke_width=0.005)
        rr.log_points('object/context', context_obj_positions, colors=(125, 125, 125), radii=0.005)
        rr.log_line_strip('object/gt', target_obj_positions, color=(0, 0, 0.2), stroke_width=0.01)
        rr.log_line_strip('object/pred', pred_obj_positions, color=(0, 0, 1.), stroke_width=0.01)

        rr.log_line_strip('robot/gt', gt_robot_positions, color=(0.2, 0, 0), stroke_width=0.01)


if __name__ == '__main__':
    main()
