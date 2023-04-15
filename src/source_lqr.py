from pathlib import Path

import numpy as np
import torch
import wandb

import rrr
from env import Env
from model import DynamicsNetwork


def state_to_vec(state):
    return np.array([state['object_pos'][0],
                     state['object_pos'][1],
                     state['object_pos'][2],
                     state['robot_pos'][0],
                     state['robot_pos'][1],
                     state['robot_pos'][2]])


def lqr_gain(A, B, Q, R):
    # solve the Riccati equation
    P = np.zeros_like(Q)
    for _ in range(100):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    return np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

def linearize(model, state):
    # compute the linearization around the current state
    x = state_to_vec(state)
    u = np.zeros(2)

    # inputs should be a tensor of shape (1, 8)
    # where the 8 is [rx, ry, rz, ox, oy, oz, ux, uy]
    # r means robot, o means object, u means action
    x = torch.tensor(x).float()
    u = torch.tensor(u).float()
    inputs = torch.concatenate([x, u]).unsqueeze(0)
    outputs = model.forward(inputs)
    # get the gradients of the outputs with respect to the inputs
    outputs.backward(torch.eye(8))

    # the gradients are stored in the .grad attribute of the inputs
    dx = x.grad.detach().numpy() # [3, 6]
    du = u.grad.detach().numpy() # [3, 2]

    return A, B
def lqr(model, state):
    # linearize the model around the current state
    A, B = linearize(model, state)

    # compute the LQR gain
    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1, 1])
    K = lqr_gain(A, B, Q, R)

    # compute the LQR action
    x = state_to_vec(state)
    u = -K @ x

    return u


def main():
    # get the source dyamics model
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')

    # initialize rerun
    rrr.init()

    # initialize the target simulation environment
    env = Env('target.xml')

    while True:
        state = env.get_state()

        action = lqr(model, state)

        env.step(action)


if __name__ == '__main__':
    main()
