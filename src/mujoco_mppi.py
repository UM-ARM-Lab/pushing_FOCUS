import numpy as np

from rollout import parallel_rollout


def softmax(x, temp):
    x = x * temp
    return np.exp(x) / np.exp(x).sum()


def normalized(x):
    return (x - x.min()) / (x.max() - x.min())


class MujocoMPPI:

    def __init__(self, pool, model, num_samples, horizon, noise_sigma, lambda_=1., gamma=0.9, seed=0):
        self.pool = pool
        self.model = model
        self.num_samples = num_samples
        self.horizon = horizon

        self.gammas = np.power(gamma, np.arange(horizon))[None]

        # dimensions of state and control
        self.nx = model.nq
        self.nu = model.nu
        self.lambda_ = lambda_

        if isinstance(noise_sigma, (float, int)):
            self.noise_sigma = np.ones(self.nu) * noise_sigma
        else:
            self.noise_sigma = noise_sigma
        self.noise_rng = np.random.RandomState(seed)

        self.U = None
        self.cost = None
        self.cost_normalized = None
        self.reset()

    def roll(self):
        """ shift command 1 time step. Used before sampling a new command. """
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = self.noise_rng.randn(self.nu) * self.noise_sigma

    def command(self, dynamics_and_cost_func, **kwargs):
        """
        Recompute the best command. One iteration of the MPPI algorithm.
        You probably want to run this a few times before using the output.

        Args:
            dynamics_and_cost_func: function that takes in the perturbed action and returns the cost
            kwargs: keyword args that will be forward to dynamics_and_cost_func

        NOTE: this function could just have no args and the current args could be moved to the constructor?
        """
        noise = self.noise_rng.randn(self.num_samples, self.horizon, self.nu) * self.noise_sigma
        perturbed_action = self.U + noise
        perturbed_action = self.bound_action(perturbed_action)

        costs = dynamics_and_cost_func(perturbed_action, **kwargs)

        self.cost = np.sum(self.gammas * costs, axis=-1)

        if np.all(self.cost == self.cost[0]):
            raise ValueError('all costs are the same!!!')

        self.cost_normalized = normalized(self.cost)

        weights = softmax(-self.cost_normalized, self.lambda_)
        print(f'weights: std={float(np.std(weights)):.2f} max={float(np.max(weights)):.2f}')

        # compute the (weighted) average noise and add that to the reference control
        weighted_avg_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.U += weighted_avg_noise

        action = self.U[0]

        self.U = self.bound_action(self.U)

        return action

    def reset(self):
        """
        Resets the control samples.
        """
        self.U = np.zeros([self.horizon, self.nu])

    def bound_action(self, perturbed_action):
        xy = 0.08
        rot = 1
        return np.clip(perturbed_action, [-xy, -xy, -rot], [xy, xy, rot])
