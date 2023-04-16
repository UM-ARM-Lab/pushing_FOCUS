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

        self.noise_sigma = np.ones(self.nu) * noise_sigma
        self.noise_rng = np.random.RandomState(seed)

        self.U = None
        self.reset()

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.rollout_results = None
        self.actions = None

    def roll(self):
        """ shift command 1 time step. Used before sampling a new command. """
        self.U = np.roll(self.U, -1, axis=0)
        # copy the last action and add noise to it
        self.U[-1] = self.noise_rng.randn(self.nu) * self.noise_sigma

    def command(self, data, get_result_func, cost_func):
        """
        Use this for no warmstarting.

        cost func needs to take in the output of get_result_func and return a cost for each sample.
        get_result_func needs to take in the model and data and return a result for each sample, which
        can be any object or tuple of objects.
        """
        self.roll()

        return self._command(data, get_result_func, cost_func)

    def _command(self, data, get_result_func, cost_func):
        """
        Use this for warmstarting.

        cost func needs to take in the output of get_result_func and return a cost for each sample.
        get_result_func needs to take in the model and data and return a result for each sample, which
        can be any object or tuple of objects.
        """
        noise = self.noise_rng.randn(self.num_samples, self.horizon, self.nu) * self.noise_sigma
        perturbed_action = self.U + noise

        results = parallel_rollout(self.pool, self.model, data, perturbed_action, get_result_func)

        self.rollout_results = results
        self.actions = perturbed_action
        costs = cost_func(results, perturbed_action)

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

        self.U = normalized(self.U) * min(np.abs(self.U).max(), 0.1)

        return action

    def reset(self):
        """
        Resets the control samples.
        """
        self.U = np.zeros([self.horizon, self.nu])
