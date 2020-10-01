import torch
import numpy as np


class Policy(object):

    def __call__(self, args, log=None):
        raise NotImplementedError()

    def log_prob(self, samples, args):
        raise NotImplementedError()


class OrnsteinUhlenbeckActionNoise(object):

    def __init__(self, n_actions, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.theta = theta
        self.mu = torch.zeros(n_actions)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(0, 1, self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros(*self.mu.shape)


class NormalNoise(object):

    def __init__(self, action_dim, sigma=0.2):
        super(NormalNoise, self).__init__()
        self.sigma = sigma
        self.action_dim = action_dim

    def __call__(self):
        return self.sigma*torch.normal(0, 1, (self.action_dim,))

    def reset(self):
        pass


class Normal(Policy):

    def __init__(self, action_high, action_dim, noise=None, **noise_kwargs):
        noises = {
            'ornstein_uhlenbeck': OrnsteinUhlenbeckActionNoise(action_dim, **noise_kwargs),
            'normal': NormalNoise(action_dim, **noise_kwargs)
        }
        if noise is not None:
            noise = noises[noise]
        self.noise = noise
        self.action_high = action_high

    @classmethod
    def _get_mu_std(cls, log_probs):
        # if tuple, deviation was provided, else, assume deterministic
        if type(log_probs) == tuple:
            mu, log_std = log_probs
            std = torch.exp(log_std)
        else:
            mu, std = log_probs, torch.zeros(log_probs.shape)
            log_std = None
        return mu, std, log_std

    def __call__(self, log_probs, log=None):
        if self.noise is not None and log.episode_step == 0:
            self.noise.reset()
        mu, std, _ = Normal._get_mu_std(log_probs)
        # TODO user rsample from torch.distributions.Normal
        actions = torch.normal(mu, std)
        if self.noise is not None:
            actions += self.noise()
        actions.clamp_(-self.action_high, self.action_high)
        return actions

    def log_prob(self, samples, log_probs):
        mu, std, log_std = Normal._get_mu_std(log_probs)
        var = std ** 2
        return -((samples - mu) ** 2) / (2 * var) - log_std - np.log(np.sqrt(2 * np.float32(np.pi)))