import numpy as np
import gym


class LinearScalarization(gym.Wrapper):

    def __init__(self, env, weights):
        super(LinearScalarization, self).__init__(env)
        self.weights = weights
        self.mo_return = None

    def reset(self):
        o = super(LinearScalarization, self).reset()
        self.mo_return = 0.
        return o

    def step(self, a):
        o, r, d, i = super(LinearScalarization, self).step(a)
        self.mo_return += r
        scalarized = np.dot(r, self.weights)
        return o, scalarized, d, i