from dataclasses import dataclass, astuple
import random
import torch
import numpy as np


class BatchTransition(object):

    def __init__(self, observation, action, reward, next_observation, terminal):
        # ensure that there is no tensors (Batch,), but always have a second dim
        if len(action.shape) == 1: action = action[:, None]
        if len(reward.shape) == 1: reward = reward[:, None]
        if len(terminal.shape) == 1: terminal = terminal[:, None]
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.terminal = terminal


class Memory(object):

    def __init__(self, size=100000):

        self.size = size
        self.memory = []
        self.current = 0

    def add(self, transition):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.current] = astuple(transition)
        self.current = (self.current + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # TODO if multisized batch, will mix different envs steps together
        # use torch.stack(i).transpose(1,0).reshape(-1, *i.shape[1:])
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)])
        return batch

    def last(self, batch_size):
        assert len(self.memory) >= batch_size, 'not enough samples in memory'
        s_i, e_i = self.current - batch_size, self.current
        # if s_i < 0, part of batch is at end of list, wrap around
        if s_i < 0:
            batch = self.memory[s_i:] + self.memory[:e_i]
        else:
            batch = self.memory[s_i:e_i]
        # TODO if multisized batch, will mix different envs steps together
        # use torch.stack(i).transpose(1,0).reshape(-1, *i.shape[1:])
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)])
        return batch