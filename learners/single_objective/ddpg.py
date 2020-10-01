import numpy as np
import torch
import copy
from itertools import chain
from learners.single_objective.agent import NNAgent, Transition
from wrappers.tensor import TensorWrapper

import torch.nn as nn
import torch.nn.functional as F


class DDPG(NNAgent):

    def __init__(self, env,
                 policy=None,
                 memory=None,
                 actor=None,
                 critic=None,
                 batch_size=1,
                 learn_start=0,
                 n_steps_update=30,
                 k=50,
                 gamma=1.,
                 lr=1e-3,
                 copy_every=1,
                 tau=1.,
                 **nn_kwargs):
        params = chain(actor.parameters(), critic.parameters())
        # if actor and critic share layers, only insert them once in the optimizer
        # do not use a set as it is unordered
        params = list(dict.fromkeys(params))
        optimizer = torch.optim.Adam(params, lr=lr)
        super(DDPG, self).__init__(optimizer=optimizer, **nn_kwargs)
        # make only 1 deepcopy with both nn to ensure shared layers stay shared
        target_actor, target_critic = copy.deepcopy([actor, critic])
        # tensor wrapper to make batches of steps, and convert np.arrays to tensors
        env = TensorWrapper(env)
        self.env = env
        self.policy = policy
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.memory = memory

        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.n_steps_update = n_steps_update
        self.k = k
        self.tau = tau
        self.copy_every = copy_every

    def start(self, log=None):
        obs = self.env.reset()
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            action = self.policy(actor_out, log=log)
        next_obs, reward, terminal, _ = self.env.step(action)

        # add in replay memory
        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal)
        self.memory.add(t)

        if log.total_steps >= self.learn_start and log.total_steps % self.n_steps_update == 0:
            for k_i in range(self.k):
                batch = self.memory.sample(self.batch_size)

                # compute targets
                with torch.no_grad():
                    a_ns = self.target_actor(batch.next_observation)
                    # q-value depends on state and action
                    q_ns = self.target_critic(batch.next_observation, a_ns)
                    # immediate reward if final step, else reward + discounted Q-value
                    q_target = batch.reward + self.gamma*q_ns*torch.logical_not(batch.terminal)

                # update critic
                q_s = self.critic(batch.observation, batch.action)
                critic_loss = F.mse_loss(q_s, q_target)

                # update actor
                a_ns = self.actor(batch.observation)
                # deactivate gradients for critic
                # TODO what happens with shared layers between actor and critic ?
                for p in self.critic.parameters():
                    p.requires_grad = False
                # backwards will not take into account critic network, but flow back through actor network
                actor_loss = -self.critic(batch.observation, a_ns).mean()
                # reactivate gradients
                for p in self.critic.parameters():
                    p.requires_grad = True

                loss = actor_loss + critic_loss
                self.optimizer_step(loss)

                # update target networks, based on the total number of updates that happened
                if (self.k * log.total_steps + k_i) % self.copy_every == 0:
                    with torch.no_grad():
                        self.copy_weights(self.actor, self.target_actor)
                        self.copy_weights(self.critic, self.target_critic)
                
        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal}

    def copy_weights(self, model, target_model):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

    def evalstep(self, previous, log=None):
        with torch.no_grad():
            actor_out = self.actor(previous['observation'])
            # action = self.policy(actor_out, log=log)
            action = actor_out + 0.1*torch.normal(0, 1, actor_out.shape)
            action = action.clamp(-1., 1.)
        next_obs, reward, terminal, info = self.env.step(action)

        return {'observation': next_obs,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'env_info': info}

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'policy': self.policy,}
                # 'memory': self.memory}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd['actor'])
        self.critic.load_state_dict(sd['critic'])
        self.policy = sd['policy']
        # self.memory = sd['memory']