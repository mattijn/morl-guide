import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from metrics import non_dominated, non_dominated_rank, crowding_distance, compute_hypervolume
from log.logger import Logger


def n_parameters(model):
    return np.sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()])


def set_parameters(model, z):
    assert len(z) == n_parameters(model), 'not providing correct amount of parameters'
    s = 0
    for p in model.parameters():
        n_p = torch.prod(torch.tensor(p.shape))
        p.data = z[s:s+n_p].view_as(p)
        s += n_p


def run_episode(env, model):

    e_r = 0; done = False
    o = env.reset()
    while not done:
        with torch.no_grad():
            action = model(torch.from_numpy(o).float()[:,None])
            action = action.detach().numpy().flatten()
        n_o, r, done, _ = env.step(action)
        e_r += r
        o = n_o

    return torch.from_numpy(e_r).float()


def indicator_hypervolume(points, ref, nd_penalty=0.):
    # compute hypervolume of dataset
    nd_i = non_dominated(points)
    nd = points[nd_i]
    hv = compute_hypervolume(nd, ref)
    # hypervolume without a point from dataset
    hv_p = np.zeros(len(points))
    # penalization if point is dominated
    is_nd = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        is_nd[i] = np.any(np.all(nd == points[i], axis=1))
        if is_nd[i]:
            # if there is only one non-dominated point, and this point is non-dominated,
            # then it amounts for the full hypervolume
            if len(nd) == 1:
                hv_p[i] = 0.
            else:
                # remove point from nondominated points, compute hv
                rows = np.all(nd == points[i], axis=1)
                hv_p[i] = compute_hypervolume(nd[np.logical_not(rows)], ref)
        # if point is dominated, no impact on hypervolume
        else:
            hv_p[i] = hv

    indicator = hv - hv_p - nd_penalty*np.logical_not(is_nd)
    return indicator


def indicator_non_dominated(points):
    ranks = non_dominated_rank(points)
    indicator = -ranks + crowding_distance(points, ranks=ranks)
    return indicator


class MONES(object):

    def __init__(self,
                 make_env,
                 policy,
                 n_population=1,
                 n_runs=1,
                 indicator='non_dominated',
                 ref_point=None,
                 logdir='runs'):
        self.make_env = make_env
        self.policy = policy

        self.n_population = n_population
        self.n_runs = n_runs
        self.ref_point = ref_point
        env = make_env()
        self.n_objectives = 1 if not hasattr(env, 'reward_space') else len(env.reward_space.low)

        self.logdir = logdir
        self.logger = Logger(self.logdir)

        if indicator == 'hypervolume':
            assert ref_point is not None, 'reference point is needed for hypervolume indicator'
            self.indicator = lambda points, ref=ref_point: indicator_hypervolume(points, ref)
        elif indicator == 'non_dominated':
            self.indicator = indicator_non_dominated
        elif indicator == 'single_objective':
            self.indicator = lambda x: x.flatten()
            self.n_objectives = 1
        else:
            raise ValueError('unknown indicator, choose between hypervolume and non_dominated')
    
    def start(self):
        # make distribution
        n_params = n_parameters(self.policy)
        mu, sigma = torch.rand(n_params, requires_grad=True), torch.rand(n_params, requires_grad=True)
        mu, sigma = nn.Parameter(mu), nn.Parameter(sigma)
        self.dist = torch.distributions.Normal(mu, sigma)

        # optimizer to change distribution parameters
        self.opt = torch.optim.Adam([{'params': mu}, {'params': sigma}], lr=1e-1)

    def step(self):
        # using current theta, sample policies from Normal(theta)
        population, z = self.sample_population()
        # run episode for these policies
        returns = self.evaluate_population(self.make_env(), population)
        returns = returns.detach().numpy()

        indicator_metric = self.indicator(returns)
        metric = torch.tensor(indicator_metric)[:,None]

        # use fitness ranking TODO doesn't help
        # returns = centered_ranking(returns)
        # standardize the rewards to have a gaussian distribution
        metric = (metric - torch.mean(metric, dim=0)) / torch.std(metric, dim=0)

        # compute loss
        log_prob = self.dist.log_prob(z).sum(1, keepdim=True)

        mu, sigma = self.dist.mean, self.dist.scale

        # directly compute inverse Fisher Information Matrix (FIM)
        # only works because we use gaussian (and no correlation between variables) 
        fim_mu_inv = torch.diag(sigma.detach()**2)
        fim_sigma_inv = torch.diag(2/sigma.detach()**2)

        loss = -log_prob*metric

        # update distribution parameters
        self.opt.zero_grad()
        loss.mean().backward()
        # now that we have grads, multiply them with FIM_INV
        nat_grad_mu = fim_mu_inv@mu.grad
        nat_grad_sigma = fim_sigma_inv@sigma.grad
        mu.grad =  nat_grad_mu
        sigma.grad = nat_grad_sigma

        self.opt.step()

        return {'returns': returns, 'metric': np.mean(indicator_metric)}

    def train(self, iterations):
        self.start()

        for i in range(iterations):
            info = self.step()
            returns = info['returns']
            # logging
            self.logger.put('train/metric', info['metric'], i, 'scalar')
            self.logger.put('train/returns', returns, i, f'{returns.shape[-1]}d')
            if self.ref_point is not None:
                hv = compute_hypervolume(returns)
                self.logger.put('train/hypervolume', hv, i, 'scalar')

            print(f'Iteration {i} \t Metric {info["metric"]} \t')

        print('='*20)
        print('DONE TRAINING, LAST POPULATION ND RETURNS')
        print(non_dominated(returns))
            

    def sample_population(self):
        population = []; z = []
        for _ in range(self.n_population):
            z_i = self.dist.sample()
            m_i = copy.deepcopy(self.policy)
            set_parameters(m_i, z_i)
            population.append(m_i)
            z.append(z_i)
        return population, torch.stack(z)

    def evaluate_population(self, env, population):
        returns = torch.zeros(len(population), self.n_objectives)
        for i in range(len(population)):
            p_return = torch.zeros(self.n_runs, self.n_objectives)
            for r in range(self.n_runs):
                p_return[r] = run_episode(env, population[i])
            returns[i] = torch.mean(p_return, dim=0)
        return returns

