import torch
import numpy as np


def get_corner_weights(solutions, return_):
    # you need at least one discovered solution to make intersections
    if not len(solutions):
        return torch.tensor([]).view(0, return_.shape[0]), torch.tensor([])
    # given current list of returns,
    # combine intersection of all combinations of them with newly discovered return
    n_objectives = solutions.shape[1]
    all_combs = torch.tensor([]).view(0, n_objectives, n_objectives+1)
    # intersection with n (2+) planes (n-1 old, 1 new)
    for n_returns in range(2, min(solutions.shape[0], n_objectives-1)+2):
        combs = torch.combinations(torch.arange(solutions.shape[0]), r=n_returns-1)
        combs = solutions[combs]
        combs = torch.cat((combs, return_.repeat(combs.shape[0], 1, 1)), dim=1)
        # we have combinations of the intersections of i+1 vertices
        # fix weights to zero for missing rows
        missing_rows = n_objectives-n_returns
        if n_returns < n_objectives:
            # fill missing rows with all possible combinations of w_i=0
            tofill = torch.combinations(torch.arange(n_objectives), missing_rows)
            zeros = torch.zeros(tofill.shape[0], missing_rows, n_objectives)
            for i in range(missing_rows):
                zeros[torch.arange(zeros.shape[0]), i, tofill[:,i]] = 1
            # append them for all combinations
            zeros = zeros.repeat(combs.shape[0], 1, 1)
            combs = combs.repeat_interleave(tofill.shape[0], dim=0)
            # make batch of squared matrices
            combs = torch.cat((combs, zeros), dim=1)
        # add utility (sum of returns - utility = 0),
        # except for added dim (w_i*1 - 0 = 0)
        u = torch.tensor([-1]*n_returns+[0]*missing_rows, dtype=torch.float)
        u = u.repeat(combs.shape[0], 1).unsqueeze(-1)
        combs = torch.cat((combs, u), dim=-1)

        all_combs = torch.cat((all_combs, combs), 0)

    # now that we have all intersection planes, 
    # we can compute the common set of weights for each intersection
    # sum of weights = 1
    w = torch.tensor([1]*n_objectives+[0], dtype=torch.float).repeat(all_combs.shape[0], 1, 1)
    a = torch.cat((all_combs, w), dim=1)

    # we are only taking into account full rank matrices
    s = a.svd().S
    full_rank = s[:,-1] > 1e-6
    a, s = a[full_rank], s[full_rank]
    # solve set of linear equations where 
    # return@weight - utility = 0,
    # leftover weights = 0,
    # sum of weights = 1,
    b = torch.zeros(a.shape[0], a.shape[1], 1)
    b[:,-1] = 1
    res = torch.solve(b, a)[0]
    # split weights and utility
    weights, utility = res[:,:-1], res[:, -1]
    # only keep weights with positive values
    # weights = weights[torch.all(weights >= 0, dim=1).flatten()]
    positive = torch.all(weights >= 0, dim=1).flatten()
    # only keep weights that sum to one
    # weights = weights[(torch.sum(weights, 1) == 1).flatten()]
    normalized = ((torch.sum(weights, 1) - 1).abs() < 1e-6).flatten()
    valid = torch.logical_and(positive, normalized)
    weights, utility = weights[valid], utility[valid]
    return weights.squeeze(-1), utility.squeeze(-1)


def get_obsolete_weights(corner_weights, corner_utilities, solutions):
    # check for each corner weight if it is on the current coverage set
    # compute the utility of each corner weight with each solution
    cw = corner_weights.expand(solutions.shape[0], *corner_weights.shape)
    u_on_returns = (cw*solutions.unsqueeze(1)).sum(-1)
    # get the highest utility of each corner weight (wrt solutions)
    max_u = u_on_returns.max(dim=0).values
    # compare this utility with the corner_utility, discard if higher
    dominated = corner_utilities.squeeze(-1) < max_u-1e-6
    return dominated


def get_optimistic_bounds(corner_weights, weights, returns):
    # use sawtooth algorithm to approximate optimistic bound of each corner_weight
    weights = weights.abs()
    non_corners = torch.all(weights != 1., dim=1)
    simplex_bound = returns.max(dim=0)[0]
    if torch.any(non_corners):
        w = weights[non_corners]
        cw = corner_weights.expand(w.shape[0], *corner_weights.shape)
        mw = cw/w[:,None]
        mw[torch.isnan(mw)] = np.inf
        mw = mw.min(dim=-1)[0]
        f = (w*returns[non_corners]).sum(-1) - w@simplex_bound
        min_f = (mw*f[:,None]).min(dim=0)[0]
        optimistic_utility = min_f + corner_weights@simplex_bound
    else:
        optimistic_utility = corner_weights@simplex_bound

    cw = corner_weights.expand(weights.shape[0], *corner_weights.shape)
    current_utility = (cw@returns.unsqueeze(2)).max(0)[0]

    improvement = optimistic_utility - current_utility.squeeze(-1)
    return improvement


class OLS(object):

    def __init__(self, 
                 make_env,
                 make_agent=None):
        self.make_env = make_env
        self.make_agent = make_agent

        self.nO = 2 #make_env().reward_space.n

    def train(self,
              timesteps=np.inf, 
              eval_freq=np.inf,
              **kwargs):
        assert timesteps != np.inf, 'please use timesteps for OLS'

        corner_weights = torch.diag(torch.ones(self.nO))
        corner_utilities = torch.ones(len(corner_weights),)*np.inf
        corner_improvement = torch.arange(len(corner_weights))*-1

        solutions = torch.tensor([]).view(0, self.nO)
        solution_weights = torch.tensor([]).view(0, self.nO)

        t = 0
        while t < timesteps and len(corner_improvement) > 0:
            t += 1
            # choose weight with highest improvement bound
            best_weight_i = torch.argmax(corner_improvement)
            weight = corner_weights[best_weight_i]
            # remove it and its associated values
            to_keep = torch.ones_like(corner_utilities, dtype=bool)
            to_keep[best_weight_i] = False

            corner_weights = corner_weights[to_keep]
            corner_utilities = corner_utilities[to_keep]
            corner_improvement = corner_improvement[to_keep]

            # use chosen weights to make SO environment
            env = self.make_env(weight)
            return_ = self.make_agent(env)
            # return_, _ = self.make_agent(env).eval()

            # if the solver found new return, update
            if not torch.any(torch.all(return_ == solutions, dim=1)):
                # update corner weights
                new_weights, new_utilities = get_corner_weights(solutions, return_)
                if len(new_weights):
                    corner_weights = torch.cat((corner_weights, new_weights), dim=0)
                    corner_utilities = torch.cat((corner_utilities, new_utilities), dim=0)

                # update known solutions
                solutions = torch.cat((solutions, return_[None,:]), dim=0)
                solution_weights = torch.cat((solution_weights, weight[None,:]), dim=0)
                # due to new solution, some corner weights are obsolete
                obsolete = get_obsolete_weights(corner_weights, corner_utilities, solutions)

                if torch.any(obsolete):
                    corner_weights = corner_weights[torch.logical_not(obsolete)]
                    corner_utilities = corner_utilities[torch.logical_not(obsolete)]

                # first ensure the extremas are chosen as weights, after that, use bounds
                if timesteps >= self.nO:
                    corner_improvement = get_optimistic_bounds(corner_weights, solution_weights, solutions)
                else:
                    corner_improvement = torch.arange(len(corner_weights))*-1

                print(f'New solution, current size: {len(solutions)} \t total calls {t} \t weight queue {len(corner_weights)}')

        return solutions 


if __name__ == '__main__':
    from gym.spaces import Discrete

    def parabol(n=5, nO=2):
        x = torch.linspace(0, 1, n)
        x = torch.combinations(x, nO-1, with_replacement=True)
        y = 1-torch.sum(x**2, dim=1, keepdim=True)
        pf = torch.cat((x, y), dim=1)
        return pf

    nO=5
    pf = parabol(5, nO)

    def make_env(weight=None):
        if weight is None:
            class T(object):
                def __init__(self):
                    self.reward_space = Discrete(nO)
            return T()
        return pf@weight

    class make_agent(object):
        def __init__(self, utilities):
            self.utilities = utilities
        def eval(self):
            return pf[torch.argmax(self.utilities)], {}

    ols = OLS(make_env, make_agent)
    ols.train(timesteps=10000)

