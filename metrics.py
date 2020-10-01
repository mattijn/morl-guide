import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pygmo import hypervolume


def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    else:
        return solutions[is_efficient]


def compute_hypervolume(points, ref):
    # pygmo uses hv minimization,
    # negate rewards to get costs
    points = np.array(points) * -1.
    hv = hypervolume(points)
    # use negative ref-point for minimization
    return hv.compute(ref*-1)


def crowding_distance(points, ranks=None):
    crowding = np.zeros(points.shape)
    # compute crowding distance separately for each non-dominated rank
    if ranks is None:
        ranks = non_dominated_rank(points)
    unique_ranks = np.unique(ranks)
    for rank in unique_ranks:
        current_i = ranks == rank
        current = points[current_i]
        if len(current) == 1:
            crowding[current_i] = 1
            continue
        # first normalize accross dimensions
        current = (current-current.min(axis=0))/(current.ptp(axis=0)+1e-8)
        # sort points per dimension
        dim_sorted = np.argsort(current, axis=0)
        point_sorted = np.take_along_axis(current, dim_sorted, axis=0)
        # compute distances between lower and higher point
        distances = np.abs(point_sorted[:-2] - point_sorted[2:])
        # pad extrema's with 1, for each dimension
        distances = np.pad(distances, ((1,), (0,)), constant_values=1)
        
        current_crowding = np.zeros(current.shape)
        current_crowding[dim_sorted, np.arange(points.shape[-1])] = distances
        crowding[current_i] = current_crowding
    # sum distances of each dimension of the same point
    crowding = np.sum(crowding, axis=-1)
    # normalized by dividing by number of objectives
    crowding = crowding/points.shape[-1]
    return crowding


def non_dominated_rank(points):
    ranks = np.zeros(len(points), dtype=np.float32)
    current_rank = 0
    # get unique points to determine their non-dominated rank
    unique_points, indexes = np.unique(points, return_inverse=True, axis=0)
    # as long as we haven't processed all points
    while not np.all(unique_points==-np.inf):
        _, nd_i = non_dominated(unique_points, return_indexes=True)
        # use indexes to compute inverse of unique_points, but use nd_i instead
        ranks[nd_i[indexes]] = current_rank
        # replace ranked points with -inf, so that they won't be non-dominated again
        unique_points[nd_i] = -np.inf
        current_rank += 1
    return ranks


if __name__ == '__main__':
    points = np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0],
        [0.5,0.5,0],
        [0.5,0,0.5],
        [0,0.5,0.5],
        [0.5,0,0],
        [0,0.5,0],
        [0,0,0.5]
    ])

    nd, nd_i = non_dominated(points, return_indexes=True)

    print('='*20)
    print('NON DOMINATED POINTS')
    print(nd)

    assert np.all(nd == points[:6]), 'non dominated points incorrect'

    ranks = non_dominated_rank(points)

    print('='*20)
    print('NON DOMINATED RANKS')
    print(ranks)

    assert np.all(ranks == np.array([0,0,0,0,0,0,1,1,1])), 'non dominated rank incorrect'

    cd = crowding_distance(points)
    
    print('='*20)
    print('CROWDING DISTANCE')
    print(cd)

    ref_point = np.full((3,), -1)
    hv = compute_hypervolume(points, ref_point)

    print('='*20)
    print('HYPERVOLUME')
    print(hv)

