# Generated with SMOP  0.41

# Compute KL divergence between distributions in p and those in q

# Separatrix Demo, Institute for Disease Modeling, May 2014
import json
import numpy as np
from examples.Separatrix.Algorithms.AlgoHelper.utils import find


def computeKL(p=None, q=None):
    D_KL = 0
    for si in range(0, p.shape[0]):
        delta_D_KL = np.dot(np.log2(p[si, :] / q[si, :]), p[si, :].T)

        if np.isnan(delta_D_KL) or np.isinf(delta_D_KL):
            domain = find(p[si, :] > 0 and q[si, :] > 0)
            delta_D_KL = np.dot(np.log2(p[si, domain] / q[si, domain], p[si, domain]).T)

        D_KL = D_KL + delta_D_KL

    return D_KL


def demo():
    from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection

    # Load Separatrix settings
    Settings = json.load(open('../../Settings.json', 'r'))

    ParameterRanges = [dict(Min=0, Max=1), dict(Min=0, Max=1)]
    grid_res = 6

    p = LHSPointSelection(10, ParameterRanges)
    print(p)

    q = 0.7 * p
    print(q)

    r = computeKL(p, q)
    print(r)


def demo2():
    np.random.seed(1)
    p = np.random.rand(2, 10).T  # 10 x 2 to match Matlab: p = rand(10,2)
    # print(p)

    # p = np.round(p, 4)
    # print(p)

    np.random.seed(2)
    q = np.random.rand(2, 10).T  # 10 x 2 to match Matlab: q = rand(10,2)
    # print(q)

    # q = np.round(q, 4)
    # print(q)

    # p = np.array([[0.4170, 0.0001], [0.7203, 0.3023]])
    # q = np.array([[0.4360, 0.5497], [0.0259, 0.4353]])

    r = computeKL(p, q)
    print(r)


if __name__ == "__main__":
    # demo()
    # exit()

    demo2()
    exit()
