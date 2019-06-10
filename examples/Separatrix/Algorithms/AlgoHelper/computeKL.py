# Generated with SMOP  0.41

# Compute KL divergence between distributions in p and those in q

# Separatrix Demo, Institute for Disease Modeling, May 2014
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