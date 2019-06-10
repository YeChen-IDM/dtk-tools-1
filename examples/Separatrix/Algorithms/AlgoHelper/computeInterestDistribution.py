# Generated with SMOP  0.41


# Compute the interest distribution
# Integrate the Beta distribution from zero to Interest_Level

# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
from scipy.special import betainc


def computeInterestDistribution(alpha=None, beta=None, Interest_Level=None):
    interestDistrib = np.zeros((np.size(alpha, 0), 2), float)

    for j in range(0, np.size(alpha, 0)):
        tmp = betainc(alpha[j, 0], beta[j, 0], [Interest_Level, 1])

        # Second component is mass above separatrix
        # interestDistrib(j,:) = [tmp(1); diff(tmp)];       # Matlab
        interestDistrib[j, :] = np.concatenate(([tmp[0]], np.diff(tmp, n=1, axis=0)))

    return interestDistrib
