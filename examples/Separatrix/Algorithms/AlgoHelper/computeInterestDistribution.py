# Generated with SMOP  0.41


# Compute the interest distribution
# Integrate the Beta distribution from zero to Interest_Level

# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
from scipy.special import betainc


def computeInterestDistribution(alpha=None, beta=None, Interest_Level=None):
    interestDistrib = np.zeros((np.size(alpha, 0), 2), float)
    # print(interestDistrib.shape)
    # print(interestDistrib)

    for j in range(0, np.size(alpha, 0)):
        tmp = betainc(alpha[j, 0], beta[j, 0], [Interest_Level, 1])
        # print("j={} - ({}, {}): {}".format(j, alpha[j, 0], beta[j, 0], tmp))

        # Second component is mass above separatrix
        # interestDistrib(j,:) = [tmp(1); diff(tmp)];       # Matlab
        interestDistrib[j, :] = np.concatenate(([tmp[0]], np.diff(tmp, n=1, axis=0)))

    return interestDistrib


def test():
    np.random.seed(1)
    r = np.random.rand(10, 1)
    print(r)

    np.random.seed(2)
    r = np.random.rand(10, 1)
    print(r)


def main():
    np.random.seed(1)
    alpha = np.random.uniform(low=0, high=1, size=(10, 1))

    np.random.seed(2)
    beta = np.random.uniform(low=0, high=1, size=(10, 1))

    print(alpha)
    print(beta)
    exit()

    np.random.seed(1)
    alpha = np.random.rand(10, 1)

    np.random.seed(2)
    beta = np.random.rand(10, 1)

    print(alpha)
    print(beta)

    interestDistrib = computeInterestDistribution(alpha, beta, 0.6)
    print(interestDistrib)
    print('The End.')


if __name__ == "__main__":
    main()
    exit()

    test()
    exit()
