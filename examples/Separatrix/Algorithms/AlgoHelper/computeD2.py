# Generated with SMOP  0.41


# Compute matrix of squared distances

# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
from numpy.matlib import repmat


def computeD2(row_points=None, col_points=None):
    Nrows = np.size(row_points, 0)

    Ncols = np.size(col_points, 0)

    Ndim = np.size(row_points, 1)

    D2 = np.zeros((Nrows, Ncols))

    for d in range(0, Ndim):
        D2 = D2 + (np.kron(np.ones((1, Ncols)), row_points[:, d].reshape(Nrows, 1)) - np.kron(np.ones((Nrows, 1)),
                                                                                              col_points[:, d])) ** 2

    return D2


def demo():
    data = [[3, 6, 5], [7, -2, 4], [1, 0, -9]]

    A = np.array(data)

    row_points = A
    col_points = A

    D2 = computeD2(row_points, col_points)
    print(D2)


if __name__ == "__main__":
    demo()
