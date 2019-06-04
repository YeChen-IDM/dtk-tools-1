import numpy as np
from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection


# def sub2ind(shape, row_inds=[], col_indx=[]):
#     inds = np.ravel_multi_index([row_inds, col_indx], shape, order='F')
#
#     return sorted(inds)
#
#
# def ind2sub(shape, index=[]):
#     row_inds, col_indx = np.unravel_index(index, shape, order='F')
#
#     return row_inds, col_indx
#
#
# def get_data_by_index(X, index=[]):
#     # row_inds, col_indx = ind2sub(X.shape, index)
#     row_inds, col_indx = np.unravel_index(index, X.shape, order='F')
#
#     return X[row_inds, col_indx]


def sub2ind(dim, row_inds=[], col_indx=[]):
    prod = zip(row_inds, col_indx)

    inds = []
    for i, j in prod:
        idx = j * dim[0] + i
        inds.append(idx)

    inds = [int(i) - 1 for i in inds]   # Python is 0-based
    return inds


def find(X):
    """
    Replace Matlab: find function
    import matplotlib.mlab as mlab
    mlab.find method is good but received warrning:
        MatplotlibDeprecationWarning: The find function was deprecated in Matplotlib 2.2 and will be removed in 3.1.
    :param X: matrix
    :return: np.array
    """
    # G = np.where(X > 0)
    # inds = np.ravel_multi_index([G[0], G[1]], X.shape, order='F')
    inds = (X > 0).ravel(1).nonzero()

    return sorted(inds)


def generate_requested_points(Num_Points, Num_Dimensions, ParameterRanges):
    # Use LHS to create points for now
    points = LHSPointSelection(Num_Points, Num_Dimensions, ParameterRanges)

    # Hit the corners (required for current MCMC implementation)
    points = zeroCorners(points)
    return points


def zeroCorners(pts=None):
    def ismember(a, b):
        """
        https://stackoverflow.com/questions/22699756/python-version-of-ismember-with-rows-and-index
        """
        if pts.shape[1] == 1:
            return np.flatnonzero(np.in1d(b[:, 0], a[:, 0]))
        else:  # 2-D
            return np.flatnonzero(np.in1d(b[:, 0], a[:, 0]) & np.in1d(b[:, 1], a[:, 1]))

    def increment(vec=None):
        fz = np.size(vec, axis=1)
        fz = fz - 1
        while vec[0, fz] == 1:
            vec[0, fz] = 0
            fz = fz - 1

        vec[0, fz] = 1
        return vec

    Ndim = np.size(pts, 1)
    vec = np.zeros((1, Ndim), float)
    cnt = 0
    while vec.sum() < Ndim:

        # if not any(ismember(pts, vec, 'rows')):
        if not any(ismember(pts, vec)):
            pts[cnt, :] = vec
            cnt = cnt + 1
        vec = increment(vec)

    if not any(ismember(pts, vec)):
        pts[cnt, :] = vec

    return pts
