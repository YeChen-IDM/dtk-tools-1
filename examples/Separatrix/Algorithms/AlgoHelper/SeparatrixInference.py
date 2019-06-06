# Generated with SMOP  0.41

import numpy as np
from examples.Separatrix.Algorithms.AlgoHelper.utils import sub2ind
from examples.Separatrix.Algorithms.AlgoHelper.computeD2 import computeD2


# SeparatrixInference performs the inferece step of the Separatrix Algorithm.
# Inputs:
#  sample_x: (Number of samples) by (number of dimensions) of sample points
#  sample_y: (Number of samples) by one of {0,1} results
#  inference_x: (Number of samples) by (number of dimensions) of inference points
#  params: parameters structure

# Outputs:
#  alpha: Effective number of successes at each inference point, respectively
#  beta: Effective number of failures at each inference point, respectively

# The resulting success probability distribution at each inference point
# can be computed by plugging alpha and beta into a Beta distribution.

# Separatrix Demo, Institute for Disease Modeling, May 2014


def SeparatrixInference(sample_x=None, sample_y=None, inference_x=None, params=None):
    sample_y = sample_y.reshape(sample_y.shape[0], 1)

    ## Number of sample points, inference points, and dimensions
    Nsamp = np.size(sample_x, 0)

    Ninference = np.size(inference_x, 0)

    D = params["Num_Dimensions"]

    ## Compute the density of sample points at each sample point

    # Compute squared distance from each sample point to the k-th
    # nearest neighboring sample point (eqn 5)
    kNN_rho = np.max((params["Min_Neighbors_for_KNN"], np.floor(params["c_rho"] * Nsamp ** params["gamma_rho"])))

    D2_samp_samp = computeD2(sample_x, sample_x)

    D2_samp_samp_sorted = np.sort(D2_samp_samp, axis=0)  # sort each column in increasing

    D2_samp_samp_sorted = D2_samp_samp_sorted.flatten(1)

    D_kNN_rho = np.sqrt(
        D2_samp_samp_sorted[sub2ind([Nsamp, Nsamp], (kNN_rho + 1) * np.ones((1, Nsamp))[0], range(0, Nsamp))])

    #  Compute density
    rho_xx = np.zeros((Nsamp, 1))

    for j in range(0, Nsamp):
        rho_xx[j] = (2 * np.pi) ** (-D / 2) * np.mean(
            (params["h_rho"] * D_kNN_rho) ** (-D) * np.exp(-1 / 2 * D2_samp_samp[:, j] / (
                    params["h_rho"] * D_kNN_rho) ** 2))

    ## Apply the density-corrected kernel

    # Compute squared distance from each inference point to to the k-th
    # nearest neighboring sample point (eqn 5)
    D2_samp_inf = computeD2(sample_x, inference_x)

    # Compute number of sample points at each inference point using a
    # scale model (eqn 6)
    Nhat_inf = (np.exp(-1 / (2 * params["Scale_Model_Sigma"] ** 2) * D2_samp_inf)).sum(axis=0).reshape(
        np.size(D2_samp_inf, 1), 1)

    kNN_inference = np.maximum(params["Min_Neighbors_for_KNN"],
                               np.floor(params["c_inference"] * Nhat_inf ** params["Gamma_Inference"]))

    D2_samp_inf_sorted = np.sort(D2_samp_inf, axis=0)

    D2_samp_inf_sorted = D2_samp_inf_sorted.flatten(1)

    D_kNN_inf = np.sqrt(
        D2_samp_inf_sorted[sub2ind([Nsamp, Ninference], (kNN_inference + 1).T[0], range(0, Ninference))])

    D_kNN_inf = np.minimum(params["max_g"], np.maximum(params["min_g"], D_kNN_inf)).reshape(D_kNN_inf.shape[0], 1)

    # Compute the kernel and apply to get alpha and beta
    kernel = np.dot(np.diag((1.0 / rho_xx).flatten()),
                    np.exp(-1 / 2 * np.dot(D2_samp_inf, np.diag((1.0 / D_kNN_inf ** 2).flatten()))))

    alpha = np.dot(kernel.T, sample_y)
    beta = np.dot(kernel.T, (1 - sample_y))

    # Mu is a scaling factor so that alpha + beta = Nhat
    mu = Nhat_inf / (alpha + beta)

    alpha = mu * alpha + 1  # Add one for Beta distribution
    beta = mu * beta + 1

    return alpha, beta
