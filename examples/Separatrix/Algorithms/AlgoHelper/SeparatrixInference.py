# Generated with SMOP  0.41

import numpy as np

from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection
from examples.Separatrix.Algorithms.AlgoHelper.tanhModel import tanhModel
from examples.Separatrix.Algorithms.AlgoHelper.utils import sub2ind

ParameterRanges = [dict(Min=0, Max=1), dict(Min=0, Max=1)]
grid_res = 10

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
from examples.Separatrix.Algorithms.AlgoHelper.computeD2 import computeD2


# Zdu: my function
# def sub2ind(dim, row_inds=[], col_indx=[]):
#     prod = zip(row_inds, col_indx)
#
#     inds = []
#     for i, j in prod:
#         idx = j * dim[0] + i
#         inds.append(idx)
#
#     inds = [int(i) for i in inds]
#     return inds


def SeparatrixInference(sample_x=None, sample_y=None, inference_x=None, params=None):
    sample_y = sample_y.reshape(sample_y.shape[0], 1)

    ## Number of sample points, inference points, and dimensions
    Nsamp = np.size(sample_x, 0)

    Ninference = np.size(inference_x, 0)

    D = params["Num_Dimensions"]

    ## Compute the density of sample points at each sample point

    # Compute squared distance from each sample point to the k-th
    # nearest neighboring sample point (eqn 5)
    # zdu_t = params["c_rho"] * Nsamp ** params["gamma_rho"]
    # print(zdu_t)
    kNN_rho = np.max((params["Min_Neighbors_for_KNN"], np.floor(params["c_rho"] * Nsamp ** params["gamma_rho"])))

    D2_samp_samp = computeD2(sample_x, sample_x)

    D2_samp_samp_sorted = np.sort(D2_samp_samp, axis=0)  # sort each column in increasing

    # D_kNN_rho = sqrt(D2_samp_samp_sorted( sub2ind([Nsamp Nsamp], (kNN_rho+1)*ones(1,Nsamp), 1:Nsamp) ))';  # matlab
    # Zdu: make it as an array
    D2_samp_samp_sorted = D2_samp_samp_sorted.flatten(1)

    D_kNN_rho = np.sqrt(
        D2_samp_samp_sorted[sub2ind([Nsamp, Nsamp], (kNN_rho + 1) * np.ones((1, Nsamp))[0], range(0, Nsamp))])

    #  Compute density
    rho_xx = np.zeros((Nsamp, 1))

    for j in range(0, Nsamp):
        rho_xx[j] = (2 * np.pi) ** (-D / 2) * np.mean(
            (params["h_rho"] * D_kNN_rho) ** (-D) * np.exp(-1 / 2 * D2_samp_samp[:, j]) / (
                    params["h_rho"] * D_kNN_rho) ** 2)

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

    # D_kNN_inf = np.sqrt(
    #     D2_samp_inf_sorted[sub2ind([Nsamp, Ninference], (kNN_inference + 1).T[0], range(0, Ninference))])

    try:
        D_kNN_inf = np.sqrt(
            D2_samp_inf_sorted[sub2ind([Nsamp, Ninference], (kNN_inference + 1).T[0], range(0, Ninference))])
    except Exception as ex:
        print(ex)

    D_kNN_inf = np.minimum(params["max_g"], np.maximum(params["min_g"], D_kNN_inf)).reshape(D_kNN_inf.shape[0], 1)

    # Compute the kernel and apply to get alpha and beta
    kernel = np.diag(1.0 / rho_xx) * np.exp(-1 / 2 * D2_samp_inf * np.diag(1.0 / D_kNN_inf ** 2))

    alpha = np.dot(kernel.T, sample_y)

    beta = np.dot(kernel.T, (1 - sample_y))

    # Mu is a scaling factor so that alpha + beta = Nhat
    mu = Nhat_inf / (alpha + beta)

    alpha = mu * alpha + 1

    beta = mu * beta + 1

    return alpha, beta


if __name__ == "__main__":
    ONE_DIMENSIONAL_EXAMPLE = 1

    TWO_DIMENSIONAL_EXAMPLE = 2

    # Change the following line to select one of the above options:
    EXAMPLE = TWO_DIMENSIONAL_EXAMPLE

    ## Parameters to configure the Separatrix Algorithm
    # These parameters are for the 1D example, will change below for 2D example
    params = dict(Model='SigmoidalModel', Number_Of_Iterations=10, Random_Seed=1, Output_Dir='OutputDir',
                  Interest_Level=0.7, Num_Initial_Samples=50, Num_Next_Samples=50, Num_Test_Points=100,
                  Num_Candidates_Points=200, Scale_Model_Sigma=0.02, h_rho=0.3, c_rho=0.5, gamma_rho=0.8,
                  c_inference=4, Gamma_Inference=0.8, Min_Neighbors_for_KNN=3, Fraction_LHS=0,
                  Max_Fraction_igBDOE=0.25, min_g=0, max_g=1, MCMC_Resolution=1000, MCMC_Num_Iterations=2,
                  Blur_Sigma=0.02, Inference_Grid_Resolution=100)

    if EXAMPLE == TWO_DIMENSIONAL_EXAMPLE:
        # Changes for 2-D Example
        params["Model"] = 'tanhModel'
        params["Number_Of_Iterations"] = 20
        params["Inference_Grid_Resolution"] = 20
        params["Interest_Level"] = 0.6
        params["Scale_Model_Sigma"] = 0.05
        params["gamma_rho"] = 0.67
        params["Gamma_Inference"] = 0.67

    params["Num_Dimensions"] = EXAMPLE

    sample_x = LHSPointSelection(5, 2, ParameterRanges)
    print(sample_x)

    if params["Num_Dimensions"] == 1:
        inference_x = np.linspace(0, ParameterRanges[1]["Max"], grid_res).T
    else:
        ix, iy = np.meshgrid(np.linspace(0, ParameterRanges[0]['Max'], grid_res),
                             np.linspace(0, ParameterRanges[1]['Max'], grid_res))

        print(ix)
        print(iy)
        # inference_x = np.concatenate((ix.flatten(1), iy.flatten()), axis=1)
        # print(np.vstack((ix.flatten(1), iy.flatten(1))))
        inference_x = np.vstack((ix.flatten(1), iy.flatten(1))).T

    # Zdu: seems inference_x has no relation wtih sample_x or sample_y
    print(inference_x)

    # Compute true/false outputs at the initial sample points
    np.random.seed(1)
    myrng = np.random.rand()
    model = tanhModel(myrng=myrng)

    # Probability = model.Truth(sample_x)
    # r = np.random.uniform(low=model.myrng, high=1, size=np.size(sample_x, 0))
    # Outcomes = r < Probability
    sample_y = model.Sample(sample_x)
    print(sample_y)

    # True Separatrix
    # [TODO]: true_separatrix will be used only for plotting, we can skip it for now!
    # true_separatrix = model.TrueSeparatrix(params['Interest_Level'])
    # print(true_separatrix)

    ## Perform initial Separatrix inference
    Alpha, Beta = SeparatrixInference(sample_x, sample_y, inference_x, params)
    print(Alpha)
    print(Beta)

    print('The End.')
