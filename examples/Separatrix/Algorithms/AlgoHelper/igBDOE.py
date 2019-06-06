# Generated with SMOP  0.41
# igBDOE_test.m

# igBDOE performs the "interest guided Bayesian design of experiments"
# step of the Separatrix Algorithm.

# Inputs:
#  sample_x: (Number of samples) by (number of dimensions) of sample points
#  sample_y: (Number of samples) by one of {0,1} results
#  inference_x: (Number of samples) by (number of dimensions) of inference points
#  ParameterRanges: Struct of min and max of each parameter, min=0 and max=1 required for this implementation
#  params: structure of fixed parameters

# Outputs:
#  new_sample_x: Num_Next_Samples by (number of dimensions) matrix of new points to sample

# Separatrix Demo, Institute for Disease Modeling, May 2014
import numpy as np
from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection
from examples.Separatrix.Algorithms.AlgoHelper.ChooseTestAndSamplePointsForNextIteration import \
    ChooseTestAndSamplePointsForNextIteration
from examples.Separatrix.Algorithms.AlgoHelper.SeparatrixInference import SeparatrixInference
from examples.Separatrix.Algorithms.AlgoHelper.computeInterestDistribution import computeInterestDistribution
from examples.Separatrix.Algorithms.AlgoHelper.computeKL import computeKL


def igBDOE(sample_x=None, sample_y=None, inference_x=None, ParameterRanges=None, params=None, testPoints=None,
           possibleSamplePoints=None):
    NumPoints = np.size(sample_x, 0)
    # Read parameters from config:
    NumNextPoints = params["Num_Next_Samples"]
    # Determine how many LHS points to take:
    NumLHS = np.maximum(
        params["Num_Next_Samples"] - np.minimum(np.floor(params["Max_Fraction_igBDOE"] * NumPoints), NumNextPoints),
        np.floor(params["Fraction_LHS"] * NumNextPoints))
    # Remaining points to be computed using BDOE:
    NumNextPoints = NumNextPoints - NumLHS
    # If all LHS, take shortcut and return:
    if NumNextPoints <= 0:
        new_sample_x = LHSPointSelection(params["Num_Next_Samples"], params["Num_Dimensions"], ParameterRanges)
        return new_sample_x, testPoints, possibleSamplePoints

    # Compute inference at possible sample points, used to compute mu (the
    # expected success probability at each possible sample point)
    mu_alpha, mu_beta = SeparatrixInference(sample_x, sample_y, possibleSamplePoints, params)
    # This is just for the MCMC selection of test and possible sample points
    # for the next iteration:
    alpha, beta = SeparatrixInference(sample_x, sample_y, inference_x, params)
    InterestDistributionInferencePoints = computeInterestDistribution(alpha, beta, params["Interest_Level"])
    # Now compute the (base) interest distribution at the test points:
    alpha_test, beta_test = SeparatrixInference(sample_x, sample_y, testPoints, params)
    InterestDistributionTestPoints = computeInterestDistribution(alpha_test, beta_test, params["Interest_Level"])
    # Compute expected KL divergence at each possible sample point
    NpossibleSamplePoints = np.size(possibleSamplePoints, 0)
    ED_KL = np.zeros((NpossibleSamplePoints, 1))
    for i in range(0, NpossibleSamplePoints):
        # Recompute inference with success at ith possible sample point
        alpha_success, beta_success = SeparatrixInference(
            np.concatenate((sample_x, possibleSamplePoints[i, :].reshape(1, possibleSamplePoints.shape[1])), axis=0),
            np.append(sample_y, 1), testPoints, params)
        InterestDistributionSuccess = computeInterestDistribution(alpha_success, beta_success, params["Interest_Level"])
        D_KL_success = computeKL(InterestDistributionTestPoints, InterestDistributionSuccess)
        alpha_fail, beta_fail = SeparatrixInference(
            np.concatenate((sample_x, possibleSamplePoints[i, :].reshape(1, possibleSamplePoints.shape[1])), axis=0),
            np.append(sample_y, 0), testPoints, params)
        InterestDistributionFail = computeInterestDistribution(alpha_fail, beta_fail, params["Interest_Level"])
        D_KL_failure = computeKL(InterestDistributionTestPoints, InterestDistributionFail)
        mu = mu_alpha[i] / (mu_alpha[i] + mu_beta[i])
        ED_KL[i] = mu * D_KL_success + (1 - mu) * D_KL_failure

    # Choose next sample points based on expected KL divergence
    NextPointsInds = ED_KL.argsort(axis=0)[::-1]
    NextPointsInds = NextPointsInds[0:int(NumNextPoints)]
    NextPointsInds = list(NextPointsInds[:, 0])
    new_sample_x = possibleSamplePoints[NextPointsInds, :]

    # Add in LHS, if any
    if NumLHS > 0:
        new_sample_x = np.concatenate(
            (new_sample_x, LHSPointSelection(int(NumLHS), params["Num_Dimensions"], ParameterRanges)), axis=0)

    # MCMC to distribute from posterior of interest variance
    mcmcTestPoints, mcmcPossibleSamplePoints = ChooseTestAndSamplePointsForNextIteration(testPoints, inference_x,
                                                                                         InterestDistributionTestPoints,
                                                                                         InterestDistributionInferencePoints,
                                                                                         possibleSamplePoints,
                                                                                         ParameterRanges, params)

    return new_sample_x, mcmcTestPoints, mcmcPossibleSamplePoints
