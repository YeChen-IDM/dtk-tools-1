# Generated with SMOP  0.41


# Generate test and possible sample points for the next iteration
# by using MCMC (Gibbs sampler) to draw samples from the variance of the
# interest distribution.  Then follow-up with Gaussian blur.

# Separatrix Demo, Institute for Disease Modeling, May 2014

import math
import numpy as np
import copy
import random
from scipy.interpolate import interp1d
from examples.Separatrix.Algorithms.AlgoHelper.LHSPointSelection import LHSPointSelection
from examples.Separatrix.Algorithms.AlgoHelper.utils import find

from scipy.interpolate import Rbf, InterpolatedUnivariateSpline


def ChooseTestAndSamplePointsForNextIteration(testPoints=None, Interest_x=None, InterestDistributionTestPoints=None,
                                              InterestDistributionInferencePoints=None, possibleSamplePoints=None,
                                              ParameterRanges=None, params=None):
    mcmcTestPoints = testPoints

    mcmcPossibleSamplePoints = possibleSamplePoints

    MCMCres = params["MCMC_Resolution"]

    # [TODO]: zdu, maybe there is a bug
    NtestPoints = np.size(mcmcTestPoints, 0)

    NpossibleSamplePoints = np.size(possibleSamplePoints, 0)

    # Note that we are using the variance at both the test points and at the inference points.
    # Coordinates:
    FullInterestVariancePoints = np.concatenate((testPoints, Interest_x), axis=0)

    # Interest distribution variance:
    FullInterestDistribution = np.concatenate((InterestDistributionTestPoints, InterestDistributionInferencePoints),
                                              axis=0)

    FullInterestVariance = FullInterestDistribution[:, 0] - FullInterestDistribution[:, 0] ** 2

    # Avoid duplicates
    UniqueInterestVariancePoints, UniqueInterestVariancePointsInds = np.unique(FullInterestVariancePoints, axis=0,
                                                                               return_index=True)

    UniqueInterestVariance = FullInterestVariance[UniqueInterestVariancePointsInds]

    # Want to infer the variance of the interest distribution at MCMCres points.
    # Instead of directly calculating the variance, we interpolate.
    # However, this interpolation can go badly due to the fact that probability
    # is constrained to [0,1].  To address this constraint, we do the interpolation
    # in logit space and then return to linear space.

    # Interpolation gets confused at -Inf and Inf logit,
    # so map probability to ( smallNumber, 1-smallNumber )
    logitInterestVariance = np.log(UniqueInterestVariance / (1 - UniqueInterestVariance))
    logitInterestVariance = logitInterestVariance.reshape(logitInterestVariance.shape[0], 1)

    if params["Num_Dimensions"] == 1:
        InterpPts = LHSPointSelection(MCMCres, 1, ParameterRanges)

        x = np.linspace(0, UniqueInterestVariancePoints.shape[0], UniqueInterestVariancePoints.shape[0])
        x = x.reshape(x.shape[0], 1)
        f = interp1d(x.ravel(), UniqueInterestVariancePoints.ravel(), kind='linear')
        interpval = f(InterpPts)

        InterpolatedInterestVariancePDF = 1. / (1 + np.exp(-interpval))

        inds = np.isnan(InterpolatedInterestVariancePDF)
        InterpolatedInterestVariancePDF[inds] = 0

        # Replace points with the randomly selected values
        try:
            for i in range(0, params["Num_Test_Points"]):
                mcmcTestPoints[i, 0] = InterpPts[roulette(InterpolatedInterestVariancePDF)]

            for i in range(0, NpossibleSamplePoints):
                mcmcPossibleSamplePoints[i, 0] = InterpPts[roulette(InterpolatedInterestVariancePDF)]
        except Exception as ex:
            print(ex)

    else:  # 2-D
        # ZDU: will use Rbf to replace scatteredInterpolant
        # zinterp = scatteredInterpolant(UniqueInterestVariancePoints, logitInterestVariance)
        X = UniqueInterestVariancePoints[:, 0].reshape(UniqueInterestVariancePoints.shape[0], 1)
        Y = UniqueInterestVariancePoints[:, 1].reshape(UniqueInterestVariancePoints.shape[0], 1)
        rbf = Rbf(X, Y, logitInterestVariance, function='linear')

        for iteration in range(0, params["MCMC_Num_Iterations"]):
            for dimension in range(0, params["Num_Dimensions"]):
                InterpPts = LHSPointSelection(MCMCres, 1, ParameterRanges)

                for ii in range(0, params["Num_Test_Points"]):
                    ip = np.kron(np.ones((MCMCres, 1)), mcmcTestPoints[ii, :])
                    ip[:, dimension] = np.array(InterpPts[:, 0])
                    InterpolatedInterestVariancePDF = 1 / (1 + np.exp(-rbf(ip[:, 0], ip[:, 1])))
                    inds = np.isnan(InterpolatedInterestVariancePDF).astype(int)
                    InterpolatedInterestVariancePDF[inds] = 0
                    mcmcTestPoints[ii, dimension] = InterpPts[roulette(InterpolatedInterestVariancePDF, MCMCres)][0]

                for ii in range(0, NpossibleSamplePoints):
                    sp = np.kron(np.ones((MCMCres, 1)), mcmcPossibleSamplePoints[ii, :])
                    sp[:, dimension] = np.array(InterpPts[:, 0])
                    InterpolatedInterestVariancePDF = 1 / (1 + np.exp(- rbf(sp[:, 0], sp[:, 1])))
                    inds = np.isnan(InterpolatedInterestVariancePDF).astype(int)
                    InterpolatedInterestVariancePDF[inds] = 0
                    mcmcPossibleSamplePoints[ii, dimension] = InterpPts[roulette(InterpolatedInterestVariancePDF)][0]

    if params["Blur_Sigma"] > 0:
        # Apply Gaussian blur
        for dimension in range(0, params["Num_Dimensions"]):
            tp_blur = mcmcTestPoints[:, dimension] + params[
                "Blur_Sigma"] * np.random.standard_normal(NtestPoints)  # Bug?? params["Num_Test_Points"]
            inds = find((tp_blur < 0).astype(int) | (tp_blur > 1).astype(int))
            tp_blur[inds] = mcmcTestPoints[inds, dimension]
            mcmcTestPoints[:, dimension] = tp_blur

            psp_blur = mcmcPossibleSamplePoints[:, dimension] + params[
                "Blur_Sigma"] * np.random.standard_normal(NpossibleSamplePoints)
            inds = find((psp_blur < ParameterRanges[dimension]["Min"]).astype(int) | (
                    psp_blur > ParameterRanges[dimension]["Max"]).astype(int))
            psp_blur[inds] = mcmcPossibleSamplePoints[inds, dimension]
            mcmcPossibleSamplePoints[:, dimension] = psp_blur

    return mcmcTestPoints, mcmcPossibleSamplePoints


# HELPER FUNCTIONS:
def roulette(pdf=None, MCMCres=None):
    # Roulette wheel selection from pdf
    cdf = pdf.cumsum(axis=0)

    r = random.random() * cdf[-1]
    idx = find(cdf > r)

    if idx is None or len(idx) == 0:
        return MCMCres
    else:
        return idx[0][0]
