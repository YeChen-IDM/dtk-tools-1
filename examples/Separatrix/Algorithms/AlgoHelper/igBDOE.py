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

import os
import numpy as np


def igBDOE(sample_x=None, sample_y=None, inference_x=None, ParameterRanges=None, params=None):
    return sample_x

