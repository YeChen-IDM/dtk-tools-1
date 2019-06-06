import numpy as np

# Choose points using LHS
# Separatrix Demo, Institute for Disease Modeling, May 2014

def LHSPointSelection(NumPoints=15, NumDimensions=2, ParameterRanges=[dict(Min=0, Max=1), dict(Min=0, Max=1)]):
    edges = np.linspace(0, 1, NumPoints + 1)

    step = 1 / NumPoints

    Points = np.zeros((NumPoints, NumDimensions), float)

    # Initialize 1d matrix
    Points[:, 0] = ParameterRanges[0]['Min'] + (ParameterRanges[0]['Max'] - ParameterRanges[0]['Min']) * (
            edges[np.random.permutation(NumPoints)] + step * np.random.uniform(low=ParameterRanges[0]['Min'],
                                                                               high=ParameterRanges[0]['Max'],
                                                                               size=NumPoints))

    for i in range(1, NumDimensions):
        Points[:, i] = ParameterRanges[i]['Min'] + (ParameterRanges[i]['Max'] - ParameterRanges[i]['Min']) * (
                edges[np.random.permutation(NumPoints)] + step * np.random.uniform(low=ParameterRanges[i]['Min'],
                                                                                   high=ParameterRanges[i]['Max'],
                                                                                   size=NumPoints))

    return Points