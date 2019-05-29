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


if __name__ == "__main__":
    import pandas as pd

    dim = 1
    Points = LHSPointSelection(10, NumDimensions=dim, ParameterRanges=[dict(Min=0, Max=1), dict(Min=0, Max=1)])
    print(Points)

    if dim == 2:
        row_count = np.size(Points, 0)
        col_count = np.size(Points, 1)
        print(row_count, ", ", col_count)

        df = pd.DataFrame(Points)
        df.columns = ['Point_X', 'Point_Y']
        print(df)
    elif dim == 1:
        df = pd.DataFrame(Points)
        df.columns = ['Point_X']
        print(df)
