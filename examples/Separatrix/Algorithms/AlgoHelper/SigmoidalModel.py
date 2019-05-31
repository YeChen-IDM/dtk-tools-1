# Sigmoidal model
#
# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
from examples.Separatrix.Algorithms.AlgoHelper.ISeparatrixModel import ISeparatrixModel


class SigmoidalModel(ISeparatrixModel):
    def __init__(self, myrng, config=None):
        super().__init__(None, None, None, config)  # Sets config
        self.myrng = myrng
        self.Parameter_Ranges = [dict(min=0, max=1)]
        self.Parameter_Names = {'Parameter x'}
        self.Initialize()

    def Initialize(self):
        self.Model_Name = self.__class__.__name__

        # self.myrng = RandStream('mt19937ar', 'seed', self.config.RandomSeed)
        np.random.seed(1)
        self.myrng = np.random.rand()

    def Sample(self, Points):
        Outcomes = np.random.uniform(low=self.myrng, high=1, size=(np.size(Points, 0), 1)) < self.Truth(Points)
        Outcomes = [1 if a else 0 for a in Outcomes]

        return np.array(Outcomes)

    def TrueSeparatrix(self, interestLevel):
        Separatrix = 1 / 10 * np.arctanh(2 * interestLevel - 1) + 0.6
        return Separatrix

    def Truth(self, points):
        if (np.size(points, 1) == 1):  # 1D
            Probability = 1 / 2 * (np.tanh(10 * (points - 0.6)) + 1)
            return Probability

        print('SinusoidalModel is only compatible with 1D input data')
        Probability = []
        return Probability
