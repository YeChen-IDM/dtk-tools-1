# Generated with SMOP  0.41
# test.m
# Tanh model
#
# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
from examples.Separatrix.Algorithms.AlgoHelper.ISeparatrixModel import ISeparatrixModel


class tanhModel(ISeparatrixModel):

    def __init__(self, Model_Name='My Model', Parameter_Names=['Point_X', 'Point_Y'],
                 Parameter_Ranges=[dict(Min=0, Max=1), dict(Min=0, Max=1)], myrng=1, mx=10, bx=0.5, my=3, by=0.3,
                 EvaluationPoints=[], config=None):
        super().__init__(Model_Name, Parameter_Names, Parameter_Ranges, config)
        self.myrng = myrng
        self.mx = 10
        self.bx = 0.5
        self.my = 3
        self.by = 0.3
        self.EvaluationPoints = EvaluationPoints
        # self.ISeparatrixModel(config)  # Sets config

        self.Initialize()

    def Initialize(self):
        # self.myrng = RandStream('mt19937ar', 'seed', self.config.RandomSeed)
        np.random.seed(1)
        self.myrng = np.random.rand()

        # self.Model_Name = self.__class__.__name__

    def Sample(self, Points=None):
        # Outcomes = rand(self.myrng, np.size(Points, 0), 1) < self.Truth(Points)
        Outcomes = np.random.uniform(low=self.myrng, high=1, size=np.size(Points, 0)) < self.Truth(Points)
        Outcomes = [1 if a else 0 for a in Outcomes]

        return np.array(Outcomes)

    def Truth(self, points=None):
        x = points[:, 0]
        y = points[:, 1]
        Probability = (np.tanh(self.mx * (x - self.bx)) + 1) * (np.tanh(self.my * (y - self.by)) + 1) / 4
        return Probability

    def TrueSeparatrix(self, iso=None):
        from scipy.interpolate import interp1d
        from scipy.interpolate import interp2d

        # use the following for Interpolant
        xx, yy = np.meshgrid(np.linspace(self.Parameter_Ranges[0]["Min"], self.Parameter_Ranges[0]["Max"], 30),
                             np.linspace(self.Parameter_Ranges[1]["Min"], self.Parameter_Ranges[1]["Max"], 30))

        pts2D = np.vstack((xx.flatten(1), yy.flatten(1))).T

        # zinterp = scatteredInterpolant(pts2D, self.Truth(pts2D))
        X = pts2D[:, 0].reshape(pts2D.shape[0], 1)
        Y = pts2D[:, 1].reshape(pts2D.shape[0], 1)

        xx, yy = np.meshgrid(np.linspace(self.Parameter_Ranges[0]["Min"], self.Parameter_Ranges[0]["Max"], 1000),
                             np.linspace(self.Parameter_Ranges[1]["Min"], self.Parameter_Ranges[1]["Max"], 1000))


        # Approach #1
        # Zdu testing used to plot 2D: Approach #4: 2D contour
        # rbf = Rbf(X, Y, self.Truth(pts2D), function='linear')
        # zz = rbf(xx, yy)
        # qcs = plt.contour(xx, yy, zz, levels=[iso])
        # return qcs

        # Approach #2
        pts2D = np.vstack((xx.flatten(1), yy.flatten(1))).T
        zz = griddata(pts2D, self.Truth(pts2D), (xx, yy), method='linear')
        # print(zz)

        # Approach #3: Not working yet...
        # f = interp2d(X, Y, self.Truth(pts2D), kind='linear')  # ‘linear’, ‘cubic’, ‘quintic’}, optional
        # Z = f(X.flatten(), Y.flatten())      # ValueError: x and y should both be 1-D arrays
        # print(Z)

        ax = plt.axes(projection='3d')
        qcs = ax.contour3D(xx, yy, zz, levels=[iso], cmap='binary')
        # qcs = ax.contour3D(xx, yy, zz, iso*np.ones((2,1)), cmap='binary')   # ERROR: Contour levels must be increasing
        # qcs2 = ax.contour(X, Y, Z, 1)
        # qcs = plt.contour(X, Y, Z, levels=[iso], cmap='binary')


        return qcs


    def OverlayIsocline(self, h=None, iso=None):
        return


def test():
    np.random.seed(1)
    myrng = np.random.rand()
    model = tanhModel(myrng=myrng)
    model.TrueSeparatrix(iso=0.6)


if __name__ == "__main__":
    test()
    exit()
