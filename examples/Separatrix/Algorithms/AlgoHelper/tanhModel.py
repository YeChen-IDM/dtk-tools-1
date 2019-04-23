# Generated with SMOP  0.41
# test.m
# Tanh model
#
# Separatrix Demo, Institute for Disease Modeling, May 2014

import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from examples.Separatrix.Tests.Separatrix_test.ISeparatrixModel import ISeparatrixModel


class tanhModel(ISeparatrixModel):

    def __init__(self, Model_Name='My Model', Parameter_Names=['Point_X', 'Point_Y'], Parameter_Ranges=[dict(min=0, max=1), dict(min=0, max=1)], myrng=1, mx=10, bx=0.5, my=3, by=0.3, EvaluationPoints=[], config=None):
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
        # Use contour3 to extract the separatrix
        print('In TrueSeparatrix...')
        # return None

        l1 = np.linspace(self.Parameter_Ranges[0]["min"], self.Parameter_Ranges[0]["max"], 2)
        l2 = np.linspace(self.Parameter_Ranges[1]["min"], self.Parameter_Ranges[1]["max"], 2)

        xx, yy = np.meshgrid(np.linspace(self.Parameter_Ranges[0]["min"], self.Parameter_Ranges[0]["max"], 2),
                          np.linspace(self.Parameter_Ranges[1]["min"], self.Parameter_Ranges[1]["max"], 2))

        pts2D = np.vstack((xx.flatten(1), yy.flatten(1))).T
        print(pts2D)

        Probability = self.Truth(pts2D)

        # zinterp = scatteredInterpolant(pts2D, self.Truth(pts2D))
        # [TODO]: seems the following doesn't match to the above function!
        zinterp = griddata(pts2D, self.Truth(pts2D), (xx, yy), method='linear')
        print(zinterp)

        xx, yy = np.meshgrid(np.linspace(self.Parameter_Ranges[0]["min"], self.Parameter_Ranges[0]["max"], 5),
                          np.linspace(self.Parameter_Ranges[1]["min"], self.Parameter_Ranges[1]["max"], 5))

        # print(xx)
        # print(yy)
        zz = zinterp(xx, yy)
        print(zz)

        # h = figure(235235)

        true_separatrix, __ = contour3(xx, yy, zinterp(xx, yy), iso*np.ones((2, 1)))

        true_separatrix = true_separatrix.T

        # close_(h)
        return true_separatrix


    def OverlayIsocline(self, h=None, iso=None):
        # pts = np.linspace(0, 1, 30)
        #
        # xx, yy = np.meshgrid(pts, pts, nargout=2)
        #
        # set(0, 'currentfigure', h)
        # hold('on')
        # __, hlines = contour3(xx, yy, reshape(self.Truth(np.concatenate((xx.ravel()), yy.ravel())), 30, 30), np.dot(iso, ones(2, 1)),
        #                       'k--', nargout=2)
        #
        # set(hlines, 'linewidth', 2)
        # set(hlines, 'zdata', get(hlines, 'zdata') + 1)
        return
