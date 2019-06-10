# Separatrix model interface
#
# Separatrix Demo, Institute for Disease Modeling, May 2014

from examples.Separatrix.Algorithms.AlgoHelper.BaseHandel import BaseHandel


class ISeparatrixModel(BaseHandel):
    def __init__(self, Model_Name, Parameter_Names, Parameter_Ranges, config):
        self.Model_Name = Model_Name
        self.Parameter_Names = Parameter_Names
        self.Parameter_Ranges = Parameter_Ranges
        self.config = config

    def Sample(self, points):
        pass

    def Truth(self, points):
        pass

    def TrueSeparatrix(self, interestLevel):
        pass
