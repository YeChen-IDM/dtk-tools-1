import os
import sys
sys.path.append(os.path.dirname(__file__))

from simtools.Analysis.AnalyzeManager import AnalyzeManager
from MyAnalyzer import PopulationAnalyzer

if __name__ == "__main__":
    analyzers = [PopulationAnalyzer()]
    am = AnalyzeManager(exp_list=["39953ccf-e899-e811-a2c0-c4346bcb7275"], analyzers=analyzers)
    am.analyze()