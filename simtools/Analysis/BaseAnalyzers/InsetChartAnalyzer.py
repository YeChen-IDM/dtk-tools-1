from collections import Iterable
from simtools.Analysis.BaseAnalyzers import BaseAnalyzer


class InsetChartAnalyzer(BaseAnalyzer):

    def __init__(self, channels=None, **kwargs):
        super().__init__(**kwargs)
        assert(isinstance(channels, Iterable) and len(channels) > 0)
        self.channels = channels
        self.filenames.append("output\\InsetChart.json")

    def select_simulation_data(self, data, simulation):
        return {c:data[self.filenames[0]]["Channels"][c] for c in self.channels}

