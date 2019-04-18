import logging
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from calibtool.IterationState import IterationState
from calibtool.plotters.BasePlotter import BasePlotter
from calibtool.utils import StatusPoint


logger = logging.getLogger(__name__)


class ModelPlotter(BasePlotter):
    def __init__(self, combine_sites=True):
        super(ModelPlotter, self).__init__(combine_sites)

    @property
    def directory(self):
        return self.get_plot_directory()

    #ZD [TODO]: self.iteration_state.analyzer_list doesn't keep site info, here we assume all analyzers have different names!!!
    def get_site_analyzer(self, site_name, analyzer_name):
        for site, analyzers in self.site_analyzer_names.items():
            if site_name != site:
                continue
            for analyzer in self.iteration_state.analyzer_list:
                if analyzer_name == analyzer.name:
                    return analyzer
        raise Exception('Unable to find analyzer=%s for site=%s' % (analyzer_name, site_name))

    def get_analyzer_data(self, iteration, site_name, analyzer_name):
        site_analyzer = '%s_%s' % (site_name, analyzer_name)
        return IterationState.restore_state(self.iteration_state.calibration_name, iteration).analyzers[site_analyzer]

    def visualize(self, iteration_state):
        self.iteration_state = iteration_state
        self.site_analyzer_names = iteration_state.site_analyzer_names
        iteration_status = self.iteration_state.status
        if iteration_status != StatusPoint.plot:
            return  # Only plot once results are available
        try:
            if self.combine_sites:
                for site_name, analyzer_names in self.site_analyzer_names.items():
                    self.plot_analyzers(site_name, analyzer_names, self.all_results)
            else:
                for site_name, analyzer_names in self.site_analyzer_names.items():
                    self.plot_analyzers(site_name, analyzer_names, self.all_results)
        except:
            logger.info("ModelPlotter could not plot for one or more analyzer(s).")

    def plot_analyzers(self, site_name, analyzer_names, samples):
        for analyzer_name in analyzer_names:
            site_analyzer = '%s_%s' % (site_name, analyzer_name)

    def cleanup(self):
        """
        cleanup the existing plots
        :param calib_manager:
        :return:
        """
        pass
