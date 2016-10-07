import logging
import os

import matplotlib.pyplot as plt

from calibtool.plotters.BasePlotter import BasePlotter
from calibtool.visualize import combine_by_site

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set_style('white')
except:
    pass

class OptimToolPlotter(BasePlotter):
    def __init__(self):
        super(OptimToolPlotter, self).__init__( False )

    def visualize(self, calib_manager):
        self.all_results = calib_manager.all_results
        logger.debug(self.all_results)

        self.directory = calib_manager.iteration_directory()
        self.param_names = calib_manager.param_names()
        self.site_analyzer_names = calib_manager.site_analyzer_names()

        latest_results = calib_manager.next_point.latest_results
        res = calib_manager.next_point.res

        plt.plot( latest_results, res.fittedvalues, 'o')
        plt.plot( [min(latest_results), max(latest_results)], [min(latest_results), max(latest_results)], 'r-')
        plt.title( res.rsquared )
        plt.savefig( os.path.join(self.directory, 'Regression.pdf') )
        plt.close()


    def cleanup_plot(self, calib_manager):
        print 'CLEANUP_PLOT?'
