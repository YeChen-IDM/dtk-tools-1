import logging
import os

import seaborn as sns
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
        print 'OptimToolPlotter: INIT'
        super(OptimToolPlotter, self).__init__( False )

    def visualize(self, calib_manager):
        print 'OptimToolPlotter: VISUALIZE'
        self.all_results = calib_manager.all_results
        logger.debug(self.all_results)

        self.directory = calib_manager.iteration_directory()
        self.param_names = calib_manager.param_names()
        self.site_analyzer_names = calib_manager.site_analyzer_names()

        #latest_results = calib_manager.next_point.latest_results
        latest_results = calib_manager.all_results.reset_index(drop=True).set_index(['iteration'])

        if calib_manager.iteration not in latest_results.index.unique():
            return

        latest_results = latest_results.loc[calib_manager.iteration, 'total'].values

        print calib_manager.next_point.fitted_values
        print calib_manager.next_point.rsquared
        print calib_manager.iteration


### REGRESSION ###
        if calib_manager.next_point.fitted_values and calib_manager.next_point.rsquared:
            fitted_values = calib_manager.next_point.fitted_values[calib_manager.iteration]
            rsquared = calib_manager.next_point.rsquared[calib_manager.iteration]

            fig, ax = plt.subplots()
            plt.plot( latest_results, fitted_values, 'o')
            plt.plot( [min(latest_results), max(latest_results)], [min(latest_results), max(latest_results)], 'r-')
            plt.title( rsquared )
            plt.savefig( os.path.join(self.directory, 'Optimization_Regression.pdf'))
            plt.close()

### BY ITERATION ###

        fig, ax = plt.subplots()
        all_results = calib_manager.all_results.copy().reset_index(drop=True)#.set_index(['iteration', 'sample'])
        sns.violinplot(x='iteration', y='total', data=all_results, ax = ax)
#, hue=None, data=res, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs))
        # sample is index
        # cols of iteration, total, Dielmo_ClinicalIncidenceByAgeCohortAnalyzer, and x labels
        plt.savefig( os.path.join(self.directory, 'Optimization_Progress.pdf'))

        print 'OptimToolPlotter: [ DONE ]'


    def cleanup_plot(self, calib_manager):
        print 'CLEANUP_PLOT?'
