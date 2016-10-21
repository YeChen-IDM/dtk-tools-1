import logging
import os

import gc   # TEMP

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
        print calib_manager.all_results
        print calib_manager.all_results.reset_index()
        print calib_manager.all_results.shape

        print 'CM.ITER', calib_manager.iteration
        print 'IS.NP', calib_manager.iteration_state.next_point
        exit()

        if calib_manager.iteration_state.next_point.fitted_values:
            print len(calib_manager.next_point.fitted_values)
        exit()

        if calib_manager.iteration not in results_by_iteration.index.unique():
            return
        #results_by_iteration = calib_manager.all_results.reset_index().set_index(['iteration'])

        if calib_manager.next_point.fitted_values and calib_manager.next_point.rsquared:
            fitted_values = calib_manager.next_point.fitted_values[calib_manager.iteration]
            fitted_values_df = pd.DataFrame(fitted_values)
            fitted_values_df.index.name = 'sample'

        latest_results = results_by_iteration.loc[calib_manager.iteration, 'total'].values

        print calib_manager.next_point.fitted_values
        print calib_manager.next_point.rsquared
        print calib_manager.iteration


### REGRESSION ###
        if calib_manager.next_point.fitted_values and calib_manager.next_point.rsquared:
            fitted_values = calib_manager.next_point.fitted_values[calib_manager.iteration]
            rsquared = calib_manager.next_point.rsquared[calib_manager.iteration]

            fig, ax = plt.subplots()
            plt.plot( latest_results, fitted_values, 'o', figure=fig)
            plt.plot( [min(latest_results), max(latest_results)], [min(latest_results), max(latest_results)], 'r-')
            plt.title( rsquared )
            plt.savefig( os.path.join(self.directory, 'Optimization_Regression.pdf'))

            fig.clf()
            plt.close(fig)

            del fig, ax

### STATE ###
        param_names = calib_manager.param_names()

        results_this_iter = results_by_iteration.loc[calib_manager.iteration]
        latest_samples = results_by_iteration.loc[calib_manager.iteration, param_names].values

        fig, ax = plt.subplots()
        plt.plot( latest_samples, latest_results, 'ko', figure=fig)
        yl = ax.get_ylim()
        plt.plot( [calib_manager.next_point.X_center[calib_manager.iteration], calib_manager.next_point.X_center[calib_manager.iteration]], yl, 'b-', figure=fig)
        if calib_manager.next_point.fitted_values and calib_manager.next_point.rsquared:
            results_this_iter['Fitted Values'] = calib_manager.next_point.fitted_values[calib_manager.iteration]
            print 'PN', param_names
            print 'RTI:BEFORE',results_this_iter
            results_this_iter.sort_values(by=param_names, inplace=True)
            print 'RTI:AFTER',results_this_iter
            print zip(results_this_iter[param_names], results_this_iter['Fitted Values'])
            plt.plot( results_this_iter[param_names], results_this_iter['Fitted Values'], 'r-', figure=fig)

            rsquared = calib_manager.next_point.rsquared[calib_manager.iteration]
            plt.title( rsquared )
        plt.savefig( os.path.join(self.directory, 'Optimization_Sample_Results.pdf'))

        fig.clf()
        plt.close(fig)

        del fig, ax






### BY ITERATION ###

        all_results = calib_manager.all_results.copy().reset_index(drop=True)#.set_index(['iteration', 'sample'])
        fig, ax = plt.subplots()
        g = sns.violinplot(x='iteration', y='total', data=all_results, ax = ax)
#, hue=None, data=res, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs))
        # sample is index
        # cols of iteration, total, Dielmo_ClinicalIncidenceByAgeCohortAnalyzer, and x labels
        plt.savefig( os.path.join(self.directory, 'Optimization_Progress.pdf'))


        fig.clf()
        plt.close(fig)
        del g, fig, ax



        


        gc.collect()

        print 'OptimToolPlotter: [ DONE ]'


    def cleanup_plot(self, calib_manager):
        print 'CLEANUP_PLOT?'
