import logging
import os

import gc   # TEMP

import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
        print 'IS.NP.FV', calib_manager.iteration_state.next_point['fitted_values_dict']

        param_names = calib_manager.param_names()
        results = calib_manager.all_results
        print 'RESULTS:\n', results
        results_this_iteration = results.reset_index().set_index('iteration').loc[calib_manager.iteration].sort_values('sample')
        latest_results = results_this_iteration['total'].values
        latest_samples = results_this_iteration[param_names].values
        D = latest_samples.shape[1]

        fitted_values_df = pd.DataFrame.from_dict(calib_manager.iteration_state.next_point['fitted_values_dict'])

        print 'MERGE'*10
        print 'ALL:', calib_manager.all_results.reset_index()
        print 'FV:', fitted_values_df.reset_index()
        merged = pd.DataFrame.merge( calib_manager.all_results.reset_index(), fitted_values_df.reset_index()[['sample', 'iteration', 'Fitted']], on=['sample', 'iteration'])

        #data = merged.set_index('iteration').loc[calib_manager.iteration]
        data = merged.set_index('iteration')
        print 'DATA INDEX:', data.index.unique()
        print 'CM ITERATION:', calib_manager.iteration
        data = data.loc[calib_manager.iteration]
        print data

        rsquared_all = calib_manager.iteration_state.next_point['rsquared']
        rsquared = rsquared_all[ calib_manager.iteration ]

        X_center_all = calib_manager.iteration_state.next_point['X_center']
        print 'X_center_all', X_center_all
        X_center = X_center_all[ calib_manager.iteration ]
        print 'X_center', X_center

        ### REGRESSION ###
        fig, ax = plt.subplots()
        plt.plot( data['total'], data['Fitted'], 'o', figure=fig)
        plt.plot( [min(latest_results), max(latest_results)], [min(latest_results), max(latest_results)], 'r-')
        plt.xlabel('Simulation Output')
        plt.ylabel('Linear Regression')
        plt.title( rsquared )
        plt.savefig( os.path.join(self.directory, 'Optimization_Regression.pdf'))

        fig.clf()
        plt.close(fig)

        del fig, ax

### STATE ###
        if D == 1:
            data_sorted = data.sort_values(param_names)
            sorted_samples = data_sorted[param_names]
            sorted_results = data_sorted['total']
            sorted_fitted = data_sorted['Fitted']

            fig, ax = plt.subplots()
            plt.plot( sorted_samples, sorted_results, 'ko', figure=fig)
            yl = ax.get_ylim()
            plt.plot( 2*[calib_manager.next_point.X_center[calib_manager.iteration]], yl, 'b-', figure=fig)

            plt.plot( sorted_samples, sorted_fitted, 'r-', figure=fig)

            plt.title( rsquared )
            plt.savefig( os.path.join(self.directory, 'Optimization_Sample_Results.pdf'))

            fig.clf()
            plt.close(fig)

            del fig, ax

        elif D == 2:
            x0 = data[param_names[0]]
            x1 = data[param_names[1]]
            y = data['total']
            y_fit = data['Fitted']

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter( x0, x1, y, c='k', marker='o', figure=fig)
            #ax.scatter( X_center[0], X_center[1], y[0], c='b', marker='x', figure=fig)

            ax.scatter( x0, x1, y_fit, c='r', marker='d', figure=fig)

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
