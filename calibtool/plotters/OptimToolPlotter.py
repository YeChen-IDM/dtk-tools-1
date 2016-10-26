import logging
import os

import numpy as np
import gc   # TEMP?

import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

    def plot_state_evolution(self, **kwargs):
        print 'K', kwargs
        data = kwargs.pop('data')
        print 'D', data
        plt.plot(data['Iteration'], data['Value'], color='k', marker='o')
        plt.plot(data['Iteration'], data['Min'], color='r')
        plt.plot(data['Iteration'], data['Max'], color='r')

    def visualize(self, calib_manager):
        print 'OptimToolPlotter: VISUALIZE'
        self.all_results = calib_manager.all_results
        logger.debug(self.all_results)

        self.directory = calib_manager.iteration_directory()
        self.param_names = calib_manager.param_names()
        self.site_analyzer_names = calib_manager.site_analyzer_names()

        X_center_all = calib_manager.iteration_state.next_point['X_center']
        X_center = X_center_all[ calib_manager.iteration ]

        ### STATE EVOLUTION ###
        params = calib_manager.iteration_state.next_point['params']
        state_evolution = pd.DataFrame(columns = ['Parameter', 'Iteration', 'Value', 'Min', 'Max'])
        state_evolution['Iteration'] = state_evolution['Iteration'].astype(int)
        for (i,p) in enumerate(self.param_names):
            xc = [x[i] for x in X_center_all]
            sd = pd.DataFrame( xc, columns=['Value'] )
            sd.index.name = 'Iteration'
            sd['Parameter'] = p
            sd['Min'] = params[p]['Min']
            sd['Max'] = params[p]['Max']
            state_evolution = pd.concat([state_evolution, sd.reset_index()])

        g = sns.FacetGrid(state_evolution, row=None, col='Parameter', hue=None, col_wrap=2, sharex=False, sharey=False, size=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=True, legend_out=True, despine=True, margin_titles=True, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None)
        g.map_dataframe(self.plot_state_evolution).set_titles('{col_name}')
        plt.savefig( os.path.join(self.directory, 'Optimization_State_Evolution.pdf'))


        results = calib_manager.all_results
        results_by_iteration = results.reset_index().set_index('iteration')

        if calib_manager.iteration not in results_by_iteration.index.unique():
            print 'No results for iteration %d yet, skipping some plots in OptimToolPlotter.' % calib_manager.iteration
            return

        results_this_iteration = results_by_iteration.loc[calib_manager.iteration].sort_values('sample')
        latest_results = results_this_iteration['total'].values
        latest_samples = results_this_iteration[self.param_names].values
        D = latest_samples.shape[1]

        fitted_values_df = pd.DataFrame.from_dict(calib_manager.iteration_state.next_point['fitted_values_dict'])

        merged = pd.DataFrame.merge( calib_manager.all_results.reset_index(), fitted_values_df.reset_index()[['sample', 'iteration', 'Fitted']], on=['sample', 'iteration'])

        #data = merged.set_index('iteration').loc[calib_manager.iteration]
        data = merged.set_index('iteration')
        data = data.loc[calib_manager.iteration]

        rsquared_all = calib_manager.iteration_state.next_point['rsquared']
        rsquared = rsquared_all[ calib_manager.iteration ]

        regression_parameters = calib_manager.iteration_state.next_point['regression_parameters']

        ### REGRESSION ###
        fig, ax = plt.subplots()
        h1 = plt.plot( data['total'], data['Fitted'], 'o', figure=fig)
        h2 = plt.plot( [min(latest_results), max(latest_results)], [min(latest_results), max(latest_results)], 'r-')
        plt.xlabel('Simulation Output')
        plt.ylabel('Linear Regression')
        plt.title( rsquared )
        plt.savefig( os.path.join(self.directory, 'Optimization_Regression.pdf'))

        fig.clf()
        plt.close(fig)

        del h1, h2, ax, fig

        ### STATE ###
        if D == 1:
            data_sorted = data.sort_values(self.param_names)
            sorted_samples = data_sorted[self.param_names]
            sorted_results = data_sorted['total']
            sorted_fitted = data_sorted['Fitted']

            fig, ax = plt.subplots()
            h1 = plt.plot( sorted_samples, sorted_results, 'ko', figure=fig)
            yl = ax.get_ylim()
            h2 = plt.plot( 2*[calib_manager.next_point.X_center[calib_manager.iteration]], yl, 'b-', figure=fig)

            h3 = plt.plot( sorted_samples, sorted_fitted, 'r-', figure=fig)

            plt.title( rsquared )
            plt.savefig( os.path.join(self.directory, 'Optimization_Sample_Results.pdf'))

            fig.clf()
            plt.close(fig)

            del h1, h2, h3, ax, fig

        elif D == 2:
            x0 = data[self.param_names[0]]
            x1 = data[self.param_names[1]]
            y = data['total']
            y_fit = data['Fitted']
            rp = regression_parameters[calib_manager.iteration]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            h1 = ax.scatter( x0, x1, y, c='k', marker='o', figure=fig)
            i=int(calib_manager.iteration)
            h2 = ax.scatter( X_center_all[i][0], 
                        X_center_all[i][1], 
                        rp[0] + rp[1]*X_center_all[i][0] + rp[2]*X_center_all[i][1],
                        c='b', marker='.', s=200, figure=fig)

            h3 = ax.plot(    [xc[0] for xc in X_center_all[i:i+2]],
                        [xc[1] for xc in X_center_all[i:i+2]], 
                        [
                            rp[0] + rp[1]*X_center_all[i][0] + rp[2]*X_center_all[i][1],
                            rp[0] + rp[1]*X_center_all[i+1][0] + rp[2]*X_center_all[i+1][1]
                        ], c='b', figure=fig)


            h4 = ax.scatter( x0, x1, y_fit, c='r', marker='d', figure=fig)

            xl = ax.get_xlim()
            yl = ax.get_ylim()

            x_surf=np.linspace(xl[0], xl[1], 25)                # generate a mesh
            y_surf=np.linspace(yl[0], yl[1], 25)
            x_surf, y_surf = np.meshgrid(x_surf, y_surf)
            z_surf = rp[0] + rp[1]*x_surf + rp[2]*y_surf
            h5 = ax.plot_surface(x_surf, y_surf, z_surf, cmap=cm.hot, rstride=1, cstride=1,
                linewidth=0, antialiased=True, edgecolor=(0,0,0,0), alpha=0.5)

            h6 = plt.contour(x_surf, y_surf, z_surf, 10,
                  #[-1, -0.1, 0, 0.1],
                  alpha=0.5,
                  cmap=plt.cm.bone)


            plt.title( rsquared )
            plt.savefig( os.path.join(self.directory, 'Optimization_Sample_Results.pdf'))

            fig.clf()
            plt.close(fig)

            del h1, h2, h3, h4, h5, h6, ax, fig



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
        del g, ax, fig

        gc.collect()

        print 'OptimToolPlotter: [ DONE ]'


    def cleanup_plot(self, calib_manager):
        print 'CLEANUP_PLOT?'
