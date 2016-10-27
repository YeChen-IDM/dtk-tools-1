import logging
from sys import exit# as exit
import os

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.spatial.distance import seuclidean
from scipy.stats import uniform, norm

import statsmodels.api as sm
import matplotlib.pyplot as plt

from calibtool.NextPointAlgorithm import NextPointAlgorithm

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimTool(NextPointAlgorithm):
    '''
    OptimTool

    The basic idea of OptimTool is
    '''

    def __init__(self, params,
                mu_r = 0.1,
                sigma_r = 0.02,
                center_repeats = 2,
                initial_samples = 1e2, # Should be number of samples to draw from the hypersphere
                samples_per_iteration = 1e2,
                rsquared_thresh = 0.5 # Above this value, the ascent direction is used.  Below, it's best result.
            ):

        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.center_repeats = center_repeats
        self.rsquared_thresh = rsquared_thresh

        self.n_dimensions = len(params)
        self.params = params
        assert( self.n_dimensions == len(self.get_param_names()) )

        self.samples_per_iteration = int(samples_per_iteration)

        self.data = pd.DataFrame(columns=[['Iteration', 'Sample', 'Results', 'Fitted'] + self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['Sample'] = self.data['Sample'].astype(int)

        self.meta = pd.DataFrame(columns=['Iteration', 'Parameter', 'Value'])   # Parameters: Rsquared, Regression_Parameters
        self.meta['Iteration'] = self.meta['Iteration'].astype(int)

        #self.X_center = pd.DataFrame([X_guess], columns=self.get_param_names())   # Parameters: Rsquared, Regression_Parameters
        #self.X_center['Iteration'] = 0
        iteration = 0

        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Value', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)
        for param_name in self.get_param_names():
            self.state.loc[len(self.state)] = [iteration, param_name, params[param_name]['Guess'], params[param_name]['Min'], params[param_name]['Max']]

        initial_samples = self.sample_hypersphere(initial_samples, self._get_X_center(iteration))

        self.add_samples( initial_samples, iteration=0 )

        logger.info('%s instance with %d initial %d-dimensional samples and %d per iteration',
                    self.__class__.__name__, self.data.shape[0],
                    self.n_dimensions, self.samples_per_iteration)

    def _get_X_center(self, iteration):
        state_by_iter = self.state.reset_index(drop=True).set_index(['Iteration', 'Parameter'])
        assert( iteration in state_by_iter.index.get_level_values('Iteration') )
        state_this_iter = state_by_iter.loc[iteration]
        return [ state_this_iter.loc[p]['Value'] for p in self.get_param_names() ]


    def add_samples(self, samples, iteration):
        samples_df = pd.DataFrame( samples, columns = self.get_param_names() )
        samples_df.index.name = 'Sample'
        samples_df['Iteration'] = iteration
        samples_df.reset_index(inplace=True)

        self.data = pd.concat([self.data, samples_df])

        logger.debug('Samples:\n%s' % samples_df[self.get_param_names()].values)

    def get_samples_for_iteration(self, iteration):
        return self.data.query('Iteration == @ iteration').sort_values('Sample')[self.get_param_names()].values

    def choose_samples_for_next_iteration(self, iteration, results):
        logger.info('%s: Choosing samples at iteration %d:', self.__class__.__name__, iteration)
        logger.debug('Results:\n%s', results)

        data_by_iter = self.data.set_index('Iteration')
        if iteration+1 in data_by_iter.index.unique():
            # Perhaps the results have changed? Check?
            # TODO: Change to logger (everywhere)
            print '%s: Have already selected samples for the next interation, why call choose_samples_for_next_iteration?' % self.__class__.__name__
            print data_by_iter.loc[iteration+1]
            return data_by_iter.loc[iteration+1, self.get_param_names()].values

        # TODO: Need to sort by sample?
        self.data = self.data.set_index('Iteration')
        self.data.loc[iteration,'Results'] = results

        latest_samples = self.data.loc[iteration, self.get_param_names()].values
        latest_results = self.data.loc[iteration, 'Results'].values

        print 'ITERATION:', iteration
        print 'LATEST SAMPLES:', latest_samples
        print 'LATEST RESULTS:', latest_results

        mod = sm.OLS(latest_results, sm.add_constant(latest_samples) )
        mod_fit = mod.fit()
        print mod_fit.summary()
        self.fit_summary = mod_fit.summary().as_csv()

        self.data.loc[iteration, 'Fitted'] = mod_fit.fittedvalues

        # TODO: Cleaner:
        r2_df = pd.DataFrame( [[iteration, 'Rsquared', mod_fit.rsquared]], columns=['Iteration', 'Parameter', 'Value'] )
        self.meta = pd.concat([self.meta, r2_df])
        for (p,v) in zip( ['Constant'] + self.get_param_names(), mod_fit.params): # mod.endog_names
            regression_param_df = pd.DataFrame( [[iteration, p, v]], columns=['Iteration', 'Parameter', 'Value'] )
            self.meta = pd.concat([self.meta, regression_param_df])

        # Choose next X_center
        if mod_fit.rsquared > self.rsquared_thresh:
            print 'Good R^2 (%f), using params: '%mod_fit.rsquared, mod_fit.params
            params = mod_fit.params[1:] # Drop constant
            den = np.sqrt( sum([ (v['Max']-v['Min'])**2 * p**2  for p,v in zip(params, self.params.values()) ]) )

            old_center = self._get_X_center(iteration)
            new_center = [c + (v['Max']-v['Min'])**2 * p * self.mu_r / den for c,p,v in zip(old_center, params, self.params.values()) ]

            print "TODO: MAKE SURE NEW X_CENTER IS WITHIN CONSTRAINTS"
        else:
            print 'Bad R^2 (%f)'%mod_fit.rsquared
            max_idx = np.argmax(latest_results)
            'Stepping to argmax of %f at:'%latest_results[max_idx], latest_samples[max_idx]
            new_center = latest_samples[max_idx]

        new_state = pd.DataFrame( {
            'Iteration':[iteration+1]*self.n_dimensions,
            'Parameter': self.get_param_names(),
            'Value': new_center.tolist(),
            'Min': [v['Min'] for v in self.params.values()],
            'Max': [v['Max'] for v in self.params.values()]
        } )

        self.state = pd.concat( [self.state, new_state] )
        samples_for_next_iteration = self.sample_hypersphere(self.samples_per_iteration, new_center)

        self.data.reset_index(inplace=True)
        self.add_samples( samples_for_next_iteration, iteration + 1 )

        # Validate?
        logger.debug('Next samples:\n%s', samples_for_next_iteration)
        print 'UPDATED SAMPLES:\n', samples_for_next_iteration

        return samples_for_next_iteration


    def sample_hypersphere(self, N, X_center):
        # Pick samples on hypersphere
        sn = norm(loc=0, scale=1)

        assert(N > self.center_repeats)

        deviation = []
        for i in range(N - self.center_repeats):
            rvs = sn.rvs(size = self.n_dimensions)
            nrm = np.linalg.norm(rvs)

            deviation.append( [r/nrm for r in rvs] )

        rad = norm(loc = self.mu_r, scale = self.sigma_r)

        samples = np.empty([N, self.n_dimensions])
        samples[:self.center_repeats] = X_center
        for i, dev in enumerate(deviation):
            r = rad.rvs()
            # Scale by param range
            samples[self.center_repeats + i] = [x + r * p * (v['Max']-v['Min']) for x,p,v in zip(X_center, dev, self.params.values())]

        return samples

    def end_condition(self):
        print "end_condition"
        # Stopping Criterion: good rsqared with small norm?
        # Return True to stop, False to continue
        logger.info('Continuing iterations ...')
        return False

    def get_final_samples(self):
        print "get_final_samples"
        '''
        Resample Stage:
        '''
        state_by_iteration = self.state.set_index('Iteration')
        last_iter = sorted(state_by_iteration.index.unique())[-1]

        #return dict( samples = self.X_center[-1] )
        return self._get_X_center(last_iter)

    def get_state(self) :
        optimtool_state = dict(
            mu_r = self.mu_r,
            sigma_r = self.sigma_r,
            center_repeats = self.center_repeats,
            rsquared_thresh = self.rsquared_thresh,
            n_dimensions = self.n_dimensions,
            params = self.params,
            samples_per_iteration = self.samples_per_iteration,

            data = self.data.to_dict(orient='list'),
            meta = self.meta.to_dict(orient='list'),
            state = self.state.to_dict(orient='list')
        )
        return optimtool_state

    def set_state(self, state):
        self.mu_r = state['mu_r']
        self.sigma_r = state['sigma_r']
        self.center_repeats = state['center_repeats']
        self.rsquared_thresh = state['rsquared_thresh']
        self.n_dimensions = state['n_dimensions']
        self.params = state['params']
        self.samples_per_iteration = state['samples_per_iteration']

        self.data = pd.DataFrame.from_dict(state['data'], orient='columns')
        self.meta = pd.DataFrame.from_dict(state['meta'], orient='columns')
        self.state = pd.DataFrame.from_dict(state['state'], orient='columns')


    def get_param_names(self):
        return self.params.keys()
