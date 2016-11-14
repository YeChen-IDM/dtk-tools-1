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

    def __init__(self, params, constrain_sample_fn,
                mu_r = 0.1,
                sigma_r = 0.02,
                center_repeats = 2,
                samples_per_iteration = 1e2,
                rsquared_thresh = 0.5 # Above this value, the ascent direction is used.  Below, it's best result.
            ):

        self.params = params # TODO: Check min <= center <= max
        self.constrain_sample_fn = constrain_sample_fn

        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.center_repeats = center_repeats
        self.rsquared_thresh = rsquared_thresh

        self.n_dimensions = len(params)

        assert( self.n_dimensions == len(self.get_param_names()) )

        self.samples_per_iteration = int(samples_per_iteration)

        self.data = pd.DataFrame(columns=[['Iteration', '__sample_index__', 'Results', 'Fitted'] + self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

        self.regression = pd.DataFrame(columns=['Iteration', 'Parameter', 'Value'])   # Parameters: Rsquared, Regression_Parameters
        self.regression['Iteration'] = self.regression['Iteration'].astype(int)

        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Center', 'Min', 'Max'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

        iteration = 0
        for param in self.params:
            print (iteration, param['Name'], param['Guess'], param['Min'], param['Max'])
            self.state.loc[len(self.state)] = [iteration, param['Name'], param['Guess'], param['Min'], param['Max']]

        X_center = self._get_X_center(iteration)
        initial_samples = self.choose_and_clamp_hypersphere_samples_for_iteration(iteration, samples_per_iteration, X_center)

        logger.info('%s instance with %d initial %d-dimensional samples and %d per iteration',
                    self.__class__.__name__, self.data.shape[0],
                    self.n_dimensions, self.samples_per_iteration)


    def _get_X_center(self, iteration):
        state_by_iter = self.state.reset_index(drop=True).set_index(['Iteration', 'Parameter'])
        assert( iteration in state_by_iter.index.get_level_values('Iteration') )
        state_this_iter = state_by_iter.loc[iteration]
        return [ state_this_iter.loc[p]['Center'] for p in self.get_param_names() ]


    def add_samples(self, samples, iteration):
        samples_df = pd.DataFrame( samples, columns = self.get_param_names() )
        samples_df.index.name = '__sample_index__'
        samples_df['Iteration'] = iteration
        samples_df.reset_index(inplace=True)

        self.data = pd.concat([self.data, samples_df])

        logger.debug('__sample_index__:\n%s' % samples_df[self.get_param_names()].values)


    def get_samples_for_iteration(self, iteration):
        return self.data.query('Iteration == @iteration').sort_values('__sample_index__')[self.get_param_names()] # Query makes a copy


    def clamp(self, X, Xmin, Xmax):
        # UGLY, but functional: TODO cleanup!
        Xa = np.asmatrix(X) if not isinstance(X, np.ndarray) else X

        Xa = np.asmatrix(X)
        nRows = Xa.shape[0]
        for (i,p) in enumerate( self.get_param_names() ):
            Xa[:,i] = np.minimum( Xmax[p], np.maximum( Xmin[p], Xa[:,i] ) )

        if isinstance(X, list):
            Xout = Xa.tolist()
            if nRows == 1:
                Xout = Xout[0]
        else:
            Xout = Xa

        return Xout


    def choose_samples_for_next_iteration(self, iteration, results):
        logger.info('%s: Choosing samples at iteration %d:', self.__class__.__name__, iteration)
        logger.debug('Results:\n%s', results)

        data_by_iter = self.data.set_index('Iteration')
        if iteration+1 in data_by_iter.index.unique():
            # Perhaps the results have changed? Check?
            # TODO: Change to logger (everywhere)
            print "%s: I'm way ahead of you, samples for the next iteration were computed previously." % self.__class__.__name__
            return data_by_iter.loc[iteration+1, self.get_param_names()].values

        # TODO: Need to sort by sample?
        self.data = self.data.set_index('Iteration')
        print results
        print len(results)
        print self.data.tail()
        print iteration
        print self.data.loc[iteration]

        print 'RESULTS:\n', results
        print 'DATA.LOC:\n', self.data.loc[iteration,'Results']
        print 'ITERATION:', iteration
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

        # Choose next X_center
        if mod_fit.rsquared > self.rsquared_thresh:
            print 'Good R^2 (%f), using params: '%mod_fit.rsquared, mod_fit.params
            coef = mod_fit.params[1:] # Drop constant
            den = np.sqrt( sum([ (p['Max']-p['Min'])**2 * c**2  for c,p in zip(coef, self.params) ]) )

            old_center = self._get_X_center(iteration)
            new_center = [x + (p['Max']-p['Min'])**2 * c * self.mu_r / den for x,c,p in zip(old_center, coef, self.params) ]

        else:
            print 'Bad R^2 (%f)'%mod_fit.rsquared
            max_idx = np.argmax(latest_results)
            'Stepping to argmax of %f at:'%latest_results[max_idx], latest_samples[max_idx]
            new_center = latest_samples[max_idx].tolist()

        r2_df = pd.DataFrame( [[iteration, 'Rsquared', mod_fit.rsquared]], columns=['Iteration', 'Parameter', 'Value'] )
        thresh_df = pd.DataFrame( [[iteration, 'Rsquared_Threshold', self.rsquared_thresh]], columns=['Iteration', 'Parameter', 'Value'] )
        self.regression = pd.concat([self.regression, r2_df, thresh_df])
        for (p,v) in zip( ['Constant'] + self.get_param_names(), mod_fit.params): # mod.endog_names
            regression_param_df = pd.DataFrame( [[iteration, p, v]], columns=['Iteration', 'Parameter', 'Value'] )
            self.regression = pd.concat([self.regression, regression_param_df])

        # CLAMP
        X_min = self.state.pivot('Iteration', 'Parameter', 'Min').loc[iteration, self.get_param_names()]
        X_max = self.state.pivot('Iteration', 'Parameter', 'Max').loc[iteration, self.get_param_names()]
        new_center = self.clamp(new_center, X_min, X_max)

        # USER CONSTRAINT FN
        new_center_dict = self.constrain_sample_fn( {k:v for k,v in zip(self.get_param_names(), new_center)} )
        new_center = np.array( [new_center_dict[p] for p in self.get_param_names()] )

        new_state = pd.DataFrame( {
            'Iteration':[iteration+1]*self.n_dimensions,
            'Parameter': self.get_param_names(),
            'Center': new_center,
            'Min': [p['Min'] for p in self.params],
            'Max': [p['Max'] for p in self.params]
        } )

        self.state = pd.concat( [self.state, new_state] )

        samples_for_next_iteration = self.choose_and_clamp_hypersphere_samples_for_iteration(iteration+1, self.samples_per_iteration, new_center)

        logger.debug('Next samples:\n%s', samples_for_next_iteration)
        print 'UPDATED SAMPLES:\n', samples_for_next_iteration

        return samples_for_next_iteration


    def choose_and_clamp_hypersphere_samples_for_iteration(self, iteration, N, X_center):

        samples = self.sample_hypersphere(N, X_center)

        # CLAMP
        X_min = self.state.pivot('Iteration', 'Parameter', 'Min').loc[iteration, self.get_param_names()]
        X_max = self.state.pivot('Iteration', 'Parameter', 'Max').loc[iteration, self.get_param_names()]
        samples = self.clamp(samples, X_min, X_max)

        self.data.reset_index(inplace=True)
        self.add_samples( samples, iteration )

        # USER CONSTRAINT FN
        self.data = self.data.set_index('Iteration')
        tmp = self.data.loc[iteration].apply( self.constrain_sample_fn, axis=1)

        self.data.loc[iteration] = self.data.loc[iteration].apply( self.constrain_sample_fn, axis=1)
        self.data.reset_index(inplace=True)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

        data_by_iter = self.data.reset_index().set_index('Iteration')
        return data_by_iter.loc[iteration]


    def sample_hypersphere(self, N, X_center):
        # Pick samples on hypersphere
        sn = norm(loc=0, scale=1)

        assert(N > self.center_repeats)

        deviations = []
        for i in range(N - self.center_repeats):
            rvs = sn.rvs(size = self.n_dimensions)
            nrm = np.linalg.norm(rvs)

            deviations.append( [r/nrm for r in rvs] )

        rad = norm(loc = self.mu_r, scale = self.sigma_r)

        samples = np.empty([N, self.n_dimensions])
        samples[:self.center_repeats] = X_center
        for i, deviation in enumerate(deviations):
            r = rad.rvs()
            # Scale by param range
            samples[self.center_repeats + i] = [x + r * d * (param['Max']-param['Min']) for x, d, param in zip(X_center, deviation, self.params)]

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

    def prep_for_dict(self, df):
        # Needed for Windows compatibility
        #nulls = df.isnull()
        #if nulls.values.any():
        #    df[nulls] = None
        #return df.to_dict(orient='list')

        return df.where(~df.isnull(), other=None).to_dict(orient='list')

    def get_state(self):

        print 'OTDATA:\n', self.data

        optimtool_state = dict(
            mu_r = self.mu_r,
            sigma_r = self.sigma_r,
            center_repeats = self.center_repeats,
            rsquared_thresh = self.rsquared_thresh,
            n_dimensions = self.n_dimensions,
            params = self.params,
            samples_per_iteration = self.samples_per_iteration,

            data = self.prep_for_dict(self.data),
            data_dtypes = {name:str(data.dtype) for name, data in self.data.iteritems()},

            regression = self.prep_for_dict(self.regression),
            regression_dtypes = {name:str(data.dtype) for name, data in self.regression.iteritems()},

            state = self.prep_for_dict(self.state),
            state_dtypes = {name:str(data.dtype) for name, data in self.state.iteritems()}
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

        data_dtypes =state['data_dtypes']
        self.data = pd.DataFrame.from_dict(state['data'], orient='columns')
        for c in self.data.columns: # Argh
            self.data[c] = self.data[c].astype( data_dtypes[c] )

        regression_dtypes =state['regression_dtypes']
        self.regression = pd.DataFrame.from_dict(state['regression'], orient='columns')
        for c in self.regression.columns: # Argh
            self.regression[c] = self.regression[c].astype( regression_dtypes[c] )

        state_dtypes =state['state_dtypes']
        self.state = pd.DataFrame.from_dict(state['state'], orient='columns')
        for c in self.state.columns: # Argh
            self.state[c] = self.state[c].astype( state_dtypes[c] )


    def get_param_names(self):
        return [p['Name'] for p in self.params]
