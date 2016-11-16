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

        self.args = locals()     # Store inputs in case set_state is called later and we want to override with new (user) args
        del self.args['self']
        self.need_resolve = False

        self.constrain_sample_fn = constrain_sample_fn

        self.regression = pd.DataFrame(columns=['Iteration', 'Parameter', 'Value'])   # Parameters: Rsquared, Regression_Parameters
        self.regression['Iteration'] = self.regression['Iteration'].astype(int)

        self.state = pd.DataFrame(columns=['Iteration', 'Parameter', 'Center', 'Min', 'Max', 'Dynamic'])
        self.state['Iteration'] = self.state['Iteration'].astype(int)

        self.params = params # TODO: Check min <= center <= max
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.center_repeats = center_repeats
        self.rsquared_thresh = rsquared_thresh
        self.samples_per_iteration = int(samples_per_iteration)


    def resolve_args(self, iteration):
        # Have args from user and from set_state.
        # Note this is called only right before commissioning a new iteration, likely from 'resume'
        print 'resolve_args', iteration

        # TODO: be more sensitive with params, user could have added or removed variables, need to adjust
        self.params = self.args('params') # TODO: Check min <= center <= max
        self.mu_r = self.args('mu_r')
        self.sigma_r = self.args('sigma_r')
        self.center_repeats = self.args('center_repeats')
        self.rsquared_thresh = self.args('rsquared_thresh')
        self.samples_per_iteration = int(self.args('samples_per_iteration'))


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
        # Update args
        if self.need_resolve:
            resolve_args(self, iteration)

        if iteration == 0:
            # Choose initial samples
            samples = self.choose_initial_samples()
        else:
            # Regress inputs and results from previous iteration
            # Move X_center, choose hypersphere, save in dataframe
            samples = self.choose_samples_via_gradient_ascent(iteration)

        #return self.data.query('Iteration == @iteration').sort_values('__sample_index__')[self.get_param_names()] # Query makes a copy
        return samples

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


    def set_results_for_iteration(self, iteration, results):
        logger.info('%s: Choosing samples at iteration %d:', self.__class__.__name__, iteration)
        logger.debug('Results:\n%s', results)

        data_by_iter = self.data.set_index('Iteration')
        if iteration+1 in data_by_iter.index.unique():
            # Been here before, reset
            data_by_iter = data_by_iter.loc[:iteration]

            regression_by_iter = self.regression.set_index('Iteration')
            self.regression = regression_by_iter.loc[:iteration-1].reset_index()

            state_by_iter = self.state.set_index('Iteration')
            self.state = state_by_iter.loc[:iteration].reset_index()

        # Store results ... even if changed
        data_by_iter.loc[iteration,'Results'] = results
        self.data = data_by_iter.reset_index()


    def choose_initial_samples(self):
        self.data = pd.DataFrame(columns=[['Iteration', '__sample_index__', 'Results', 'Fitted'] + self.get_param_names()])
        self.data['Iteration'] = self.data['Iteration'].astype(int)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

        self.n_dimensions = len(self.params)

        iteration = 0
        for param in self.params:
            print (iteration, param['Name'], param['Guess'], param['Min'], param['Max'], param['Dynamic'])
            self.state.loc[len(self.state)] = [iteration, param['Name'], param['Guess'], param['Min'], param['Max'], param['Dynamic']]

        X_center = self._get_X_center(iteration)
        initial_samples = self.choose_and_clamp_hypersphere_samples_for_iteration(iteration, X_center)

        return initial_samples


    def choose_samples_via_gradient_ascent(self, iteration):

        dynamic_params = [p['Name'] for p in self.params if p['Dynamic']]

        latest_dynamic_samples = self.data.loc[iteration, dynamic_params].values
        latest_results = self.data.loc[iteration, 'Results'].values

        mod = sm.OLS(latest_results, sm.add_constant(latest_dynamic_samples) )

        mod_fit = mod.fit()
        print mod_fit.summary()

        """
        #L1_wt : scalar : The fraction of the penalty given to the L1 penalty term. Must be between 0 and 1 (inclusive). If 0, the fit is ridge regression. If 1, the fit is the lasso.
        mod_fit = mod.fit_regularized(method='coord_descent', maxiter=10000, alpha=1.0, L1_wt=1.0, start_params=None, cnvrg_tol=1e-08, zero_tol=1e-08)
        print mod_fit.summary()

        from sklearn import linear_model
        clf = linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        clf.fit(latest_dynamic_samples, latest_results)

        print(clf.coef_)
        print(clf.intercept_)
        y = clf.predict(latest_dynamic_samples)
        print zip(latest_results, y)
        print 'R2:', clf.score(latest_dynamic_samples, latest_results)


        print 'LassoCV'
        cvf = linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
        cvf.fit(latest_dynamic_samples, latest_results)

        print(cvf.coef_)
        print(cvf.intercept_)
        y = cvf.predict(latest_dynamic_samples)
        print zip(latest_results, y)
        print 'R2:', cvf.score(latest_dynamic_samples, latest_results)
        """

        self.fit_summary = mod_fit.summary().as_csv()

        self.data.loc[iteration, 'Fitted'] = mod_fit.fittedvalues

        # Choose next X_center
        if mod_fit.rsquared > self.rsquared_thresh:
            print 'Good R^2 (%f), using params: '%mod_fit.rsquared, mod_fit.params
            coef = mod_fit.params[1:] # Drop constant
            den = np.sqrt( sum([ (p['Max']-p['Min'])**2 * c**2  for c,p in zip(coef, self.params) ]) )

            old_center = self._get_X_center(iteration)
            old_center_of_dynamic_params = [c for c,p in zip(old_center, self.params) if p['Dynamic']]
            print 'OLD CENTER:', old_center
            print 'OLD CENTER DYNAMIC:', old_center_of_dynamic_params
            new_dynamic_center = [x + (p['Max']-p['Min'])**2 * c * self.mu_r / den for x,c,p in zip(old_center_of_dynamic_params, coef, self.params) ]

        else:
            print 'Bad R^2 (%f)'%mod_fit.rsquared
            max_idx = np.argmax(latest_results)
            'Stepping to argmax of %f at:'%latest_results[max_idx], latest_dynamic_samples[max_idx]
            new_dynamic_center = latest_dynamic_samples[max_idx].tolist()

        new_center_dict = {k:v for k,v in zip(self.get_param_names(), self._get_X_center(iteration))}
        new_center_dict.update( {k:v for k,v in zip(dynamic_params, new_dynamic_center)} )
        new_center = [new_center_dict[param_name] for param_name in self.get_param_names()]

        r2_df = pd.DataFrame( [[iteration, 'Rsquared', mod_fit.rsquared]], columns=['Iteration', 'Parameter', 'Value'] )
        thresh_df = pd.DataFrame( [[iteration, 'Rsquared_Threshold', self.rsquared_thresh]], columns=['Iteration', 'Parameter', 'Value'] )
        repeats_df = pd.DataFrame( [[iteration, 'Center_Repeats', self.center_repeats]], columns=['Iteration', 'Parameter', 'Value'] )
        self.regression = pd.concat([self.regression, r2_df, thresh_df, repeats_df])
        for (p,v) in zip( ['Constant'] + self.get_param_names(), mod_fit.params): # mod.endog_names
            regression_param_df = pd.DataFrame( [[iteration, p, v]], columns=['Iteration', 'Parameter', 'Value'] )
            self.regression = pd.concat([self.regression, regression_param_df])
        for (p,v) in zip( ['P_Constant'] + ['P_'+s for s in self.get_param_names()], mod_fit.pvalues): # mod.endog_names
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
            'Max': [p['Max'] for p in self.params],
            'Dynamic': [p['Dynamic'] for p in self.params]
        } )

        self.state = pd.concat( [self.state, new_state] )

        samples = self.choose_and_clamp_hypersphere_samples_for_iteration(iteration+1, new_center)

        logger.debug('Next samples:\n%s', samples_for_next_iteration)

        return samples


    def choose_and_clamp_hypersphere_samples_for_iteration(self, iteration, X_center):
        # TODO: This function does too much, break up

        N = self.samples_per_iteration
        samples = self.sample_hypersphere(N, X_center)

        # Move static parameters to X_center
        for i,p in enumerate(self.params):
            if not p['Dynamic']:
                samples[:,i] = X_center[i]

        # CLAMP
        X_min = self.state.pivot('Iteration', 'Parameter', 'Min').loc[iteration, self.get_param_names()]
        X_max = self.state.pivot('Iteration', 'Parameter', 'Max').loc[iteration, self.get_param_names()]
        samples = self.clamp(samples, X_min, X_max)

        with pd.option_context("display.max_columns", 500):
            print self.data.head()

        self.data.reset_index(inplace=True)
        self.add_samples( samples, iteration )

        # USER CONSTRAINT FN
        self.data = self.data.set_index('Iteration')
        tmp = self.data.loc[iteration].apply( self.constrain_sample_fn, axis=1)

        self.data.loc[iteration] = self.data.loc[iteration].apply( self.constrain_sample_fn, axis=1)
        self.data.reset_index(inplace=True)
        self.data['__sample_index__'] = self.data['__sample_index__'].astype(int)

        data_by_iter = self.data.reset_index().set_index('Iteration')

        with pd.option_context("display.max_columns", 500):
            print data_by_iter.loc[iteration].head()

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

    def set_state(self, state, iteration):
        print 'set_state\n'*25

        self.mu_r = state['mu_r']
        self.sigma_r = state['sigma_r']
        self.center_repeats = state['center_repeats']
        self.rsquared_thresh = state['rsquared_thresh']
        self.n_dimensions = state['n_dimensions']
        self.params = state['params']   # NOTE: This line will override any updated user params passed to __init__
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

        self.need_resolve = True

    def get_param_names(self):
        return [p['Name'] for p in self.params]

