import logging
from sys import exit# as exit

import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import seuclidean

from calibtool.NextPointAlgorithm import NextPointAlgorithm

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimTool(NextPointAlgorithm):
    '''
    OptimTool

    The basic idea of OptimTool is
    '''

    def __init__(self, prior_fn,
                 x0 = [],
                 initial_samples = 1e2, # Should be a collection of samples for OptimTool, not a number of samples to draw from the prior
                 samples_per_iteration = 1e2,
                 current_state = {}
             ):

        print "__init__"
        print " --> initial_samples:\n", initial_samples

        super(OptimTool, self).__init__(prior_fn, initial_samples, samples_per_iteration, current_state)
        '''
        self.prior_fn = prior_fn
        self.samples_per_iteration = int(samples_per_iteration)
        self.set_current_state(current_state)
        if self.samples.size == 0:
            self.set_initial_samples(initial_samples)

        self.iteration = 0
        self.max_resampling_attempts = 100  # IMIS
        self.n_dimensions = self.samples[0].size

        logger.info('%s instance with %d initial %d-dimensional samples and %d per iteration',
                    self.__class__.__name__, self.samples.shape[0], 
                    self.n_dimensions, self.samples_per_iteration)
        '''

        self.X = x0


    def set_current_state(self, state):
        print "set_current_state"
        '''
        Initialize the current state,
        either to initially empty defaults or the de-serialized state
        passed according to the 'state' argument.
        '''

        super(OptimTool, self).set_current_state(state)
        '''
        self.samples = state.get('samples', np.array([]))
        self.latest_samples = state.get('latest_samples', np.array([]))

        if (self.samples.size > 0) ^ (self.latest_samples.size > 0):
            raise Exception('Both samples (size=%d) and latest_samples (size=%d) '
                            'should be empty or already initialized.',
                            self.samples.size, self.latest_samples.size)

        self.results = state.get('results', [])
        self.priors = state.get('priors', [])
        '''

        self.n_initial_samples = state.get('x_curr', [])


    def set_initial_samples(self, initial_samples):
        print "set_initial_samples"
        super(OptimTool, self).set_initial_samples(initial_samples)
        '''
        if isinstance(initial_samples, (int, float)):  # allow float like 1e3
            self.samples = self.sample_from_function(self.prior_fn, int(initial_samples))
        elif isinstance(initial_samples, (list, np.ndarray)):
            self.samples = np.array(initial_samples)
        else:
            raise Exception("The 'initial_samples' parameter must be a number or an array.")

        logger.debug('Initial samples:\n%s' % self.samples)
        self.latest_samples = self.samples[:]
        '''


    def update_iteration(self, iteration):
        print "update_iteration"
        super(OptimTool, self).update_iteration(iteration)
        '''
        self.iteration = iteration
        logger.info('Updating %s at iteration %d:', self.__class__.__name__, iteration)

        self.priors.extend(self.prior_fn.pdf(self.latest_samples))
        logger.debug('Priors:\n%s', self.priors)
        '''



        ''' from IMIS
        if not self.iteration :
            sampling_envelope = self.priors
        else:
            w = float(self.n_initial_samples) / self.samples_per_iteration
            stack = np.vstack([[np.multiply(self.priors, w)], self.gaussian_probs])
            logger.debug('Stack weighted prior + gaussian sample prob %s:\n%s', stack.shape, stack)
            norm = (w + self.D + (self.iteration - 2))
            sampling_envelope = np.sum(stack, 0) / norm

        logger.debug('Sampling envelope:\n%s', sampling_envelope)

        self.weights = [p * l / e for (p, l, e) in zip(self.priors, self.results, sampling_envelope)] # TODO: perform in log space
        self.weights /= np.sum(self.weights)
        logger.debug('Weights:\n%s', self.weights)
        '''

    def update_samples(self):
        print "update_samples"
        '''
        Perform linear regression.
        Compute goodness of fit.
        '''

        super(OptimTool, self).update_samples()
        '''
        next_samples = self.sample_from_function(
            self.next_point_fn(), self.samples_per_iteration)

        self.latest_samples = self.verify_valid_samples(next_samples)
        logger.debug('Next samples:\n%s', self.latest_samples)

        self.samples = np.concatenate((self.samples, self.latest_samples))
        logger.debug('All samples:\n%s', self.samples)
        '''

    def next_point_fn(self):
        print "next_point_fn"
        # Pick points on hypersphere
        return multivariate_normal(mean=self.gaussian_centers[-1], cov=self.gaussian_covariances[-1])

    def end_condition(self):
        print "end_condition"
        # Stopping Criterion:
        # Return True to stop, False to continue
        logger.info('Continuing iterations ...')
        return False

    def get_final_samples(self):
        print "get_final_samples"
        '''
        Resample Stage:
        '''
        return dict(samples=self.samples)

        '''
        nonzero_idxs = self.weights > 0
        idxs = [i for i, w in enumerate(self.weights[nonzero_idxs])]
        try:
            resample_idxs = np.random.choice(idxs, self.n_resamples, replace=True, p=self.weights[nonzero_idxs])
        except ValueError:
            # To isolate dtk-tools issue #96
            print(nonzero_idxs)
            print(self.weights)
            print(idxs)
            raise

        return dict(samples=self.samples[resample_idxs], weights=self.weights[resample_idxs])
        '''

    def get_current_state(self) :
        print "get_current_state"
        state = super(OptimTool, self).get_current_state()
        '''
        return dict(samples=self.samples, 
                    latest_samples=self.latest_samples,
                    priors=self.priors,
                    results=self.results)
        '''

        optimtool_state = dict( X = self.X )
        state.update(optimtool_state)
        return state
