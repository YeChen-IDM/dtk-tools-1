# Execute directly: 'python example_optimization.py'
# or via the calibtool.py script: 'calibtool run example_optimization.py'

import numpy as np
from scipy.stats import uniform, norm

from calibtool.CalibManager import CalibManager
from calibtool.Prior import MultiVariatePrior
from calibtool.algo.OptimTool import OptimTool
from calibtool.analyzers.DTKCalibFactory import DTKCalibFactory
from calibtool.plotters.LikelihoodPlotter import LikelihoodPlotter
from calibtool.plotters.SiteDataPlotter import SiteDataPlotter
from calibtool.plotters.OptimToolPlotter import OptimToolPlotter

from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.SetupParser import SetupParser

cb = DTKConfigBuilder.from_defaults('MALARIA_SIM')

analyzer = DTKCalibFactory.get_analyzer(
    'ClinicalIncidenceByAgeCohortAnalyzer', weight=1)

sites = [DTKCalibFactory.get_site('Dielmo', analyzers=[analyzer]),
         DTKCalibFactory.get_site('Ndiop', analyzers=[analyzer])]


print 'TEMP: only Dielmo'
sites = [sites[0]]

cb.add_input_file('test.q',"test")

prior = MultiVariatePrior.by_param(
    x_Temporary_Larval_Habitat      = uniform(loc=0.1, scale=1.8),  # from 0.1 to 1.9
    MSP1_Merozoite_Kill_Fraction    = uniform(loc=0.4, scale=0.3),  # from 0.4 to 0.7
    Nonspecific_Antigenicity_Factor = uniform(loc=0.1, scale=0.8)  # from 0.1 to 0.9
)

plotters = [LikelihoodPlotter(True), SiteDataPlotter(True), OptimToolPlotter()]

def sample_point_fn(cb, param_values):
    """
    A simple example function that takes a list of sample-point values
    and sets parameters accordingly using the parameter names from the prior.
    Note that more complicated logic, e.g. setting campaign event coverage or habitat abundance by species,
    can be encoded in a similar fashion using custom functions rather than the generic "set_param".
    """
    #cb.input_files['test.q'] +=  str(param_values[0])

    params_dict = dict(zip(prior.params, param_values))
    for param, value in params_dict.iteritems():
        cb.set_param(param,value)
    cb.set_param('Simulation_Duration',365)
    return params_dict

x0 = [1, 0.55, 0.45]
mu_r = 0.05
sigma_r = 0.01

next_point_kwargs = dict(
        x0=x0,
        mu_r = mu_r,
        sigma_r = sigma_r,
        initial_samples = 8,
        samples_per_iteration = 8
    )

calib_manager = CalibManager(name='ExampleOptimization',
                             setup=SetupParser(),
                             config_builder=cb,
                             sample_point_fn=sample_point_fn,
                             sites=sites,
                             next_point=OptimTool(prior, **next_point_kwargs),
                             sim_runs_per_param_set=1, # <-- Replicates
                             max_iterations=5,
                             num_to_plot=10,
                             plotters=plotters)

#run_calib_args = {'selected_block': "EXAMPLE"}
run_calib_args = {}

if __name__ == "__main__":
    run_calib_args.update(dict(location='LOCAL'))
    calib_manager.run_calibration(**run_calib_args)
