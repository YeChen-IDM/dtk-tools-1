# Execute directly: 'python example_optimization.py'
# or via the calibtool.py script: 'calibtool run example_optimization.py'

import numpy as np
from scipy.stats import uniform, norm
import collections

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

#cb.add_input_file('test.q',"test")

params = collections.OrderedDict()
params['x_Temporary_Larval_Habitat'] = { 'Min': 0.1, 'Max': 1.9 }
params['MSP1_Merozoite_Kill_Fraction'] = { 'Min': 0.4, 'Max': 0.7 }
params['Nonspecific_Antigenicity_Factor'] = { 'Min': 0.1, 'Max': 0.9 }

plotters = [LikelihoodPlotter(True), SiteDataPlotter(True), OptimToolPlotter()]

def sample_point_fn(cb, param_values):
    """
    A simple example function that takes a list of sample-point values
    and sets parameters accordingly using the parameter names from the prior.
    Note that more complicated logic, e.g. setting campaign event coverage or habitat abundance by species,
    can be encoded in a similar fashion using custom functions rather than the generic "set_param".
    """
    params_dict = dict(zip(params.keys(), param_values))
    for param, value in params_dict.iteritems():
        cb.set_param(param,value)
    cb.set_param('Simulation_Duration',365)
    return params_dict

x0 = [0.2, 0.45, 0.25]
mu_r = 0.10
sigma_r = 0.01

next_point_kwargs = dict(
        x0=x0,
        mu_r = mu_r,
        sigma_r = sigma_r,
        initial_samples = 32,
        samples_per_iteration = 32,
        center_repeats = 2
    )

calib_manager = CalibManager(name='ExampleOptimization',
                             setup=SetupParser(),
                             config_builder=cb,
                             sample_point_fn=sample_point_fn,
                             sites=sites,
                             next_point=OptimTool(params, **next_point_kwargs),
                             sim_runs_per_param_set=1, # <-- Replicates
                             max_iterations=10,
                             num_to_plot=10,
                             plotters=plotters)

#run_calib_args = {'selected_block': "EXAMPLE"}
run_calib_args = {}

if __name__ == "__main__":
    run_calib_args.update(dict(location='LOCAL'))
    calib_manager.run_calibration(**run_calib_args)
