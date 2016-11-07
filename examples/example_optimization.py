# Execute directly: 'python example_optimization.py'
# or via the calibtool.py script: 'calibtool run example_optimization.py'

import numpy as np
from scipy.stats import uniform, norm
import collections
import random

from calibtool.CalibManager import CalibManager
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
print 'Dielmo only for now'
sites = [sites[0]]

# TODO: 'Frozen': False
params = [
    {
        'Name': 'Min Days Between Clinical Incidents',
        'MapTo': 'Min_Days_Between_Clinical_Incidents',
        'Guess': 25,
        'Min': 1,
        'Max': 50
    },

    {
        'Name': 'Clinical Fever Threshold High',
        'MapTo': 'Clinical_Fever_Threshold_High',
        'Guess': 1.75,
        'Min': 0.5,
        'Max': 2.5
    }
]
'''
    {
        'Name': 'MSP1 Merozoite Kill Fraction',
        'MapTo': 'MSP1_Merozoite_Kill_Fraction',
        'Guess': 0.65,
        'Min': 0.4,
        'Max': 0.7
    },
    {
        'Name': 'Falciparum PfEMP1 Variants',
        'MapTo': 'Falciparum_PfEMP1_Variants',
        'Guess': 1500,
        'Min': 900, # 0
        'Max': 1700 # 1e5
    },
'''

# Antigen_Switch_Rate (1e-10 to 1e-8, log)
# Falciparum_PfEMP1_Variants (900 to 1700, linear int)
# Falciparum_MSP_Variants (5 to 50, linear int)
# OR "Min_Days_Between_Clinical_Incidents": 14, [ integer? ]

# Build optimization parameter name --> model input parameter from params above
mapping = { p['Name']:p['MapTo'] for p in params }

def constrain_sample( sample ):

    # Convert Falciparum MSP Variants to nearest integer
    sample['Min Days Between Clinical Incidents'] = int( round(sample['Min Days Between Clinical Incidents']) )

    # Clinical Fever Threshold High <  MSP1 Merozoite Kill Fraction
    '''
    if 'Clinical Fever Threshold High' in sample and 'MSP1 Merozoite Kill Fraction' in sample:
        sample['Clinical Fever Threshold High'] = \
            min( sample['Clinical Fever Threshold High'], sample['MSP1 Merozoite Kill Fraction'] )
    '''

    return sample

plotters = [    LikelihoodPlotter(combine_sites=True), 
                SiteDataPlotter(num_to_plot=10, combine_sites=True),
                OptimToolPlotter()] # OTP must be last because it calls gc.collect()

def map_sample_to_model_input(cb, sample):
    sample['Simulation_Duration'] = 10*365
    sample['Run_Number'] = random.randint(0, 1e6)

    tags = {}
    for name, value in sample.iteritems():
        print (name, value)
        if name in mapping:
            map_to_name = mapping[name]
            if map_to_name is 'Custom':
                print 'TODO: Handle custom mapping for', map_to_name
            else:
                tags.update( cb.set_param(map_to_name, value) )
        else:
            tags.update( cb.set_param(name, value) )

    return tags

optimtool = OptimTool(params, 
    constrain_sample, # <-- WILL NOT BE SAVED IN ITERATION STATE
    mu_r = 0.05,      # <-- Mean percent of parameter range for numerical derivatve.  CAREFUL with integer parameters!
    sigma_r = 0.01,   # <-- stdev of above
    samples_per_iteration = 32,
    center_repeats = 2
)

calib_manager = CalibManager(name='ExampleOptimization',
                             setup = SetupParser(),
                             config_builder = cb,
                             map_sample_to_model_input_fn = map_sample_to_model_input,
                             sites = sites,
                             next_point = optimtool,
                             sim_runs_per_param_set = 1, # <-- Replicates
                             max_iterations = 10,        # <-- Iterations
                             plotters = plotters)

#run_calib_args = {'selected_block': "EXAMPLE"}
run_calib_args = {}

if __name__ == "__main__":
    run_calib_args.update(dict(location='LOCAL'))
    calib_manager.run_calibration(**run_calib_args)
