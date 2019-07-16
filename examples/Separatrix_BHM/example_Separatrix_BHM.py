# Execute directly: 'python example_separatrix_BHM.py'
# or via the calibtool.py script: 'calibtool run example_separatrix_BHM.py'
import copy
import os
import random

from calibtool.CalibManager import CalibManager
from calibtool.algorithms.Separatrix_BHM import Separatrix_BHM
from calibtool.plotters.SeparatrixBHMPlotter import SeparatrixBHMPlotter
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.SetupParser import SetupParser

from dtk.utils.builders.TemplateHelper import TemplateHelper
from dtk.utils.builders.ConfigTemplate import ConfigTemplate
from dtk.utils.builders.TaggedTemplate import CampaignTemplate
from dtk.utils.builders.TaggedTemplate import DemographicsTemplate

from EradicationSite import EradicationSite


# Which simtools.ini block to use for this calibration
SetupParser.default_block = 'HPC'

# List of sites we want to analyze, you can list more than one, e.g. NdiopCalibSite()
sites = [ EradicationSite() ]

# Choose how you'd like to plot, you can use the SeparatrixBHMPlotter.
plotters = [ SeparatrixBHMPlotter() ]

# Create instances of the input files we want to work with
demog = DemographicsTemplate.from_file( 'demographics.json' )
cfg = ConfigTemplate.from_file( 'config.json' )
cpn = CampaignTemplate.from_file( 'campaign.json' )

# Set static parameters, these are one-time overrides to the baseline files
static_params = { 'Simulation_Duration': 90 }
cfg.set_params( static_params )

# Prepare templates
templates = TemplateHelper()
table_base = {
    'ACTIVE_TEMPLATES': [cfg, cpn],
    'TAGS': { 'Separatrix_BHM': None }
}

# Make a config builder
cb = DTKConfigBuilder()
dir_path = os.path.dirname(os.path.realpath(__file__))
cb.set_input_files_root(dir_path)
cb.ignore_missing = True

# Define the parameters on which we will conduct Separatrix analysis
params = [
    {
        'Name': 'Base Infectivity',
        #'MapTo': 'Base_Infectivity', # <-- DEMO: Custom mapping, see map_sample_to_model_input below
        'Min': 0,
        'Max': 0.4
    },
    {
        'Name': 'Vaccine Coverage',
        'MapTo': 'Demographic_Coverage__KP_Vaccine_Coverage',
        'Min': 0,
        'Max': 1.0
    },
]


def constrain_sample(sample):
    """
    This function is called on every samples and allow the user to edit them before they are passed
    to the map_sample_to_model_input function.
    It is useful to round some parameters as demonstrated below.
    Can do much more here, e.g. for
    # Clinical Fever Threshold High <  MSP1 Merozoite Kill Fraction
    if 'Clinical Fever Threshold High' and 'MSP1 Merozoite Kill Fraction' in sample:
        sample['Clinical Fever Threshold High'] = \
            min( sample['Clinical Fever Threshold High'], sample['MSP1 Merozoite Kill Fraction'] )
    You can omit this function by not specifying it in the Separatrix constructor.
    :param sample: The sample coming from the next point algorithm
    :return: The sample with constrained values
    """
    # Convert Falciparum MSP Variants to nearest integer
    if 'Some_Parameter' in sample:
        sample['Some_Parameter'] = int(round(sample['Some_Parameter']))

    return sample


def map_sample_to_model_input(cb, sample):
    """
    This method needs to map the samples generated by the next point algorithm to the model inputs (represented here by the cb).
    It is important to note that the sample may be shared by several isntances of this function.
    Therefore it is important to deepcopy the sample at the beginning if we intend to modify it (by calling .pop() for example).
       sample = copy.deepcopy(sample)
    :param cb: The config builder representing the model inputs for this particular simulation
    :param sample: The sample containing a values for all the params. e.g. {'Vaccine Coverage': 0.7, ... }
    :return: A dictionary containing the tags that will be attached to the simulation
    """
    table = copy.deepcopy(table_base)
    table['Run_Number'] = random.randint(0, 1e6) # Random random number seed
    table['Config_Name'] = 'SIR Separatirx'

    # Make a copy of samples so we can alter it safely
    sample = copy.deepcopy(sample)

    # Can perform custom mapping, e.g. a trivial example
    if 'Base Infectivity' in sample:
        value = sample.pop('Base Infectivity')
        table['Base_Infectivity'] = value # Could do value**2 or whatever

    for p in params:
        if 'MapTo' in p:
            value = sample.pop( p['Name'] )
            if isinstance( p['MapTo'], list):
                for mapto in p['MapTo']:
                    table[mapto] = value
            else:
                table[p['MapTo']] = value

    for name,value in sample.items():
        print('UNUSED PARAMETER:'+name)

    assert( len(sample) == 0 ) # All params used

    return templates.mod_dynamic_parameters(cb, table)

MAX_ITERATIONS= 10
separatrix = Separatrix_BHM(params,
    constrain_sample,
    implausibility_threshold = 3, # <-- Essentially the risk tolerance.  Higher numbers will be more careful to not reject potentially good regions of parameter space, but the result is that more iteration/simulations will be required

    target_success_probability = 0.7, # <-- This is the success probability isocline that we seek to identify
    num_past_iterations_to_include_in_metamodel = -1, # <-- When emulating the latent success probability function, include simulation results from this many previous iterations.  NOTE: Set <0 to include all previous simulation.
    samples_per_iteration = 64, #  <-- Number of samples per iteration.  Actual number of sims run is this number times number of sites.
    samples_final_iteration = 64, #  <-- Number of samples for the final iteration.  Actual number of sims run is this number times number of sites.
    max_iterations = MAX_ITERATIONS, # <-- Iterations
    training_frac = 0.8 # <-- Fraction of simulations to use a training data (will plot in cyan instead of magenta)
)

calib_manager = CalibManager(name='Example_Separatrix_BHM',    # <-- Please customize this name
                             config_builder = cb,
                             map_sample_to_model_input_fn = map_sample_to_model_input,
                             sites = sites,
                             next_point = separatrix,
                             sim_runs_per_param_set = 1, # <-- Replicates, 1 is ideal for Separatrix
                             max_iterations = MAX_ITERATIONS, # <-- Iterations
                             plotters = plotters)


run_calib_args = {
    "calib_manager": calib_manager
}

if __name__ == "__main__":
    SetupParser.init()
    cm = run_calib_args["calib_manager"]
    cm.run_calibration()