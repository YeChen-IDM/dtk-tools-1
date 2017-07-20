from calibtool.CalibManager import CalibManager
from calibtool.study_sites.DielmoCalibSite import DielmoCalibSite
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from examples.newalgo.MyAlgorithm import MyAlgorithm
from simtools.SetupParser import SetupParser

SetupParser.default_block = 'HPC'

cb = DTKConfigBuilder.from_defaults('MALARIA_SIM')

sites = [DielmoCalibSite()]
plotters = []

params = [
    {
        'Name': 'Clinical Fever Threshold High',
        'Dynamic': True,
        'MapTo': 'Clinical_Fever_Threshold_High',
        'Guess': 1.75,
        'Min': 0.5,
        'Max': 2.5
    },
    {
        'Name': 'MSP1 Merozoite Kill Fraction',
        'Dynamic': False,   # <-- NOTE: this parameter is frozen at Guess
        'MapTo': 'MSP1_Merozoite_Kill_Fraction',
        'Guess': 0.65,
        'Min': 0.4,
        'Max': 0.7
    }
]


def map_sample_to_model_input(cb, sample):
    for p in params:
        if not p['Dynamic']: continue
        cb.set_param(p['MapTo'], sample.pop(p['Name']))

    return sample


algo = MyAlgorithm()

calib_manager = CalibManager(name='ExampleOptimization',    # <-- Please customize this name
                             config_builder=cb,
                             map_sample_to_model_input_fn = map_sample_to_model_input,
                             sites = sites,
                             next_point = algo,
                             sim_runs_per_param_set = 1, # <-- Replicates
                             max_iterations = 3,         # <-- Iterations
                             plotters = plotters)

run_calib_args = {}

if __name__ == "__main__":
    SetupParser.init()
    calib_manager.run_calibration()
