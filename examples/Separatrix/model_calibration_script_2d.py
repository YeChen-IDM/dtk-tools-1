import copy
import json

from calibtool.CalibManager import CalibManager
from examples.Separatrix.Models.ModelSite import ModelSite
from examples.Separatrix.Models.ModelPlotter import ModelPlotter
from examples.Separatrix.Algorithms.ModelNextPoint import ModelNextPoint
from examples.Separatrix.Models.ModelConfigBuilder import ModelConfigBuilder
from simtools.SetupParser import SetupParser

# Which simtools.ini block to use for this calibration
SetupParser.default_block = 'HPC'

# This config builder will be modify by the different sites defined below
# [TODO]: we may use PythonModelManager but we need to consider how to create config.json for each sample point!
cb = ModelConfigBuilder()
cb.set_dll_root('Assets')

# List of sites we want to calibrate on
sites = [ModelSite()]

# The default plotters
plotters = [ModelPlotter()]

params = [
    {
        'Name': 'Point_X',
        'MapTo': 'Point_X',
        'Min': 0,
        'Max': 1
    },
    {
        'Name': 'Point_Y',
        'MapTo': 'Point_Y',
        'Min': 0,
        'Max': 1
    },
]


def map_sample_to_model_input(cb, sample):
    tags = {}
    # Make a copy of samples so we can alter it safely
    sample = copy.deepcopy(sample)

    # Go through the parameters
    for p in params:
        if 'MapTo' in p:
            if p['Name'] not in sample:
                print('Warning: %s not in sample, perhaps resuming previous iteration' % p['Name'])
                continue
            value = sample.pop(p['Name'])
            cb.set_param(p['MapTo'], value)

            # Add this change to our tags
            tags[p["Name"]] = value

    cb.set_param("Dimension", len(params))
    return tags


# Load Separatrix settings
Settings = json.load(open('Settings.json', 'r'))

model_next_point = ModelNextPoint(params, Settings=Settings, Num_Dimensions=2, Num_Initial_Samples=50,
                                  Num_Next_Samples=50)

calib_manager = CalibManager(name='Example_Model_2d_1',
                             config_builder=cb,
                             map_sample_to_model_input_fn=map_sample_to_model_input,
                             sites=sites,
                             next_point=model_next_point,
                             max_iterations=3,
                             plotters=plotters)

run_calib_args = {
    "calib_manager": calib_manager
}

if __name__ == "__main__":
    SetupParser.init()
    cm = run_calib_args["calib_manager"]
    cm.run_calibration()
