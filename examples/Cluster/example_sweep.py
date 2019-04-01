import numpy as np
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.utils.builders.sweep import GenericSweepBuilder
from dtk.vector.study_sites import configure_site
from simtools.SetupParser import SetupParser
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
simulation_directory = os.path.join(current_directory, "experiments")

# Run with the CLUSTER mode
SetupParser.default_block = "ND"
SetupParser.ini_filename = "custom.ini"

# Configure a default 5 years simulation
cb = DTKConfigBuilder.from_defaults('MALARIA_SIM', Simulation_Duration=365 * 5)

# Set the path manually to make them relative to the current directory
cb.set_experiment_executable(os.path.join(current_directory, "..", "inputs", "Eradication.exe"))
cb.set_input_files_root(os.path.join(current_directory, "..", "inputs"))

# Set it in Namawala
configure_site(cb, 'Namawala')

# Name of the experiment
exp_name = 'ExampleSweep'

# Create a builder to sweep over the birth rate multiplier
builder = GenericSweepBuilder.from_dict({'x_Birth': np.arange(1, 1.5, .1)})

run_sim_args = {
    'exp_name': exp_name,
    'exp_builder': builder,
    'config_builder': cb
}

if __name__ == "__main__":
    SetupParser.init()
    SetupParser.set(SetupParser.selected_block, "sim_root", simulation_directory)
    exp_manager = ExperimentManagerFactory.init()
    exp_manager.experiment_tags = {"foo":"bar"}
    exp_manager.run_simulations(**run_sim_args)
