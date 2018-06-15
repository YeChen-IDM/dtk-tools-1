import shutil
import os
from dtk.generic.serialization import load_Serialized_Population, add_SerializationTimesteps
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.vector.study_sites import configure_site
from simtools.Analysis.AnalyzeManager import AnalyzeManager
from simtools.Analysis.BaseAnalyzers import DownloadAnalyzer
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.SetupParser import SetupParser

SetupParser.default_block = 'HPC'
timestep_to_reload=150


if __name__ == "__main__":
    # Initialize the SetupParser
    SetupParser.init()

    # Create a configbuilder for the simulation to serialize
    cb_serializing = DTKConfigBuilder.from_defaults('VECTOR_SIM')
    configure_site(cb_serializing, 'Namawala')
    cb_serializing.set_param("Simulation_Duration", 365)

    # Create an experiment manager
    exp_manager = ExperimentManagerFactory.init()

    # Set the serialization
    add_SerializationTimesteps(config_builder=cb_serializing, timesteps=[timestep_to_reload], end_at_final=True)

    s_pop_filename = "state-{0:05d}.dtk".format(timestep_to_reload)
    # Run the simulation
    exp_manager.run_simulations(config_builder=cb_serializing,
                                exp_name="Sample serialization test")

    exp_manager.wait_for_finished(verbose=True)

    # Download the state file
    simulation = exp_manager.experiment.simulations[0]
    da = DownloadAnalyzer(filenames=["output\\{}".format(s_pop_filename)], output_path="temp")
    am = AnalyzeManager(sim_list=[simulation], analyzers=[da])
    am.analyze()

    # Add the state file
    cb_reload = DTKConfigBuilder.from_defaults('VECTOR_SIM')
    configure_site(cb_reload, 'Namawala')
    cb_reload.set_param("Simulation_Duration", 365)
    cb_reload.experiment_files.add_file(os.path.join("temp", simulation.id, s_pop_filename))

    load_Serialized_Population(cb_reload, 'Assets', [s_pop_filename])

    # Run !
    exp_manager.run_simulations(config_builder=cb_reload)

    # Cleanup temp directory
    shutil.rmtree("temp")
