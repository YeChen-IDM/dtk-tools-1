from os import path

from dtk.generic.serialization import add_SerializationTimesteps, load_Serialized_Population
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.vector.study_sites import configure_site
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.SetupParser import SetupParser

# This block will be used unless overridden on the command-line
SetupParser.default_block = 'HPC'

# Set the simulation duration. Needs to be longer than the last timestep to serialize
Simulation_Duration = 365

# Choose at which timesteps we want to serialize. Here 5, 10 and 50
timesteps_to_serialize = [5, 10, 50]

# Which timesteps we want to reload from ?
timestep_to_reload = timesteps_to_serialize[-1]  # Let's choose 50

if __name__ == "__main__":
    # Initialize the SetupParser
    SetupParser.init()

    # Create a configbuilder for the simulation to serialize
    cb_serializing = DTKConfigBuilder.from_defaults('VECTOR_SIM')
    configure_site(cb_serializing, 'Namawala')
    cb_serializing.set_param("Simulation_Duration", Simulation_Duration)

    # Create an experiment manager
    exp_manager = ExperimentManagerFactory.init()

    # Set those timesteps
    add_SerializationTimesteps(config_builder=cb_serializing, timesteps=timesteps_to_serialize, end_at_final=True)

    # Run the simulation
    exp_manager.run_simulations(config_builder=cb_serializing,
                                exp_name="Sample serialization test")

    # Wait for it to finish
    exp_manager.wait_for_finished()

    # Now that the simulation is done, we need to retrieve the path for the serialized files
    serialized_sim = exp_manager.experiment.simulations[0]
    serialized_output_path = path.join(serialized_sim.get_path(), 'output')
    s_pop_filename = "state-{0:05d}.dtk".format(timestep_to_reload)

    # Create a config builder to reload
    cb_reloading = DTKConfigBuilder.from_defaults('VECTOR_SIM')
    configure_site(cb_reloading, 'Namawala')
    cb_reloading.set_param("Simulation_Duration", Simulation_Duration)

    # Prepare to reload
    cb_reloading.params["Start_Time"] = timestep_to_reload
    cb_reloading.params["Simulation_Duration"] = Simulation_Duration - timestep_to_reload
    cb_reloading.params.pop("Serialization_Time_Steps")
    load_Serialized_Population(config_builder=cb_reloading, population_filenames=[s_pop_filename],
                               population_path=serialized_output_path)

    # Run the reloaded simulation
    exp_manager.run_simulations(config_builder=cb_reloading,
                                exp_name="Sample serialization test")

