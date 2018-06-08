from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.vector.study_sites import configure_site
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.SetupParser import SetupParser
from simtools.ModBuilder import SingleSimulationBuilder
from dtk.generic import serialization
from os import path

# This block will be used unless overridden on the command-line
SetupParser.default_block = 'HPC'

cb = DTKConfigBuilder.from_defaults('VECTOR_SIM')
configure_site(cb, 'Namawala')

# cb_serializing = DTKConfigBuilder.from_defaults('VECTOR_SIM')
# configure_site(cb_serializing, 'Namawala')

# cb_reloading = DTKConfigBuilder.from_defaults('VECTOR_SIM')
# configure_site(cb_reloading, 'Namawala')



if __name__ == "__main__":
    SetupParser.init()
    exp_manager = ExperimentManagerFactory.init()
    exp_manager.experiment_tags['role'] = 'serialization_test'

    cb.set_param("Config_Name", "Vector default sim")
    cb.set_param("Description", "Namawala from defaults")
    fullrun_builder = SingleSimulationBuilder()
    fullrun_builder.tags['role'] = 'fullrun_default'
    exp_manager.run_simulations(config_builder=cb,
                                exp_builder=fullrun_builder,
                                exp_name="Sample serialization test")

    timesteps_to_serialize = [5, 10, 50]
    old_simulation_duration = cb.params["Simulation_Duration"]
    serialized_last_timestep = timesteps_to_serialize[-1]
    serialization.add_SerializationTimesteps(config_builder=cb,
                                             timesteps=timesteps_to_serialize,
                                             end_at_final=True)
    cb.set_param("Config_Name","Vector serlializing sim")
    serialized_builder = SingleSimulationBuilder()
    serialized_builder.tags['role'] = 'serializer'
    exp_manager.run_simulations(config_builder=cb,
                                exp_builder=serialized_builder,
                                exp_name="Sample serialization test")
    exp_manager.wait_for_finished()


    serialized_sim = exp_manager.experiment.get_simulations_with_tag('role','serializer')[0]
    serialized_output_path = path.join(serialized_sim.get_path(), 'output')
    s_pop_filename = "state-00050.dtk"

    cb.set_param("Config_Name", "Vector reloading sim")
    reloaded_builder = SingleSimulationBuilder()
    reloaded_builder.tags['role'] = 'reloader'
    cb.params["Start_Time"] = serialized_last_timestep
    cb.params["Simulation_Duration"] = old_simulation_duration - serialized_last_timestep
    cb.params.pop("Serialization_Time_Steps")
    serialization.load_Serialized_Population(config_builder=cb,
                                             population_filenames=[s_pop_filename],
                                             population_path=serialized_output_path)
    exp_manager.run_simulations(config_builder=cb,
                                exp_builder=reloaded_builder,
                                exp_name="Sample serialization test")
    print(exp_manager.experiment.simulations[0].tags)

