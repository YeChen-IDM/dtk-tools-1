from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import SingleSimulationBuilder
from simtools.SetupParser import SetupParser

experiment_name = "Generic Default simluation test for tools"
generic_defaults = ['GENERIC_SIM_MIN',
                    'GENERIC_SIM_SIR',
                    'GENERIC_SIM_SEIR',
                    'GENERIC_SIM_SIRS',
                    'GENERIC_SIM_SEIRS',
                    'GENERIC_SIM_SI',
                    'GENERIC_SIM_SIS',
                    'GENERIC_SIM_SIR_Vaccinations_A',
                    'GENERIC_SIM_SIR_Vaccinations_B',
                    'GENERIC_SIM_SIR_Vaccinations_C']

if __name__ == '__main__':
    # Initialize the SetupParser to use HPC
    SetupParser.init('HPC')

    # Create an experiment manager
    exp_mgr = ExperimentManagerFactory.init()

    # For each generic simulations, create a config builder from the default and run it
    for sim_example in generic_defaults:
        cb = DTKConfigBuilder.from_defaults(sim_example)
        cb.set_param("Config_Name", sim_example)

        # Create a builder in order to run a single simulation
        # This step is optional usually but it allows us to set tags on the simulation
        builder = SingleSimulationBuilder()
        builder.tags.update({"Simulation_Type": sim_example})
        
        exp_mgr.run_simulations(exp_name=experiment_name, config_builder=cb, exp_builder=builder)

    # Wait for the experiment to finish
    exp_mgr.wait_for_finished()

    print("Congratulations! they ran.")
