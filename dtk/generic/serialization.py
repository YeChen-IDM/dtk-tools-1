def add_SerializationTimesteps(config_builder, timesteps, end_at_final=False, use_absolute_times=False):
    """
    Serialize the population of this simulation at specified timesteps.
    If the simulation is run on multiple cores, multiple files will be created.

    :param config_builder: A DTK Config builder
    :param timesteps: Array of integers representing the timesteps to use
    :param end_at_final: (False) set the simulation duration such that the last
                         serialized_population file ends the simluation. NOTE: may not work
                         if timestep size is not 1
     :use_absolute_times: (False) method will define simulation times instead of timesteps
                          see documentation on "Serailization_Type" for details
    """
    if not use_absolute_times:
        config_builder.set_param("Serialization_Type", "TIMESTEP") #Note: This should work in both 2.18 and 2.20
        config_builder.set_param("Serialization_Time_Steps", sorted(timesteps))
    else:
        config_builder.set_param("Serialization_Type", "TIME")
        config_builder.set_param("Serialization_Times", sorted(timesteps))

    if end_at_final:
        start_day = config_builder.params["Start_Time"]
        last_serialization_day = sorted(timesteps)[-1]
        end_day = start_day + last_serialization_day
        config_builder.set_param("Simulation_Duration", end_day)

def load_Serialized_Population(config_builder, population_path, population_filenames):
    """
    Sets simulation to load a serialized population from the filesystem

    :param config_builder: a DTK config builder
    :param population_path: relative path from the working directory to
                            the location of the serialized population files.
    :param population_filenames: names of files in question
    """
    config_builder.set_param("Serialized_Population_Path", population_path)
    config_builder.set_param("Serialized_Population_Filenames", population_filenames)
