import itertools
import os
import traceback

from simtools.Analysis.OutputParser import SimulationOutputParser
from simtools.Utilities.COMPSCache import COMPSCache
from simtools.Utilities.COMPSUtilities import COMPS_login, get_asset_files_for_simulation_id
from simtools.Utilities.RetryDecorator import retry


def retrieve_data(simulation) -> None:
    """
    Simple wrapper to unpack the data coming from the process pool and pass it to the function actually doing the work.

    Args:
        simulation: The simulation to process

    Returns: Nothing

    """
    # Retrieve the global variables coming from the pool initialization
    analyzers = retrieve_data.analyzers
    cache = retrieve_data.cache
    path_mapping = retrieve_data.path_mapping

    retrieve_data_for_simulation(simulation, analyzers, cache, path_mapping)


def set_exception(step: str, info: dict, cache: any) -> None:
    """
    Helper to quickly set an exception in the cache.

    Args:
        step: Which step encountered an error
        info: Dictionary for additional information to add to the message
        cache: The cache object in which to set the exception

    Returns: Nothing

    """
    from simtools.Analysis.AnalyzeManager import EXCEPTION_KEY
    message = f"\nAn exception has been raised during {step}.\n"
    # Add the info
    for ikey, ivalue in info.items():
        message += f"- {ikey}: {ivalue}\n"

    # Add the traceback
    message += f'\n{traceback.format_exc()}\n'

    cache.set(EXCEPTION_KEY, message)


def retrieve_SSMT_files(simulation, filenames, path_mapping):
    byte_arrays = {}

    for filename in filenames:
        # Create the path by replacing the part of the path that is mounted locally
        path = os.path.join(simulation.get_path(), filename).lower()
        path = path.replace(path_mapping[1], path_mapping[0])
        path = path.replace("\\", "/")

        # Open the file
        with open(path, 'rb') as output_file:
            byte_arrays[filename] = output_file.read()
    return byte_arrays


@retry(ConnectionError, tries=5, delay=3, backoff=2)
def retrieve_COMPS_AM_files(simulation, filenames):
    byte_arrays = {}

    # Login and retrieve the COMPS simulation
    COMPS_login(simulation.experiment.endpoint)
    COMPS_simulation = COMPSCache.simulation(simulation.id)

    # Separate the files in assets (for those with Assets in the path) and transient (for the others)
    assets = [path for path in filenames if path.lower().startswith("assets")]
    transient = [path for path in filenames if not path.lower().startswith("assets")]

    # Retrieve and store
    if transient:
        byte_arrays.update(dict(zip(transient, COMPS_simulation.retrieve_output_files(paths=transient))))

    if assets:
        byte_arrays.update(get_asset_files_for_simulation_id(simulation.id, paths=assets, remove_prefix='Assets'))

    return byte_arrays


def retrieve_data_for_simulation(simulation, analyzers, cache, path_mapping):
    # Filter first and get the filenames from filtered analysis
    filtered_analysis = [a for a in analyzers if a.filter(simulation)]
    filenames = set(itertools.chain(*(a.filenames for a in filtered_analysis)))

    # We dont have anything to do :)
    if not filtered_analysis:
        cache.set(simulation.id, None)
        return

    # The byte_arrays will associate filename with content
    byte_arrays = {}

    try:
        # Retrieval for SSMT
        if path_mapping:
            byte_arrays = retrieve_SSMT_files(simulation, filenames, path_mapping)

        # Retrieval for normal HPC Asset Management
        elif simulation.experiment.location == "HPC":
            byte_arrays = retrieve_COMPS_AM_files(simulation, filenames)

        # Retrieval for local file
        else:
            for filename in filenames:
                path = os.path.join(simulation.get_path(), filename)
                with open(path, 'rb') as output_file:
                    byte_arrays[filename] = output_file.read()
    except:
        set_exception(step="data retrieval",
                      info={"Simulation": simulation,
                            "Analyzers": ", ".join([a.uid for a in analyzers]),
                            "Files": ", ".join(filenames)},
                      cache=cache)
        return

    # Selected data will be a dict with analyzer.uid => data
    selected_data = {}
    for analyzer in filtered_analysis:
        # If the analyzer needs the parsed data, parse
        if analyzer.parse:
            try:
                data = {filename: SimulationOutputParser.parse(filename, content)
                        for filename, content in byte_arrays.items()}
            except:
                set_exception(step="data parsing",
                              info={"Simulation": simulation, "Analyzer": analyzer.uid},
                              cache=cache)
                return
        else:
            # If the analyzer doesnt wish to parse, give the raw data
            data = byte_arrays

        # Retrieve the selected data for the given analyzer
        try:
            selected_data[analyzer.uid] = analyzer.select_simulation_data(data, simulation)
        except:
            set_exception(step="data processing", info={"Simulation": simulation, "Analyzer": analyzer.uid},
                          cache=cache)

            return

    # Store in the cache
    cache.set(simulation.id, selected_data)
