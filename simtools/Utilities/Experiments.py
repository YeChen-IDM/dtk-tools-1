import getpass

from simtools.DataAccess.DataStore import DataStore
from simtools.DataAccess.Schema import Experiment, Simulation
from simtools.SetupParser import SetupParser
from simtools.Utilities.COMPSUtilities import get_experiment_by_id, COMPS_login, get_experiments_by_name
from simtools.Utilities.Encoding import cast_number
from simtools.Utilities.General import init_logging, utc_to_local

max_exp_name_len = 255

logger = init_logging('Utils')


def validate_exp_name(exp_name):
    if len(exp_name) > max_exp_name_len:
        logger.fatal(
            "The experiment name '%s' exceeds the max length %d (%d), please adjust your experiment name. Exiting..." %
            (exp_name, max_exp_name_len, len(exp_name)))
        return False
    else:
        return True


def retrieve_object(obj_id, obj_type, sync_if_missing=True, verbose=False, force_update=False):
    """
        Retrieve an object (Experiment, Simulation) in the local database based on its id.
        Can call a sync if missing if the flag is true
        :param obj_id: Id of the object to retrieve
        :param obj_type: Type of the object to retrieve
        :param sync_if_missing: Should we try to sync if not present?
        :return: The experiment found

        #TODO: SHould also support suites
        """
    typename = obj_type.__name__
    if not obj_id: raise ValueError("Trying to retrieve an object (%s) without providing an ID" % typename)

    # Try a hit in the DB
    if obj_type == Experiment:
        obj = DataStore.get_experiment(obj_id)
    elif obj_type == Simulation:
        obj = DataStore.get_simulation(obj_id)

    if obj:
        # If we have an experiment and we want to force the update -> delete it
        if not force_update:
            return obj
        else:
            if obj_type == Experiment:
                DataStore.delete_experiment(obj_id)
            elif obj_type == Simulation:
                DataStore.delete_simulation(obj_id)

    if not sync_if_missing:
        raise Exception('%s %s not found in the local database and sync disabled.' % (typename, obj_id))

    logger.info('%s with id %s not found in local database, trying sync.' % (typename, obj_id))
    with SetupParser.TemporarySetup(temporary_block='HPC') as sp:
        endpoint = sp.get('server_endpoint')

    if obj_type == Experiment:
        obj = COMPS_experiment_to_local_db(obj_id, endpoint, verbose)
    elif obj_type == Simulation:
        obj =None

    if obj: return obj
    raise Exception("%s '%s' could not be retrieved." % (typename, obj_id))

def retrieve_experiment(exp_id, sync_if_missing=True, verbose=False, force_update=False):
    """
    Retrieve an experiment in the local database based on its id.
    Can call a sync if missing if the flag is true.
    :param exp_id: Id of the experiment to retrieve
    :param sync_if_missing: Should we try to sync if not present?
    :return: The experiment found
    """
    if not exp_id:
        raise Exception("Trying to retrieve an experiment without providing an experiment ID")
    from uuid import UUID
    if isinstance(exp_id, UUID):
        exp_id = str(exp_id)

    # If we don't force the update -> look first in the DB
    exp = DataStore.get_experiment(exp_id)
    if not exp:
        exp = DataStore.get_most_recent_experiment(exp_id)

    if exp:
        # If we have an experiment and we want to force the update -> delete it
        if force_update:
            DataStore.delete_experiment(exp)
        else:
            return exp

    if not sync_if_missing:
        raise Exception('Experiment %s not found in the local database and sync disabled.' % exp_id)

    logger.info('Experiment with id %s not found in local database, trying sync.' % exp_id)
    with SetupParser.TemporarySetup(temporary_block='HPC') as sp:
        endpoint = sp.get('server_endpoint')

    exp = COMPS_experiment_to_local_db(exp_id, endpoint, verbose=verbose)

    if exp:
        return exp
    raise Exception("Experiment '%s' could not be retrieved." % exp_id)


def retrieve_simulation(sim_id, sync_if_missing=True, verbose=False, force_update=False):
    """
    Retrieve a simulation in the local database based on its id.
    Can call a sync if missing if the flag is true.
    :param sim_id: Id of the simulation to retrieve
    :param sync_if_missing: Should we try to sync if not present?
    :return: The simulation found
    """
    from simtools.Utilities.COMPSUtilities import get_simulation_by_id

    if not sim_id: raise Exception("Trying to retrieve a simulation without providing an simulation ID")
    from uuid import UUID
    if isinstance(sim_id, UUID): sim_id = str(sim_id)

    # If we dont force the update -> look first in the DB
    sim = DataStore.get_simulation(sim_id)
    if sim:
        # If we have a simulation and we want to force the update -> delete it
        if not force_update:
            return sim
        else:
            DataStore.delete_simulation(sim)

    if not sync_if_missing:
        raise Exception('Simulation %s not found in the local database and sync disabled.' % sim_id)

    logger.info('Simulation with id %s not found in local database, trying sync.' % sim_id)

    csim = get_simulation_by_id(sim_id)
    if csim:
        with SetupParser.TemporarySetup(temporary_block='HPC') as sp:
            endpoint = sp.get('server_endpoint')
        COMPS_experiment_to_local_db(csim.experiment_id, endpoint, verbose)

    sim = DataStore.get_simulation(sim_id)
    if sim: return sim

    raise Exception("Simulation '%s' could not be retrieved." % sim_id)


def COMPS_experiment_to_local_db(id_or_name, endpoint, verbose=False, save_new_experiment=True):
    """
    Return a DB object representing an experiment coming from COMPS.
    This function saves the newly retrieved experiment in the DB by default but this behavior can be changed by switching
    the save_new_experiment parameter allowing to return an experiment object and save later with a batch for example.
    :param id_or_name:
    :param endpoint:
    :param verbose:
    :param save_new_experiment:
    :return:
    """
    # Ensure exp_id is a string
    id_or_name = str(id_or_name)

    # First try to find the experiment locally by experiment id
    experiment = DataStore.get_experiment(id_or_name)
    if experiment and verbose:
        print('Found experiment locally.')

    if experiment and experiment.is_done():
        if verbose:
            print("Experiment ('%s') already exists in local db." % id_or_name)
        # Do not bother with finished experiments
        return None

    # second, try to find the experiment from COMPS by experiment id
    exp_comps = None
    COMPS_login(endpoint)
    from COMPS.Data import QueryCriteria
    query_criteria = QueryCriteria().select_children('tags')
    try:
        exp_comps = get_experiment_by_id(id_or_name, query_criteria)
    except ValueError as e:
        exp_comps = None
    if exp_comps is not None and verbose:
        print('Found experiment in COMPS by id.')

    # finally, try to find the experiment from comps by experiment name
    if exp_comps is None:
        try:
            exp_comps = get_experiments_by_name(id_or_name, user=getpass.getuser())[-1]
        except IndexError as e:
            exp_comps = None
        if exp_comps is not None and verbose:
            print('Found experiment in COMPS by name.')

    if exp_comps is None:
        if verbose:
            print("The experiment ('%s') doesn't exist in COMPS." % id_or_name)
        return None

    # Case: experiment doesn't exist in local db
    if not experiment:
        # Cast the creation_date
        experiment = DataStore.create_experiment(exp_id=str(exp_comps.id),
                                                 suite_id=str(exp_comps.suite_id) if exp_comps.suite_id else None,
                                                 exp_name=exp_comps.name,
                                                 tags=exp_comps.tags,
                                                 date_created=utc_to_local(exp_comps.date_created).replace(tzinfo=None),
                                                 location='HPC',
                                                 selected_block='HPC',
                                                 endpoint=endpoint)

    # Note: experiment may be new or comes from local db
    # Get associated simulations of the experiment
    sims = exp_comps.get_simulations(QueryCriteria().select(['id', 'state', 'date_created']).select_children('tags'))

    # Skip empty experiments or experiments that have the same number of sims
    if len(sims) == 0 or len(sims) == len(experiment.simulations):
        if verbose:
            if len(sims) == 0:
                print("Skip empty experiment ('%s')." % id_or_name)
            elif len(sims) == len(experiment.simulations):
                print("Skip experiment ('%s') since local one has the same number of simulations." % id_or_name)
        return None

    # Go through the sims and create them
    for sim in sims:
        # Cast the simulation tags

        # Create the simulation
        simulation = DataStore.create_simulation(id=str(sim.id),
                                                 status=sim.state, # this is already a SimulationState object
                                                 tags={tag:cast_number(val) for tag,val in sim.tags.items()},
                                                 date_created=utc_to_local(sim.date_created).replace(tzinfo=None))
        # Add to the experiment
        experiment.simulations.append(simulation)

    # Save it to the DB
    if save_new_experiment: DataStore.save_experiment(experiment, verbose=verbose)

    return experiment
