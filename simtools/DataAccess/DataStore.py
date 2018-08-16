from simtools.DataAccess.BatchDataStore import BatchDataStore
from simtools.DataAccess.ExperimentDataStore import ExperimentDataStore
from simtools.DataAccess.SettingsDataStore import SettingsDataStore
from simtools.DataAccess.SimulationDataStore import SimulationDataStore


class DataStore(SimulationDataStore, ExperimentDataStore, SettingsDataStore, BatchDataStore):
    """
    Class holding static methods to abstract the access to the database.
    """

    @classmethod
    def list_leftover(cls, suite_ids, exp_ids):
        """
        List those experiments which are associated with suite_id and not in exp_ids
        suite_ids: list of suite ids
        exp_ids: list of experiment ids
        """
        exp_list = cls.get_experiments_by_suite(suite_ids)
        exp_list_ids = [exp.exp_id for exp in exp_list]

        # Calculate orphans
        exp_orphan_ids = list(set(exp_list_ids) - set(exp_ids))
        exp_orphan_list = [exp for exp in exp_list if exp.exp_id in exp_orphan_ids]

        return exp_orphan_list

