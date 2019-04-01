from simtools.ExperimentManager.BaseExperimentManager import BaseExperimentManager
from simtools.Utilities.Experiments import validate_exp_name
from simtools.Utilities.General import init_logging

logger = init_logging("ClusterExperimentManager")

import os
import re
import shutil
from datetime import datetime
from simtools.SimulationCreator.LocalSimulationCreator import LocalSimulationCreator


class ClusterExperimentManager(BaseExperimentManager):
    """
    Manages the creation, submission, status, parsing, and analysis
    of cluster experiments
    """
    location = 'CLUSTER'

    def __init__(self, experiment, config_builder):
        self.experiment = experiment

        BaseExperimentManager.__init__(self, experiment, config_builder)

    def commission_simulations(self):
        pass

    def create_experiment(self, experiment_name, experiment_id=None, suite_id=None):
        experiment_name = self.clean_experiment_name(experiment_name)

        # Create a unique id
        experiment_id = re.sub('[ :.-]', '_', str(datetime.now()))
        logger.info("Creating exp_id = " + experiment_id)

        # Create the experiment in the base class
        super(ClusterExperimentManager, self).create_experiment(experiment_name, experiment_id, suite_id)

        # Get the path and create it if needed
        experiment_path = self.experiment.get_path()
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

    @staticmethod
    def create_suite(suite_name):
        suite_id = suite_name + '_' + re.sub('[ :.-]', '_', str(datetime.now()))
        logger.info("Creating suite_id = " + suite_id)
        return suite_id

    def hard_delete(self):
        """
        Delete experiment and output data.
        """
        # Delete local simulation data.
        exp_path = self.experiment.get_path()
        if os.path.exists(exp_path):
            try:
                shutil.rmtree(exp_path)
            except Exception as e:
                print("Could not delete path: {}\nReason: {}".format(exp_path, e))

        # Delete in the DB
        from simtools.DataAccess.DataStore import DataStore
        DataStore.delete_experiment(self.experiment)

    def cancel_experiment(self):
        pass

    def kill_simulation(self, simulation):
        pass

    def get_simulation_creator(self, work_queue):
        return LocalSimulationCreator(config_builder=self.config_builder,
                                      initial_tags=self.exp_builder.tags,
                                      work_queue=work_queue,
                                      experiment=self.experiment,
                                      cache=self.cache)

    def run_simulations(self, config_builder=None, exp_name='test', exp_builder=None,
                        suite_id=None, blocking=False, quiet=False, experiment_tags=None):
        """
        Create an experiment with simulations modified according to the specified experiment builder.
        Commission simulations and cache meta-data to local file.
        :assets: A SimulationAssets object if not None (COMPS-land needed for AssetManager)
        """
        # Check experiment name as early as possible
        if not validate_exp_name(exp_name):
            exit()

        # Store the config_builder if passed
        self.config_builder = config_builder or self.config_builder

        # Get the assets from the config builder
        # We just want to check the input files at this point even though it may change later
        self.assets = self.config_builder.get_assets()

        # Check input files existence
        if not self.config_builder.ignore_missing:
            if not self.validate_input_files(needed_file_paths=self.config_builder.get_input_file_paths(),
                                             message="config.json"):
                exit()
            if not self.validate_input_files(needed_file_paths=self.config_builder.get_dll_paths_for_asset_manager(),
                                             message="emodules_map.json"):
                exit()

        # Set the appropriate command line
        self.commandline = self.config_builder.get_commandline()

        # Set the tags
        self.experiment_tags.update(experiment_tags or {})

        # Create the simulations
        self.create_simulations(exp_name=exp_name, exp_builder=exp_builder, suite_id=suite_id, verbose=not quiet)

        # Dump the meta-data file
        self.dump_metadata()

        # Dump the required assets
        self.dump_assets()

    def dump_metadata(self):
        metadata = {
            "experiment": self.experiment.exp_name,
            "original_path": self.experiment.get_path(),
            "experiment_tags": self.experiment.tags,
            "simulations": [
                {
                    "id": s.id,
                    "tags": s.tags
                } for s in self.experiment.simulations
            ]
        }
        import json
        with open(os.path.join(self.experiment.get_path(), "metadata.json"), 'w') as fp:
            json.dump(metadata, fp, indent=4)

    def dump_assets(self):
        assets_path = os.path.join(self.experiment.get_path(), "Assets")
        os.mkdir(assets_path)

        from simtools.AssetManager.SimulationAssets import SimulationAssets
        for collection in SimulationAssets.COLLECTION_TYPES:
            for file in self.assets._gather_files(self.config_builder, collection) or []:
                # For each file found in the assets, check if the folder exist
                folder = os.path.join(assets_path, file.relative_path or "")
                if not os.path.exists(folder):
                    os.mkdir(folder)

                # And copy it there
                shutil.copyfile(file.absolute_path, os.path.join(folder, file.file_name))

    def wait_for_finished(self, verbose=False, sleep_time=5):
        return

    def succeeded(self):
        return True
