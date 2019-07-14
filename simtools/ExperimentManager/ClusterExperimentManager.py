import sys
from tempfile import TemporaryDirectory

from COMPS.Data.Simulation import SimulationState

from simtools.DataAccess.DataStore import DataStore
from simtools.ExperimentManager.BaseExperimentManager import BaseExperimentManager
from simtools.SetupParser import SetupParser
from simtools.SimulationCreator.ClusterSimulationsCreator import ClusterSimulationCreator
from simtools.Utilities.Experiments import validate_exp_name
from simtools.Utilities.General import init_logging

logger = init_logging("ClusterExperimentManager")

import os
import re
import shutil
from datetime import datetime

script_template = """#/bin/bash
#SBATCH --job-name {job_name}  # A single job name for the array
#SBATCH --partition={partition} # Which partition to use
#SBATCH --mem-per-cpu={memory} # Memory per core
#SBATCH --time={time_limit}    # Time limit
#SBATCH --tasks=1
#SBATCH --cpus-per-task={cpu_per_task}
#SBATCH --nodes={nodes}        # How many nodes to request.
#SBATCH --mail-type=ALL 
#SBATCH --mail-user={email}
#SBATCH --array=0-{folders_length} 
folders = ({folders})
current_folder = ${{folders[$SLURM_ARRAY_TASK_ID]}}
cd $current_folder
../Assets/Eradication -C ./config.json -I ../Assets
"""

# Requires the following type of section in simtools.ini
# [CLUSTERMODE]
# type = CLUSTER
# # Path where the simulation outputs will be stored
# sim_root = P:\Eradication\simulations
#
# # Path for the model to find the input files
# input_root = P:\Projects\dtk-tools-br\examples\inputs
#
# # Path where a 'reporter_plugins' folder containing the needed DLLs
# dll_root = P:\Projects\dtk-tools-br\examples\inputs\dlls
#
# # Path to the model executable
# exe_path = P:\Projects\dtk-tools-br\examples\inputs\Eradication.exe
#
# # Resources request
# nodes = 1
# cpu_per_task = 1
# memory_per_cpu = 8GB
#
# # Which email to send the notifications to
# notification_email = braybaud@intven.com
#
# # NYU partition to use
# partition = cpu_short
#
# # Limit time on this job hrs:min:sec
# time_limit = 10:00:00


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
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(os.path.join(experiment_path, "Assets"), exist_ok=True)

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
        return ClusterSimulationCreator(config_builder=self.config_builder,
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

        # Dump the shell script
        self.dump_shell_script()

        # Package
        self.package()

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

    def dump_shell_script(self):
        folders = " ".join(s.id for s in self.experiment.simulations)
        with open(os.path.join(self.experiment.get_path(), "run_script.sh"), 'w') as fp:
            fp.write(script_template.format(job_name=self.experiment.id,
                                            partition=SetupParser.get("partition"),
                                            memory=SetupParser.get("memory_per_cpu"),
                                            time_limit=SetupParser.get("time_limit"),
                                            cpu_per_task=SetupParser.get("cpu_per_task"),
                                            nodes=SetupParser.get("nodes"),
                                            email=SetupParser.get("notification_email"),
                                            folders_length=len(self.experiment.simulations)-1,
                                            folders=folders))

    def package(self):
        with TemporaryDirectory() as tmpdir:
            zipfile = os.path.join(tmpdir, self.experiment.id)
            shutil.make_archive(zipfile, 'zip', self.experiment.get_path())
            shutil.move(f"{zipfile}.zip", self.experiment.get_path())

    def wait_for_finished(self, verbose=False, sleep_time=5):
        print("CLUSTER MODE")
        print("----------------------------------")
        print(f"- Copy the {self.experiment.id}.zip file from {self.experiment.get_path()} to scratch")
        print(f"- Connect to prince (either through bastion or directly if on site)")
        print(f"- Unpack the {self.experiment.id}.zip file")
        print(f"- Run sbatch run_script.sh from the {self.experiment.id} folder")
        print(f"- Wait for the jobs to complete (you will receive an email when done)")
        print(f"- Copy the jobs with their outputs to {self.experiment.get_path()}")
        print("Press Enter when all those steps are done to continue")
        sys.stdin.readline()
        print("----------------------------------")

        for s in self.experiment.simulations:
            s.status = SimulationState.Succeeded
            DataStore.save_simulation(s)

    def succeeded(self):
        return True

