from simtools.AssetManager.FileList import FileList
from models.python.core.PythonConfigBuilder import PythonConfigBuilder
from simtools.AssetManager.AssetCollection import AssetCollection
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.ModBuilder import ModBuilder, ModFn
from simtools.Utilities.General import batch_list


class PythonModelManager(object):

    def __init__(self, user_model, asset_collection_id=None, asset_files=FileList(), user_files=FileList(), configs=[], chunk_size=1, exp_name='PythonTest'):
        self.user_model = user_model
        self.asset_collection_id = asset_collection_id
        self.asset_files = asset_files
        self.user_files = user_files
        self.configs = configs
        self.chunk_size = chunk_size
        self.exp_name = exp_name

    def add_asset_file(self, asset_file):
        pass

    def add_user_file(self, user_file):
        pass

    @staticmethod
    def set_config(cb, config):
        # cb.sim_config = config            # list
        cb.sim_config = {'config': config}  # dict
        return {"config": config}

    def execute(self, check_status=True):

        # Create config builder
        cb = PythonConfigBuilder(self.user_model, write_python_file=False)

        # Add user files
        for f in self.user_files:
            # cb.input_files.add_file(f.absolute_path)
            cb.input_files.add_asset_file(f)

        # Set the master collection id
        if not self.asset_collection_id:
            # Create a collection with everything that is in Assets
            if len(self.asset_files.files) > 0:
                ac = AssetCollection(local_files=self.asset_files)
                ac.prepare("HPC")
                cb.set_collection_id(ac.collection_id)
                print("ac.collection_id: {}".format(ac.collection_id))
        else:
            cb.set_collection_id(self.asset_collection_id)

        # Create ModBuilder for simulation creation
        if not self.configs:
            builder = None
        else:
            builder = ModBuilder.from_combos(
                # [ModFn(set_config, config) for config in configs]
                [ModFn(self.set_config, config) for config in batch_list(self.configs, self.chunk_size)]
            )

        # Create an experiment manager
        exp_manager = ExperimentManagerFactory.from_cb(cb)
        exp_manager.run_simulations(exp_name=self.exp_name, exp_builder=builder)

        # Wait for the simulations to be done
        if check_status:
            exp_manager.wait_for_finished(verbose=True)


