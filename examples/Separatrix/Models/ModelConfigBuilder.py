import json
import os
from simtools.SimConfigBuilder import SimConfigBuilder
from simtools.Utilities.Encoding import NumpyEncoder


class ModelConfigBuilder(SimConfigBuilder):

    def __init__(self, model='Assets/ModelProcessor.py'):
        super(ModelConfigBuilder, self).__init__()
        self.model = model

    def set_config_param(self, param, value):
        self.config[param] = value

    def get_config_param(self, param):
        return self.config[param] if param in self.config else None

    def get_commandline(self):
        """
        Get the complete command line to run the simulations of this experiment.
        Returns:
            The :py:class:`CommandlineGenerator` object created with the correct paths

        """
        from simtools.Utilities.General import CommandlineGenerator
        from simtools.SetupParser import SetupParser

        if SetupParser.get('type') == 'LOCAL':
            exe_path = self.model
        else:
            exe_path = self.model

        return CommandlineGenerator("Python {}".format(exe_path), None, [])

    def file_writer(self, write_fn):
        """
        Dump all the files needed for the simulation in the simulation directory.
        This includes:

        * The config file

        Args:
            write_fn: The function that will write the files. This function needs to take a file name and a content.
        """
        # Handle the config
        if self.human_readability:
            config = json.dumps(self.config, sort_keys=True, indent=3, cls=NumpyEncoder).strip('"')
        else:
            config = json.dumps(self.config, sort_keys=True, cls=NumpyEncoder).strip('"')

        write_fn('config.json', config)

    def get_dll_paths_for_asset_manager(self):
        from simtools.AssetManager.FileList import FileList
        fl = FileList(root=self.assets.dll_root, recursive=True)
        return [f.absolute_path for f in fl.files if f.file_name != self.assets.exe_path]

    def get_input_file_paths(self):
        return []


