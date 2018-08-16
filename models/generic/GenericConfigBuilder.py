import os
from abc import ABC

from simtools.SimConfigBuilder import SimConfigBuilder


class GenericConfigBuilder(SimConfigBuilder, ABC):

    def __init__(self, command, options=None, parameters=None):
        super(GenericConfigBuilder, self).__init__()
        self.command = command
        self.options = options or {}
        self.parameters = parameters or []

    def get_commandline(self):
        from simtools.Utilities.General import CommandlineGenerator
        return CommandlineGenerator(self.command, self.options, self.parameters)

    def open_file(self, filename):
        contents = open(filename).read()
        basename = os.path.basename(filename)

        return basename, contents

    def get_dll_paths_for_asset_manager(self):
        return []

    def get_input_file_paths(self):
        return []



