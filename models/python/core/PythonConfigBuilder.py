import os

from models.generic.GenericConfigBuilder import GenericConfigBuilder


class PythonConfigBuilder(GenericConfigBuilder):

    def __init__(self, python_file, python_command="python"):
        super().__init__(python_command, parameters=[python_file])
        self.python_file = python_file
        self.python_file_contents = open(python_file).read()
        self.python_file_basename = os.path.basename(self.python_file)
        self.python_command = python_command

    def file_writer(self, write_fn):
        """
        Dump all the files needed for the simulation in the simulation directory.
        This includes:

        * The model file
        * The config file

        Args:
            write_fn: The function that will write the files. This function needs to take a file name and a content.
        """
        write_fn(self.python_file_basename, self.python_file_contents)

    def get_dll_paths_for_asset_manager(self):
        return []

    def get_input_file_paths(self):
        return []



