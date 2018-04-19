import os

from simtools.SimConfigBuilder import SimConfigBuilder


class PythonConfigBuilder(SimConfigBuilder):

    def __init__(self, python_file):
        super(PythonConfigBuilder, self).__init__()
        self.python_file = python_file
        self.python_file_contents = open(python_file).read()
        self.python_file_basename = os.path.basename(self.python_file)

    def get_commandline(self):
        """
        Get the complete command line to run the simulations of this experiment.
        Returns:
            The :py:class:`CommandlineGenerator` object created with the correct paths

        """
        from simtools.Utilities.General import CommandlineGenerator
        return CommandlineGenerator("python {}".format(self.python_file_basename),{},[])

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



