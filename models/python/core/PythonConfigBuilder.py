import json
import os

from models.generic.GenericConfigBuilder import GenericConfigBuilder
from simtools.AssetManager.FileList import FileList


class PythonConfigBuilder(GenericConfigBuilder):

    def __init__(self, python_file, python_command="C:\\Python36\\python.exe", input_files=None, write_python_file=True):
        super().__init__(command="run.bat")
        self.sim_config = {}
        self.python_file = python_file
        self.write_python_file = write_python_file
        if self.write_python_file:
            self.python_file_contents = open(python_file).read()
            self.python_file_basename = os.path.basename(self.python_file)
        else:
            self.python_file_basename = python_file
        self.python_command = python_command
        self.input_files = input_files or FileList()

    def file_writer(self, write_fn):
        """
        Dump all the files needed for the simulation in the simulation directory.
        This includes:

        * The model file
        * The config file

        Args:
            write_fn: The function that will write the files. This function needs to take a file name and a content.
        """
        write_fn("run.bat", "{} -E {}".format(self.python_command, self.python_file_basename))

        if self.sim_config:
            write_fn("config.json", json.dumps(self.sim_config))

        if self.write_python_file:
            write_fn(self.python_file_basename, self.python_file_contents)

        for input_file in self.input_files:
            if input_file.file_name.lower() in ("comps_log.log", "simtools.ini", "stdout.txt", "stderr.txt"): continue
            write_fn(input_file.file_name, open(input_file.absolute_path).read())





