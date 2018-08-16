from models.python.core.PythonConfigBuilder import PythonConfigBuilder
from simtools.AssetManager.FileList import FileList


class PyModConfigBuilder(PythonConfigBuilder):

    def __init__(self, python_file, input_files=None, additional_assets=None):
        super().__init__(python_file, python_comand=r"C:\Python3.6\python.exe")
        self.input_files = input_files or FileList()
        self.assets.experiment_files = additional_assets or FileList()

    def file_writer(self, write_fn):
        """
        Dump all the files needed for the simulation in the simulation directory.
        Args:
            write_fn: The function that will write the files. This function needs to take a file name and a content.
        """
        write_fn(self.python_file_basename, self.python_file_contents)
        for input_file in self.input_files:
            write_fn(input_file.file_name, open(input_file.absolute_path).read())
