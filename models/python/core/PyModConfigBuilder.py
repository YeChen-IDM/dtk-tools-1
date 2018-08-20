from models.generic.GenericConfigBuilder import GenericConfigBuilder
from simtools.AssetManager.FileList import FileList


class PyModConfigBuilder(GenericConfigBuilder):

    def __init__(self, python_file, input_files=None, additional_assets=None):
        super().__init__(command="run.bat")

        self.input_files = input_files or FileList()
        self.assets.experiment_files = additional_assets or FileList()
        self.python_file_basename, self.python_file_contents = self.open_file(python_file)

    def file_writer(self, write_fn):
        """
        Dump all the files needed for the simulation in the simulation directory.
        Args:
            write_fn: The function that will write the files. This function needs to take a file name and a content.
        """
        write_fn(self.python_file_basename, self.python_file_contents)
        for input_file in self.input_files:
            if input_file.file_name.lower() in ("comps_log.log", "simtools.ini", "stdout.txt", "stderr.txt"): continue
            write_fn(input_file.file_name, open(input_file.absolute_path).read())

        write_fn("run.bat", "C:\\Python36\\python.exe -E {}".format(self.python_file_basename))
