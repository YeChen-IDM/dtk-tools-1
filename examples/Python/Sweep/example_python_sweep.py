from simtools.SetupParser import SetupParser
from simtools.AssetManager.FileList import FileList
from models.python.PythonModelManager import PythonModelManager

# give name to the experiment
exp_name = "ExamplePythonSweep1"

# user file to run
user_model = "Hello_model.py"

# add possible asset files
# asset_files = FileList(root="Assets", recursive=True, max_depth=10)

# add possible user files
user_files = FileList(root=".", files_in_root=["Hello_model.py"])

# set simulation configs
configs = [{'A': 1}, {'A': 2}, {'A': 3}, {'A': 4}, {'A': 5}, {'A': 6}, {'A': 7}]
chunk_size = 2


if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    pmm = PythonModelManager(user_model=user_model, user_files=user_files, configs=configs, chunk_size=chunk_size, exp_name=exp_name)
    pmm.execute(True)
