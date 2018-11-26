from models.python.PythonModelManager import PythonModelManager
from simtools.AssetManager.FileList import FileList
from simtools.SetupParser import SetupParser

# give name to the experiment
exp_name = "ExamplePythonSim1-1"

# user file to run
user_model = "Hello_model.py"

# add possible user files
user_files = FileList(root=".", files_in_root=["Hello_model.py"])


if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    pmm = PythonModelManager(user_model=user_model, user_files=user_files, exp_name=exp_name)
    pmm.execute(True)

