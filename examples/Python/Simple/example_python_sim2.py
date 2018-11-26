from models.python.PythonModelManager import PythonModelManager
from simtools.AssetManager.FileList import FileList
from simtools.SetupParser import SetupParser

# give name to the experiment
exp_name = "ExamplePythonSim2"

# user file to run
user_model = "Assets\\Hello_model.py"

# add possible user files
asset_files = FileList(root="Assets", files_in_root=["Hello_model.py"])
# asset_files = FileList(root="Assets", recursive=True, max_depth=10)


if __name__ == "__main__":
    SetupParser.default_block = 'HPC'
    SetupParser.init()

    pmm = PythonModelManager(user_model=user_model, asset_files=asset_files, exp_name=exp_name)
    pmm.execute(True)

