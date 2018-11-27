import inspect
import os

from simtools.AssetManager.FileList import FileList
from simtools.Managers.WorkItemManager import WorkItemManager


class SSMTAnalysis:

    def __init__(self, experiment_ids, analyzers, analysis_name, additional_files=None):
        self.experiment_ids = experiment_ids
        self.analyzers = analyzers
        self.analysis_name = analysis_name
        self.additional_files = additional_files or FileList()

    def analyze(self):
        # Add the analyze_ssmt.py file to the collection
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.additional_files.add_file(os.path.join(dir_path, "analyze_ssmt.py"))

        # If there is a simtools.ini, send it along
        if os.path.exists(os.path.join(os.getcwd(), "simtools.ini")):
            self.additional_files.add_file(os.path.join(os.getcwd(), "simtools.ini"))

        # Add all the analyzers files
        for a in self.analyzers:
            self.additional_files.add_file(inspect.getfile(a.__class__))

        # Create the command
        command = "python analyze_ssmt.py"
        # Add the experiments
        command += " {}".format(",".join(self.experiment_ids))
        # Add the analyzers
        command += " {}".format(",".join(f"{inspect.getmodulename(inspect.getfile(s.__class__))}.{s.__class__.__name__}"
                                         for s in self.analyzers))

        wim = WorkItemManager(item_name=self.analysis_name, command=command, user_files=self.additional_files,
                              related_experiments=self.experiment_ids)
        wim.execute()
