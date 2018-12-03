import os
import inspect
import pickle
from simtools.AssetManager.FileList import FileList
from simtools.Managers.WorkItemManager import WorkItemManager


class SSMTAnalysis:

    def __init__(self, experiment_ids, analyzers, analysis_name, additional_files=None, asset_collection_id=None, asset_files=FileList()):
        self.experiment_ids = experiment_ids
        self.analyzers = analyzers
        self.analysis_name = analysis_name
        self.additional_files = additional_files or FileList()
        self.asset_collection_id = asset_collection_id
        self.asset_files = asset_files


    def analyze(self):
        # Add the analyze_ssmt.py file to the collection
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.additional_files.add_file(os.path.join(dir_path, "analyze_ssmt.py"))

        # If there is a simtools.ini, send it along
        if os.path.exists(os.path.join(os.getcwd(), "simtools.ini")):
            self.additional_files.add_file(os.path.join(os.getcwd(), "simtools.ini"))

        # Build analyzer args pickle files
        analyzer_args = {}
        temp_file = "analyzer_args.pkl"
        for a in self.analyzers:
            sig = inspect.signature(a.__class__)
            args = {k: getattr(a, k) for k in sig.parameters.keys()}
            analyzer_args[f"{inspect.getmodulename(inspect.getfile(a.__class__))}.{a.__class__.__name__}"] = args

        pickle.dump(analyzer_args, open(temp_file, 'wb'))

        # Add analyzer args pickle as additional file
        self.additional_files.add_file(temp_file)

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

        # Create WorkItemManager
        wim = WorkItemManager(item_name=self.analysis_name, command=command, user_files=self.additional_files,
                              asset_collection_id=self.asset_collection_id, asset_files=self.asset_files,
                              related_experiments=self.experiment_ids)

        wim.execute()

        # remove temp file
        os.remove(temp_file)
