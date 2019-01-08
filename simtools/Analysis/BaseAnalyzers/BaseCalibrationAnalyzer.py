import json

from simtools.Analysis.BaseAnalyzers import BaseAnalyzer
from simtools.Utilities.Encoding import GeneralEncoder


class BaseCalibrationAnalyzer(BaseAnalyzer):
    def __init__(self, uid=None, working_dir=None, parse=True, need_dir_map=False, filenames=None,
                 reference_data=None, weight=1):
        super().__init__(uid=uid, working_dir=working_dir, parse=parse, need_dir_map=need_dir_map, filenames=filenames)
        self.reference_data = reference_data
        self.weight = weight

    def cache(self):
        return json.dumps(self, cls=GeneralEncoder)