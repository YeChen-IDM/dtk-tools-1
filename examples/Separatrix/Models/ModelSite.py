from calibtool.CalibSite import CalibSite
from examples.Separatrix.Models.ModelAnalyzer import ModelAnalyzer


class ModelSite(CalibSite):
    def __init__(self):
        super().__init__("ModelSite")

    def get_reference_data(self, reference_type=None):
        ref = []
        return ref

    def get_analyzers(self):
        return ModelAnalyzer(self.get_reference_data()),

    def get_setup_functions(self):
        return []
