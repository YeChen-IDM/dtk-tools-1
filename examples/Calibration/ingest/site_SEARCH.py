import importlib
import logging
from calibtool.CalibSite import CalibSite

logger = logging.getLogger(__name__)

class SEARCHCalibSite(CalibSite):
    metadata = {}

    def __init__(self, **kwargs):
        # required kwargs first
        analyzers = kwargs.get('analyzers', None)
        if not analyzers:
            raise Exception('An analyzer dictionary must be provided to the \'analyzers\' argument.')

        # reference_data must be supplied as a kwarg and is simply stored for direct use by analyzers
        self.reference_data = kwargs.get('reference_data', None)
        if not self.reference_data:
            raise Exception('Obs/reference data object must be provided to the \'reference_data\' argument.')

        # optional kwargs
        force_apply = kwargs.get('force_apply', False)
        max_sims_per_scenario = kwargs.get('max_sims_per_scenario', -1)

        self.analyzers = []
        for analyzer, weight in analyzers.items():
            AnalyzerClass = getattr(importlib.import_module('analyzers.%s'%analyzer), analyzer)

            self.analyzers.append(
                AnalyzerClass(
                    self,
                    weight = weight,

                    force_apply = force_apply,
                    max_sims_per_scenario = max_sims_per_scenario,

                    reference_year = 2013.5,
                    reference_population = 132187,
                    age_min = 15,
                    age_max = 65,

                    node_map = {
                            1: "ControlLowPrevalence",
                            2: "ControlMediumPrevalence",
                            3: "ControlHighPrevalence",
                            4: "InterventionLowPrevalence",
                            5: "InterventionMediumPrevalence",
                            6: "InterventionHighPrevalence"
                    },

                    basedir = '.',
                    fig_format = 'png',
                    fig_dpi = 600,
                    verbose = True
                )
            )

        # Must come at the end:
        super(SEARCHCalibSite, self).__init__('SEARCH')

    def get_setup_functions(self):
        return [ ]

    def get_analyzers(self):
        return self.analyzers
