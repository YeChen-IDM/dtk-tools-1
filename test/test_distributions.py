import pandas as pd
import unittest

from dtk.utils.observations.BaseDistribution import BaseDistribution
from dtk.utils.observations.BetaDistribution import BetaDistribution
from dtk.utils.observations.GaussianDistribution import GaussianDistribution
from dtk.utils.observations.PopulationObs import PopulationObs

class TestDistributions(unittest.TestCase):

    # BaseDistribution initialization test
    def test_instantiation_from_string(self):
        distribution = BaseDistribution.from_string(distribution_name='Gaussian')
        self.assertTrue(isinstance(distribution, GaussianDistribution))
        self.assertRaises(BaseDistribution.UnknownDistributionException,
                          BaseDistribution.from_string, distribution_name='Tibia')

    #
    # BetaDistribution tests
    #

    def test_catch_invalid_counts(self):
        # count channel is missing
        data = [{'some_value': 1, PopulationObs.WEIGHT_CHANNEL: 1}, {'some_value': 2, PopulationObs.WEIGHT_CHANNEL: 1}]
        dfw = PopulationObs(dataframe=pd.DataFrame(data))
        distribution = BetaDistribution()
        self.assertRaises(BetaDistribution.InvalidEffectiveCountException, distribution.prepare,
                          dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                          weight_channel=PopulationObs.WEIGHT_CHANNEL)

        # count channel is present but has an invalid value
        df = pd.DataFrame([{'some_value': 3, BetaDistribution.COUNT_CHANNEL: 0, PopulationObs.WEIGHT_CHANNEL: 4},
                           {'some_value': 4.1, BetaDistribution.COUNT_CHANNEL: 50.1, PopulationObs.WEIGHT_CHANNEL: 3}])
        dfw = PopulationObs(dataframe=df)
        distribution = BetaDistribution()
        self.assertRaises(BetaDistribution.InvalidEffectiveCountException, distribution.prepare,
                          dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                          weight_channel=PopulationObs.WEIGHT_CHANNEL)

        # count channel is present and all valid
        df = pd.DataFrame([{'some_value': 6.2, BetaDistribution.COUNT_CHANNEL: 1, PopulationObs.WEIGHT_CHANNEL: 1},
                           {'some_value': 7.3, BetaDistribution.COUNT_CHANNEL: 50.1, PopulationObs.WEIGHT_CHANNEL: 2}])
        dfw = PopulationObs(dataframe=df)
        distribution = BetaDistribution()
        distribution.prepare(dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                             weight_channel=PopulationObs.WEIGHT_CHANNEL)

    def test_fail_if_Count_channel_used(self):
        # count channel is present and all valid
        df = pd.DataFrame([{'some_value': 6.2, 'Count': 1, PopulationObs.WEIGHT_CHANNEL: 1},
                           {'some_value': 7.3, 'Count': 50.1, PopulationObs.WEIGHT_CHANNEL: 2}])
        dfw = PopulationObs(dataframe=df)
        distribution = BetaDistribution()
        self.assertRaises(BetaDistribution.InvalidCountChannelException,
                          distribution.prepare,
                          dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                          weight_channel=PopulationObs.WEIGHT_CHANNEL)

    #
    # GaussianDistribution tests
    #

    def test_catch_invalid_uncertainties(self):
        # uncertainty channel is missing
        dfw = PopulationObs(dataframe=pd.DataFrame([{'some_value': 42, PopulationObs.WEIGHT_CHANNEL: 1}, {'some_value': 42.1, PopulationObs.WEIGHT_CHANNEL: 1}]))
        distribution = GaussianDistribution()
        self.assertRaises(GaussianDistribution.InvalidUncertaintyException, distribution.prepare,
                          dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                          weight_channel=PopulationObs.WEIGHT_CHANNEL)

        # uncertainty channel is present but has an invalid value
        df = pd.DataFrame([{'some_value': 77.1, GaussianDistribution.UNCERTAINTY_CHANNEL: 0, PopulationObs.WEIGHT_CHANNEL: 1},
                           {'some_value': 78, GaussianDistribution.UNCERTAINTY_CHANNEL: 50.1, PopulationObs.WEIGHT_CHANNEL: 1}])
        dfw = PopulationObs(dataframe=df)
        distribution = GaussianDistribution()
        self.assertRaises(GaussianDistribution.InvalidUncertaintyException, distribution.prepare,
                          dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                          weight_channel=PopulationObs.WEIGHT_CHANNEL)

        # uncertainty channel is present and all valid
        df = pd.DataFrame([{'some_value': 101.01, GaussianDistribution.UNCERTAINTY_CHANNEL: 1, PopulationObs.WEIGHT_CHANNEL: 1},
                           {'some_value': 212.12, GaussianDistribution.UNCERTAINTY_CHANNEL: 50.1, PopulationObs.WEIGHT_CHANNEL: 1}])
        dfw = PopulationObs(dataframe=df)
        distribution = GaussianDistribution()
        distribution.prepare(dfw=dfw, channel='some_value', provinciality=PopulationObs.PROVINCIAL, age_bins=[],
                             weight_channel=PopulationObs.WEIGHT_CHANNEL)

if __name__ == '__main__':
    unittest.main()
