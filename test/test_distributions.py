import unittest

from dtk.utils.observations.BaseDistribution import BaseDistribution
from dtk.utils.observations.GaussianDistribution import GaussianDistribution


class TestDistributions(unittest.TestCase):

    # BaseDistribution initialization test
    def test_instantiation_from_string(self):
        distribution = BaseDistribution.from_string(distribution_name='Gaussian')
        self.assertTrue(isinstance(distribution, GaussianDistribution))
        self.assertRaises(BaseDistribution.UnknownDistributionException,
                          BaseDistribution.from_string, distribution_name='Tibia')

if __name__ == '__main__':
    unittest.main()
