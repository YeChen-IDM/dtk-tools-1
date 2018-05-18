import numpy as np
import pandas as pd
import unittest

from dtk.utils.observations.AgeBin import AgeBin
from simtools.Utilities.DataFrame import upsample_agebin
from simtools.Utilities.DataFrame import NotUpsampleable

class TestDataFrame(unittest.TestCase):

    def setUp(self):
        data = [
            {'Gender': 'Male',   'AgeBin': '[5:10)',  'Prevalence': 0.1, 'Sim_Prevalence': 0.4, 'Count': 5},
            {'Gender': 'Male',   'AgeBin': '[10:15)', 'Prevalence': 0.2, 'Sim_Prevalence': 0.3, 'Count': 15},
            {'Gender': 'Female', 'AgeBin': '[5:10)',  'Prevalence': 0.3, 'Sim_Prevalence': 0.2, 'Count': 20},
            {'Gender': 'Female', 'AgeBin': '[10:15)', 'Prevalence': 0.4, 'Sim_Prevalence': 0.1, 'Count': 20}
        ]
        data = pd.DataFrame(data)
        self.grouped_data = data.groupby(['Gender'])
        self.aggregation_columns = ['Count']
        self.weighted_columns = ['Prevalence', 'Sim_Prevalence']
        self.weighting_column = 'Count'

    def tearDown(self):
        pass

    def test_upsample_agebin_raises_if_data_does_not_contain_requested_agebin(self):
        # overlap lower age
        age_bin = AgeBin.from_string('[0:10)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

        # overlap upper age
        age_bin = AgeBin.from_string('[10:18)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

        # overlap both lower and upper ages (requesting too large an age range on both sides)
        age_bin = AgeBin.from_string('[4:16)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

    def test_upsample_agebin_raises_if_data_not_edge_aligned(self):
        # not lower-edge aligned
        age_bin = AgeBin.from_string('[6:10)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

        # not upper-edge aligned
        age_bin = AgeBin.from_string('[5:14)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

        # not lower or upper edge aligned
        age_bin = AgeBin.from_string('[6:14)')
        self.assertRaises(NotUpsampleable,
                          self.grouped_data.apply, upsample_agebin,
                                                   age_bin=age_bin,
                                                   aggregated_cols=self.aggregation_columns,
                                                   weighted_cols=self.weighted_columns,
                                                   weighting_col=self.weighting_column)

    def test_upsample_agebin_works(self):
        age_bin = AgeBin.from_string('[5:10)')
        result = self.grouped_data.apply(upsample_agebin,
                                         age_bin=age_bin,
                                         aggregated_cols=self.aggregation_columns,
                                         weighted_cols=self.weighted_columns,
                                         weighting_col=self.weighting_column).sort_values('Gender').reset_index(drop=True).sort_index()
        expected_result = [
            {'Gender': 'Male', 'AgeBin': '[5:10)', 'Prevalence': 0.1, 'Sim_Prevalence': 0.4, 'Count': 5},
            {'Gender': 'Female', 'AgeBin': '[5:10)', 'Prevalence': 0.3, 'Sim_Prevalence': 0.2, 'Count': 20}
            ]
        expected_result = pd.DataFrame(expected_result).sort_values('Gender').reset_index(drop=True).sort_index()
        self.assertTrue(result.equals(expected_result))


        age_bin = AgeBin.from_string('[10:15)')
        result = self.grouped_data.apply(upsample_agebin,
                                         age_bin=age_bin,
                                         aggregated_cols=self.aggregation_columns,
                                         weighted_cols=self.weighted_columns,
                                         weighting_col=self.weighting_column).sort_values('Gender').reset_index(drop=True).sort_index()
        expected_result = [
            {'Gender': 'Male', 'AgeBin': '[10:15)', 'Prevalence': 0.2, 'Sim_Prevalence': 0.3, 'Count': 15},
            {'Gender': 'Female', 'AgeBin': '[10:15)', 'Prevalence': 0.4, 'Sim_Prevalence': 0.1, 'Count': 20}
            ]
        expected_result = pd.DataFrame(expected_result).sort_values('Gender').reset_index(drop=True).sort_index()
        self.assertTrue(result.equals(expected_result))


        age_bin = AgeBin.from_string('[5:15)')
        result = self.grouped_data.apply(upsample_agebin,
                                         age_bin=age_bin,
                                         aggregated_cols=self.aggregation_columns,
                                         weighted_cols=self.weighted_columns,
                                         weighting_col=self.weighting_column).sort_values('Gender').reset_index(drop=True).sort_index()
        expected_result = [
            {'Gender': 'Male', 'AgeBin': '[5:15)', 'Prevalence': 0.175, 'Sim_Prevalence': 0.325, 'Count': 20},
            {'Gender': 'Female', 'AgeBin': '[5:15)', 'Prevalence': 0.35, 'Sim_Prevalence': 0.15, 'Count': 40}
            ]
        expected_result = pd.DataFrame(expected_result).sort_values('Gender').reset_index(drop=True).sort_index()

        numerical_cols = ['Prevalence', 'Sim_Prevalence', 'Count']
        other_cols = ['Gender', 'AgeBin']

        # checking that numerical values are REALLY close; off a bit due to division in algorithm
        self.assertTrue(np.allclose(result[numerical_cols], expected_result[numerical_cols], atol=1e-16, rtol=0))

        # checking non-numerical values are EXACT
        self.assertTrue(result[other_cols].equals(expected_result[other_cols]))


if __name__ == '__main__':
    unittest.main()
