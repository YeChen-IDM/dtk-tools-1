from copy import deepcopy
import operator
import os
import pandas as pd
import unittest

from dtk.utils.observations.Channel import Channel
from dtk.utils.observations.DataFrameWrapper import DataFrameWrapper


class TestDataFrameWrapper(unittest.TestCase):
    def setUp(self):
        data_array = [
            {
                'Year': 2000.0,
                'AgeBin': '[0:5)',
                'Gender': 'Male',
                'NationalPrevalence': 0.05
            },
            {
                'Year': 2005.0,
                'AgeBin': '[0:5)',
                'Gender': 'Male',
                'Province': 'Washington',
                'NationalPrevalence': 0.06,
                'On_ART': 5000
            },
            {
                'Year': 2005.0,
                'AgeBin': '[0:5)',
                'Gender': 'Female',
                'NationalPrevalence': 0.07,
                'On_ART': 6000
            }
        ]
        self.dataframe = pd.DataFrame(data_array)
        self.stratifiers = ['Year', 'AgeBin', 'Gender', 'Province']
        self.dfw = DataFrameWrapper(dataframe=self.dataframe, stratifiers=self.stratifiers)

        # for merge() tests
        self.merged_col = 'Sim_Prevalence'
        merged_data_array = deepcopy(data_array)
        for h in merged_data_array:
            h[self.merged_col] = h['NationalPrevalence'] * 1.1
            h.pop('NationalPrevalence')
            h.pop('On_ART', None)
        # modify to have one item not in data_array, and one item data_array that is missing
        merged_data_array[-1]['Year'] = 2005.5

        self.merging_dfw = DataFrameWrapper(dataframe=pd.DataFrame(merged_data_array), stratifiers=self.stratifiers)

        # for directory loading tests
        self.data_directory = os.path.join(os.path.dirname(__file__), 'input', 'DataFrameWrapper')

        self.data_array = data_array
        self.merging_data_array = merged_data_array

    def tearDown(self):
        pass

    # initialization tests

    def test_raise_when_filename_and_dataframe_provided(self):
        self.assertRaises(ValueError, DataFrameWrapper, filename='made_up.csv', dataframe=self.dataframe)

    def test_raise_when_unsupported_file_type_provided(self):
        self.assertRaises(DataFrameWrapper.UnsupportedFileType, DataFrameWrapper, filename='made.up')

    def test_df_initialization_with_provided_stratifiers(self):
        self.assertEqual(len(DataFrameWrapper(dataframe=self.dataframe).stratifiers), 0)

        expected_stratifiers = self.stratifiers
        dfw = DataFrameWrapper(dataframe=self.dataframe, stratifiers=expected_stratifiers)
        self.assertEqual(sorted(dfw.stratifiers), sorted(expected_stratifiers))

    def test_df_initialization_with_decorated_stratifiers(self):
        cols = list(self.dataframe.columns)
        original_cols = deepcopy(cols)
        cols[0] = str(Channel.construct_channel_string(name=cols[0], decorator='s'))
        self.dataframe.columns = cols

        dfw = DataFrameWrapper(dataframe=self.dataframe)
        expected_stratifiers = original_cols[0:1]
        self.assertEqual(list(dfw.stratifiers), expected_stratifiers)

    def test_df_initialization_with_provided_stratifiers_not_in_df(self):
        expected_stratifiers = ['Year', 'AgeBin', 'Gender', 'Provence'] # misspelled Province
        self.assertRaises(DataFrameWrapper.MissingRequiredData,
                          DataFrameWrapper, dataframe=self.dataframe, stratifiers=expected_stratifiers)

    # channel tests

    def test_channels_and_stratifiers_sum_to_columns(self):
        self.assertEqual(sorted(set(self.dfw.stratifiers + self.dfw.channels)),
                         sorted(set(self.dfw._dataframe.columns)))

    # filter tests

    def test_filter_with_no_conditions_or_keep_only(self):
        # result should be unchanged
        filtered_dfw = self.dfw.filter()
        self.assertTrue(self.dfw.equals(filtered_dfw))

    def test_filter_with_no_conditions(self):
        # result should contain all rows still (this specific case), but fewer columns
        keep_only = ['NationalPrevalence']
        filtered_dfw = self.dfw.filter(keep_only=keep_only)
        # check columns, stratifiers, and row counts
        self.assertEqual(sorted(filtered_dfw.stratifiers), sorted(self.dfw.stratifiers))
        self.assertEqual(filtered_dfw.channels, keep_only)
        self.assertEqual(len(filtered_dfw._dataframe.index), len(self.dfw._dataframe.index))
        # check data
        expected_dfw = DataFrameWrapper(dataframe=self.dfw._dataframe[self.dfw.stratifiers + keep_only],
                                        stratifiers=self.dfw.stratifiers)
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

    def test_filter_with_no_keep_only(self):
        selected_year = 2005

        # check a single condition
        conditions = [['Year', operator.eq, selected_year]]
        filtered_dfw = self.dfw.filter(conditions=conditions)

        # check columns, stratifiers, and row counts
        self.assertEqual(sorted(filtered_dfw.stratifiers), sorted(self.dfw.stratifiers))
        self.assertEqual(filtered_dfw.channels, self.dfw.channels)
        self.assertEqual(len(filtered_dfw._dataframe.index), 2)
        # check data
        expected_dfw = DataFrameWrapper(dataframe=self.dfw._dataframe.loc[self.dfw._dataframe['Year'] == selected_year],
                                        stratifiers=self.dfw.stratifiers)
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

        # test multiple supplied conditions
        conditions = [['Year', operator.eq, selected_year],
                      ['NationalPrevalence', operator.gt, 0.06],
                      ['Gender', operator.ne, 'Male']]
        filtered_dfw = self.dfw.filter(conditions=conditions)
        # check columns, stratifiers, and row counts
        self.assertEqual(filtered_dfw.stratifiers, self.dfw.stratifiers)
        self.assertEqual(filtered_dfw.channels, self.dfw.channels)
        self.assertEqual(len(filtered_dfw._dataframe.index), 1)
        # check data
        expected_df = self.dfw._dataframe.loc[
            (self.dfw._dataframe['Year'] == selected_year) &
            (self.dfw._dataframe['Gender'] != 'Male') &
            (self.dfw._dataframe['NationalPrevalence'] > 0.06)
            ]
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=self.dfw.stratifiers)
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

    def test_filter_with_keep_only_not_in_df(self):
        self.assertRaises(DataFrameWrapper.MissingRequiredData, self.dfw.filter, keep_only=['Deimos Down'])

    # verify_required_items tests

    def test_verify_required_items_available_not_provided(self):
        # should work with no exception
        self.dfw.verify_required_items(needed=['Year', 'AgeBin', 'NationalPrevalence'])

        # should throw an exception as we're requesting something not in the dfw
        self.assertRaises(DataFrameWrapper.MissingRequiredData, self.dfw.verify_required_items,
                          needed=['Space Elevator'])

    def test_verify_required_items_available_provided(self):
        # should work with no exception
        self.dfw.verify_required_items(needed=['Year', 'AgeBin', 'NationalPrevalence'],
                                       available=self.dfw._dataframe.columns)

        # should throw an exception as we're requesting something not in the dfw
        self.assertRaises(DataFrameWrapper.MissingRequiredData, self.dfw.verify_required_items,
                          needed=['Io Mining Industries'],
                          available=['Terraforming Ganymede'])

    # from_directory tests

    def test_from_directory_stratifiers_provided_and_in_df(self):
        data_directory = os.path.join(self.data_directory, 'in_common_stratifiers')
        stratifiers = ['a']
        dfw = DataFrameWrapper.from_directory(directory=data_directory, stratifiers=stratifiers)
        self.assertEqual(sorted(dfw.stratifiers), sorted(stratifiers))

        # data check now
        expected_df = pd.DataFrame(
            [
                {'a': 1, 'c': 'Regolith Eaters'},
                {'a': 2, 'c': 'Strip Mine'},
                {'a': 0, 'b': 'Rover Construction'}
            ]
        )
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=['a'])
        self.assertTrue(dfw.equals(expected_dfw))

    def test_from_directory_stratifiers_provided_and_not_in_df(self):
        data_directory = os.path.join(self.data_directory, 'in_common_stratifiers')
        self.assertRaises(DataFrameWrapper.MissingRequiredData, DataFrameWrapper.from_directory,
                          directory=data_directory, stratifiers=['Symbiotic Fungus'])

    def test_from_directory_stratifiers_detected_and_match(self):
        data_directory = os.path.join(self.data_directory, 'partially_disjoint_stratifiers')
        expected_stratifiers = ['Year', 'Gender', 'Province']
        dfw = DataFrameWrapper.from_directory(directory=data_directory)
        self.assertEqual(sorted(dfw.stratifiers), sorted(expected_stratifiers))
        self.assertEqual(len(dfw._dataframe.index), 4)

        expected_df = pd.DataFrame(
            [
                {'Year': 2010, 'Gender': 'Male', 'NationalPrevalence': 0.1},
                {'Year': 2011, 'Gender': 'Male', 'NationalPrevalence': 0.2},
                {'Year': 2005, 'Gender': 'Female', 'Province': 'Idaho',   'ProvincialPrevalence': 0.05},
                {'Year': 2006, 'Gender': 'Female', 'Province': 'Montana', 'ProvincialPrevalence': 0.06}
            ]
        )
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=expected_stratifiers)
        dfw, expected_dfw = self.ensure_same_column_order(dfw, expected_dfw)
        self.assertTrue(dfw.equals(expected_dfw))

        # a few data tests to ensure proper behavior when input files use different stratifiers

        # 1.
        filtered_dfw = dfw.filter(conditions=[['Gender', operator.eq, 'Male']], keep_only='NationalPrevalence')
        expected_df = pd.DataFrame(
            [
                {'Year': 2010, 'Gender': 'Male', 'Province': None, 'NationalPrevalence': 0.1},
                {'Year': 2011, 'Gender': 'Male', 'Province': None, 'NationalPrevalence': 0.2},
            ]
        )
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=expected_stratifiers)
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

        # 2.
        filtered_dfw = dfw.filter(conditions=[['Gender', operator.eq, 'Female']], keep_only='ProvincialPrevalence')
        expected_df = pd.DataFrame(
            [
                {'Year': 2005, 'Gender': 'Female', 'Province': 'Idaho', 'ProvincialPrevalence': 0.05},
                {'Year': 2006, 'Gender': 'Female', 'Province': 'Montana', 'ProvincialPrevalence': 0.06},
            ]
        )
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=expected_stratifiers)
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

        # 3.
        # and a really silly query; there should be nothing matching it due to nan removal
        filtered_dfw = dfw.filter(conditions=[['Gender', operator.eq, 'Female']],
                                  keep_only=['ProvincialPrevalence', 'NationalPrevalence'])
        stratifiers = ['Year', 'Gender', 'Province']
        data_columns = ['ProvincialPrevalence', 'NationalPrevalence']
        expected_df = pd.DataFrame([], columns=(stratifiers + data_columns))
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=stratifiers)
        # setting types for expected value manually, as it can't be inferred due to creation as empty dataframe
        expected_dfw._dataframe = expected_dfw._dataframe.astype({'ProvincialPrevalence': 'float64',
                                                                  'NationalPrevalence': 'float64',
                                                                  'Year': 'int64'})
        filtered_dfw, expected_dfw = self.ensure_same_column_order(filtered_dfw, expected_dfw)
        self.assertTrue(filtered_dfw.equals(expected_dfw))

    # merge tests

    def test_merge_returns_object_of_right_type(self):
        # create an object of a derived type of DataFrameWrapper and verify we get that type back after merging
        from dtk.utils.observations.PopulationObs import PopulationObs
        merging_pop = PopulationObs(dataframe=self.dfw._dataframe, stratifiers=self.stratifiers)
        self.stratifiers.remove('Province')
        keep = [self.merged_col, 'NationalPrevalence']
        merged = merging_pop.merge(other_dfw=self.merging_dfw, index=self.stratifiers, keep_only=keep)
        self.assertEqual(type(merged), type(merging_pop))

    def test_raise_if_not_merging_with_a_dfw(self):
        not_a_dfw = pd.DataFrame([{'a':1, 'b':2}, {'a':3, 'b':4}])
        self.assertRaises(Exception, self.dfw.merge, other_dfw=not_a_dfw, drop_nan=self.merged_col)

    def test_raise_if_merge_objects_missing_a_requested_stratifier(self):
        self.merging_dfw.stratifiers.pop()
        self.stratifiers.append('missing_stratifier')
        self.assertRaises(DataFrameWrapper.MissingRequiredData, self.dfw.merge,
                          other_dfw=self.merging_dfw, index=self.stratifiers, keep_only=self.merged_col)

    def test_merge_result_is_correct_with_and_without_drop_nan(self):
        # # test with drop_nan

        self.stratifiers.remove('Province')
        keep = [self.merged_col, 'NationalPrevalence']
        merged = self.dfw.merge(other_dfw=self.merging_dfw, index=self.stratifiers, keep_only=keep)
        expected_data = self.data_array[0:2]
        for i in range(len(expected_data)):
            expected_data[i][self.merged_col] = self.merging_data_array[i][self.merged_col]
            expected_data[i].pop('Province', None)
            expected_data[i].pop('On_ART', None)
        expected_df = pd.DataFrame(data=expected_data)
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=self.stratifiers)
        expected_dfw, merged = self.ensure_same_column_order(expected_dfw, merged)
        self.assertTrue(merged.equals(expected_dfw))

        # test without drop_nan

        merged = self.dfw.merge(other_dfw=self.merging_dfw, index=self.stratifiers)
        # there is an un-merged pair of Province columns and On_ART in addition to the prior test
        expected_df = pd.DataFrame(
            [
                {'Year': 2000.0, 'AgeBin': '[0:5)', 'Gender': 'Male', 'NationalPrevalence': 0.05, 'Sim_Prevalence':0.05*1.1},
                {'Year': 2005.0, 'AgeBin': '[0:5)', 'Gender': 'Male', 'NationalPrevalence': 0.06, 'On_ART': 5000.0, 'Province_x': 'Washington', 'Province_y': 'Washington', 'Sim_Prevalence':0.066},
            ]
        )
        expected_dfw = DataFrameWrapper(dataframe=expected_df, stratifiers=self.stratifiers)
        expected_dfw, merged = self.ensure_same_column_order(expected_dfw, merged)
        self.assertTrue(merged.equals(expected_dfw))

    def ensure_same_column_order(self, dfw1, dfw2):
        self.assertEqual(sorted(dfw1._dataframe.columns), sorted(dfw2._dataframe.columns))
        reordered_columns = sorted(dfw1._dataframe.columns)
        dfw1._dataframe = dfw1._dataframe[reordered_columns]
        dfw2._dataframe = dfw2._dataframe[reordered_columns]
        return dfw1, dfw2

if __name__ == '__main__':
    unittest.main()
