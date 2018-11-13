import os
import pandas as pd
import unittest

from dtk.utils.observations.AgeBin import AgeBin
from dtk.utils.observations.PopulationObs import PopulationObs
import dtk.utils.observations.utils as ingest_utils


class TestCalibrationIngest(unittest.TestCase):
    def setUp(self):
        self.data_directory = os.path.join(os.path.dirname(__file__), 'input', 'Excel', 'ingest')
        # self.test_excel_file = os.path.join(self.data_directory, 'test.xlsm')
        # self.wb = openpyxl.load_workbook(self.test_excel_file)
        # self.defined_names = excel.DefinedName.load_from_workbook(self.wb)


    def tearDown(self):
        pass


    # parameter parsing


    def test_fail_if_parameters_have_missing_values(self):
        filenames = ['missing_parameter_values_dynamic.xlsm', 'missing_parameter_values_name.xlsm']
        for filename in filenames:
            filename = os.path.join(self.data_directory, filename)
            self.assertRaises(ingest_utils.IncompleteParameterSpecification,
                              ingest_utils.parse_ingest_data_from_xlsm, filename=filename)

    def test_fail_if_parameter_initial_beyond_min_max(self):
        filenames = ['parameter_below_min.xlsm', 'parameter_above_max.xlsm']
        for filename in filenames:
            filename = os.path.join(self.data_directory, filename)
            self.assertRaises(ingest_utils.ParameterOutOfRange,
                              ingest_utils.parse_ingest_data_from_xlsm, filename=filename)


    def test_fail_if_parameter_has_non_numeric_value(self):
        filenames = ['parameter_has_non_numeric_value.xlsm']
        for filename in filenames:
            filename = os.path.join(self.data_directory, filename)
            self.assertRaises(ingest_utils.ParameterOutOfRange,
                              ingest_utils.parse_ingest_data_from_xlsm, filename=filename)


    # analyzer parsing


    def test_fail_if_analyzers_have_missing_values(self):
        filenames = ['missing_analyzer_values_age_bins.xlsm',
                     'missing_analyzer_values_channel.xlsm',
                     'missing_analyzer_values_custom_age_bins.xlsm',
                     'missing_analyzer_values_distribution.xlsm',
                     'missing_analyzer_values_weight.xlsm'
                     ]
        for filename in filenames:
            filename = os.path.join(self.data_directory, filename)
            self.assertRaises(ingest_utils.IncompleteAnalyzerSpecification,
                              ingest_utils.parse_ingest_data_from_xlsm, filename=filename)

    def test_fail_if_analyzer_weight_is_non_numeric(self):
        filenames = ['analzyer_weight_has_non_numeric_value.xlsm']
        for filename in filenames:
            filename = os.path.join(self.data_directory, filename)
            self.assertRaises(ingest_utils.InvalidAnalyzerWeight,
                              ingest_utils.parse_ingest_data_from_xlsm, filename=filename)

    # reference data parsing


    def test_fail_if_reference_data_has_missing_values(self):
        filename = os.path.join(self.data_directory, 'missing_reference_values.xlsm')
        self.assertRaises(ingest_utils.IncompleteDataSpecification,
                          ingest_utils.parse_ingest_data_from_xlsm, filename=filename)


    # other


    def test_fail_if_not_parsing_an_xlsm_file(self):
        filename = os.path.join(self.data_directory, 'not_a_xlsm_file.csv')
        self.assertRaises(ingest_utils.UnsupportedFileFormat,
                          ingest_utils.parse_ingest_data_from_xlsm, filename=filename)

    def test_a_properly_filled_xlsm_sheet(self):
        filename = os.path.join(self.data_directory, 'properly_filled.xlsm')
        params, reference, analyzers = ingest_utils.parse_ingest_data_from_xlsm(filename=filename)

        # check analyzers
        expected = [
            {
                'channel': 'Prevalence',
                'distribution': 'Gaussian',
                'provinciality': 'Provincial',
                'weight': 0.5,
                'age_bins': AgeBin.ALL
            },
            {
                'channel': 'Prevalence',
                'distribution': 'Gaussian',
                'provinciality': 'Non-provincial',
                'weight': 0.25,
                'age_bins': '[15:50);[50:100)'
            },
            {
                'channel': 'Prevalence',
                'distribution': 'Gaussian',
                'provinciality': 'Non-provincial',
                'weight': 0.25,
                'age_bins': AgeBin.ALL
            }
            ]

        self.assertTrue(isinstance(analyzers, list))
        self.assertEqual(len(analyzers), len(expected))
        self.assertEqual(analyzers, expected)

        # check params
        expected = [
            {
                'Name': 'p2',
                'Dynamic': False,
                'Guess': 2000,
                'Min': 1200,
                'Max': 2400

            },
            {
                'Name': 'p1',
                'Dynamic': True,
                'Guess': 0.1,
                'Min': 0,
                'Max': 1,
                'MapTo': 'p1'

            },
        ]
        self.assertTrue(isinstance(params, list))
        self.assertEqual(len(params), len(expected))
        sorting_lambda = lambda x: x['Name']
        self.assertEqual(sorted(params, key=sorting_lambda), sorted(expected, key=sorting_lambda))

        # check reference

        self.assertTrue(isinstance(reference, PopulationObs))

        expected_stratifiers = ['Year', 'Gender', 'AgeBin', 'Province']
        self.assertEqual(sorted(reference.stratifiers), sorted(expected_stratifiers))

        # non-stratifier columns in the dataframe
        expected_channels = ['Prevalence'] #, 'confidence_interval']
        self.assertEqual(sorted(reference.channels), sorted(expected_channels))

        n_expected_rows = 4
        self.assertEqual(len(reference._dataframe.index), n_expected_rows)

        # data check
        data = [
            {'Year': 2005, 'Gender': 'Male', 'AgeBin': '[0:99)', 'Province': 'Washington', 'Prevalence': 0.25}, #, 'confidence_interval': 0.05},
            {'Year': 2005, 'Gender': 'Female', 'AgeBin': '[0:99)', 'Province': 'Washington', 'Prevalence': 0.2}, #, 'confidence_interval': 0.04},
            {'Year': 2010, 'Gender': 'Male', 'AgeBin': '[5:15)', 'Province': 'Washington', 'Prevalence': 0.3}, #, 'confidence_interval': 0.07},
            {'Year': 2010, 'Gender': 'Female', 'AgeBin': '[15:25)', 'Province': 'Oregon', 'Prevalence': 0.33}, #, 'confidence_interval': 0.08},
        ]
        df = pd.DataFrame(data)
        expected_reference = PopulationObs(dataframe=df, stratifiers=expected_stratifiers)
        reference, expected_reference = self.ensure_same_column_order(reference, expected_reference)
        self.assertTrue(reference.equals(expected_reference))

    def ensure_same_column_order(self, dfw1, dfw2):
        self.assertEqual(sorted(dfw1._dataframe.columns), sorted(dfw2._dataframe.columns))
        reordered_columns = sorted(dfw1._dataframe.columns)
        dfw1._dataframe = dfw1._dataframe[reordered_columns]
        dfw2._dataframe = dfw2._dataframe[reordered_columns]
        return dfw1, dfw2


    # observational data - weight value parsing - HIV issue 62


    def test_fail_if_obs_data_missing_weight_column(self):
        filename = os.path.join(self.data_directory, 'obs_data_missing_weight_column.xlsm')
        self.assertRaises(ingest_utils.IncompleteDataSpecification,
                          ingest_utils.parse_ingest_data_from_xlsm, filename=filename)

    def test_obs_data_specified_and_default_weights_are_correctly_parsed(self):
        filename = os.path.join(self.data_directory, 'obs_data_correct_and_default_weight_column_values.xlsm')
        params, site_info, reference, analyzers = ingest_utils.parse_ingest_data_from_xlsm(filename=filename)

        # now check reference data for correctness of obs weights
        expected = {
            'Prevalence': [1, 1, 2.2],
            'Incidence': [0.2, 1, 3]
        }
        for channel, expected_vals in expected.items():
            # considering data from sheets individually
            ref = reference.filter(keep_only=[channel, PopulationObs.WEIGHT_CHANNEL])
            actual = ref._dataframe[PopulationObs.WEIGHT_CHANNEL]
            self.assertTrue((expected_vals == actual).all())


if __name__ == '__main__':
    unittest.main()
