import openpyxl
import os
import unittest

import dtk.utils.io.excel as excel

class TestExcel(unittest.TestCase):
    def setUp(self):
        self.data_directory = os.path.join(os.path.dirname(__file__), 'input', 'Excel')
        self.test_excel_file = os.path.join(self.data_directory, 'test.xlsm')
        self.wb = openpyxl.load_workbook(self.test_excel_file)
        self.defined_names = excel.DefinedName.load_from_workbook(self.wb)


    def tearDown(self):
        pass


    def test_read_row(self):
        ws_name = 'Observations metadata'
        defined_name = self.defined_names[ws_name]['obs_data_channels']
        data = excel.read_list(ws=self.wb[ws_name], range=defined_name)
        expected = ['NationalPrevalence', None, 'ProvincialPrevalence'] + [None for i in range(21)]
        self.assertEqual(data, expected)


    def test_read_column(self):
        ws_name = 'Analyzers'
        defined_name = self.defined_names[ws_name]['analyzer_names']
        data = excel.read_list(ws=self.wb[ws_name], range=defined_name)
        expected = [None, 'ProvincialARTAnalyzer', 'NationalPrevalenceAnalyzer'] + [None for i in range(18)]
        self.assertEqual(data, expected)


    def test_read_block(self):
        ws_name = 'Obs-NationalPrevalence'
        defined_name = self.defined_names[ws_name]['csv']
        data = excel.read_block(ws=self.wb[ws_name], range=defined_name)
        expected = []
        expected.append(['Year', 'AgeBin', 'Gender', 'NationalPrevalence'])
        expected.append([2015, '[15:49)', 'Male', 0.05])
        expected.append([2016, '[15:49)', 'Male', 0.06])
        for i in range(998):
            expected.append([None for i in range(4)])
        self.assertEqual(len(data), len(expected))
        self.assertEqual(data, expected)


    def test_should_fail_if_reading_block_as_list(self):
        ws_name = 'Obs-NationalPrevalence'
        defined_name = self.defined_names[ws_name]['csv']
        self.assertRaises(Exception, excel.read_list, ws=self.wb[ws_name], range=defined_name)


    def test_using_workbook_scope_range(self):
        ws_name = 'workbook'
        defined_name = self.defined_names[ws_name]['boolean_selection']
        data = excel.read_list(ws=self.wb[defined_name.sheet], range=defined_name)
        expected = ['--select--', True, False] # strings 'TRUE' and 'FALSE' get converted by openpyxl
        self.assertEqual(data, expected)


if __name__ == '__main__':
    unittest.main()
