import os
import tempfile
import unittest

from dtk.tools.demographics.DemographicsGenerator import DemographicsGenerator
from simtools.SetupParser import SetupParser


class DemographicsGeneratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SetupParser.init()

    def test_from_grid_file(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            final_grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')
            demo = DemographicsGenerator.from_grid_file(population_input_file=final_grid_file,
                                                        demographics_filename=demo_fp)
