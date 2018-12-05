import os
import tempfile
import unittest

from dtk.tools.climate.ClimateGenerator import ClimateGenerator
from dtk.tools.climate.WeatherNode import WeatherNode
from dtk.tools.demographics.DemographicsFile import DemographicsFile
from simtools.SetupParser import SetupParser


class ClimateGeneratorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SetupParser.init()

    @staticmethod
    def get_test_demo(path):
        node_1001 = WeatherNode(lon=27.6, lat=-17.05, name='Node 1001', pop=1000)
        node_others = WeatherNode(lon=28, lat=-16.5, name='Others', pop=1000)
        nodes = [node_1001, node_others]

        # Create the file
        dg = DemographicsFile(nodes)
        climate_demog = os.path.join(path, 'climate_demog.json')
        dg.generate_file(climate_demog)
        return climate_demog

    # This will test basica validation in a happy case through the contrucstor
    def test_validate_input(self):
        with tempfile.TemporaryDirectory() as output_path:
            climate_demog = self.get_test_demo(output_path)
            cg = ClimateGenerator(demographics_file_path=climate_demog,
                                  work_order_path=os.path.join(output_path, 'wo.json'),
                                  climate_files_output_path=output_path,
                                  climate_project='IDM-Zambia',
                                  start_year='2008', num_years='1',
                                  resolution='0',
                                  project_root='v2017',
                                  idRef="Gridded world grump2.5arcmin")

    def test_bad_project_name(self):
        with self.assertRaises(ValueError) as cm:
            with tempfile.TemporaryDirectory() as output_path:
                climate_demog = self.get_test_demo(output_path)

                cg = ClimateGenerator(demographics_file_path=climate_demog,
                                      work_order_path=os.path.join(output_path, 'wo.json'),
                                      climate_files_output_path=output_path,
                                      climate_project='IDM-Unknown',
                                      start_year='2008', num_years='1',
                                      resolution='0',
                                      project_root='v2017',
                                      idRef="Gridded world grump2.5arcmin")
        # Assert that we let the error message contains the project name requested
        self.assertIn('IDM-Unknown', str(cm.exception))

    def test_bad_start_year(self):
        with self.assertRaises(ValueError) as cm:
            with tempfile.TemporaryDirectory() as output_path:
                climate_demog = self.get_test_demo(output_path)
                cg = ClimateGenerator(demographics_file_path=climate_demog,
                                      work_order_path=os.path.join(output_path, 'wo.json'),
                                      climate_files_output_path=output_path,
                                      climate_project='IDM-Zambia',
                                      start_year='1996', num_years='1',
                                      resolution='0',
                                      project_root='v2017',
                                      idRef="Gridded world grump2.5arcmin")
        # Assert that we let the error message
        # contains the project name
        self.assertIn('IDM-Zambia', str(cm.exception))
        # the requested year range
        self.assertIn('1996 to 1997', str(cm.exception))
        # the supported range
        self.assertIn('2001 to 201', str(cm.exception))

    def test_bad_resolution(self):
        with self.assertRaises(ValueError) as cm:
            with tempfile.TemporaryDirectory() as output_path:
                climate_demog = self.get_test_demo(output_path)
                cg = ClimateGenerator(demographics_file_path=climate_demog,
                                      work_order_path=os.path.join(output_path, 'wo.json'),
                                      climate_files_output_path=output_path,
                                      climate_project='IDM-Zambia',
                                      start_year='2007', num_years='1',
                                      resolution='15',
                                      project_root='v2017',
                                      idRef="Gridded world grump2.5arcmin")

        # ensure supported resolutions are listed in error message
        self.assertIn('0, 2.5, 30', str(cm.exception))
        # ensure user's bad input is in the error message
        self.assertIn('15', str(cm.exception))

    def test_bad_resolution2(self):
        with self.assertRaises(ValueError) as cm:
            with tempfile.TemporaryDirectory() as output_path:
                climate_demog = self.get_test_demo(output_path)
                cg = ClimateGenerator(demographics_file_path=climate_demog,
                                      work_order_path=os.path.join(output_path, 'wo.json'),
                                      climate_files_output_path=output_path,
                                      climate_project='IDM-Zambia',
                                      start_year='2007', num_years='1',
                                      resolution='2.5',
                                      project_root='v2017',
                                      idRef="Gridded world grump2.5arcmin")
                cg.generate_climate_files()

        # ensure supported resolutions are listed in error message
        self.assertIn('0, 2.5, 30', str(cm.exception))
        # ensure user's bad input is in the error message
        self.assertIn('15', str(cm.exception))

    def test_generate(self):
        with tempfile.TemporaryDirectory() as output_path:
            climate_demog = self.get_test_demo(output_path)
            cg = ClimateGenerator(demographics_file_path=climate_demog,
                                  work_order_path=os.path.join(output_path, 'wo.json'),
                                  climate_files_output_path=output_path,
                                  climate_project='IDM-Zambia',
                                  start_year='2008', num_years='1',
                                  resolution='0',
                                  project_root='v2017',
                                  idRef="Gridded world grump2.5arcmin")
            rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
            # TODo How to verify the generate worked?
            print(rain_fn)
