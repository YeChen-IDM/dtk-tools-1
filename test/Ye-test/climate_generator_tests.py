import os
import unittest
import shutil
import json
import tempfile

from dtk.tools.climate.ClimateGenerator import ClimateGenerator
from dtk.tools.climate.WeatherNode import WeatherNode
from dtk.tools.demographics.DemographicsFile import DemographicsFile
from simtools.SetupParser import SetupParser


class ClimateGeneratorTests(unittest.TestCase):
    # region unittest setup
    @classmethod
    def setUpClass(cls):
        SetupParser.init()
        cls.output_path = "climate_generator_test"
        if not os.path.isdir(cls.output_path):
            os.mkdir(cls.output_path)
    def setUp(self):
        self.test_output = os.path.join(self.output_path, self._testMethodName)
        if os.path.isdir(self.test_output):
            shutil.rmtree(self.test_output)
        os.mkdir(self.test_output)
    # endregion

    # region method
    @staticmethod
    def get_test_demo(path: str, idref: str=None, location: list=None):
        # default nodes lat and lon are for IDM-Zambia project
        if not location:
            location = [[-17.05, 27.6], [-16.6, 28]]
        node_1= WeatherNode(lat=location[0][0], lon=location[0][1], name='node_1', pop=1000)
        node_2 = WeatherNode(lat=location[1][0], lon=location[1][1], name='node_2', pop=1000)
        nodes = [node_1, node_2]

        # use default idref from DemographicsFile if user doesn't provide idref
        if idref:
            dg = DemographicsFile(nodes, idref)
        else:
            dg = DemographicsFile(nodes)
        # Create the file
        climate_demog = os.path.join(path, 'climate_demog.json')
        dg.generate_file(climate_demog)
        return climate_demog
    # endregion

    # region Happy cases
    def test_valid_input(self):
        """
        Tests constructor of ClimateGenerator with valid inputs.
        Returns:

        """
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2008', num_years='1',
                              resolution='0',
                              project_root='v2017')

    def test_project_name(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts climate data is not from the correct climate_project.
        Returns:

        """
        climate_project = 'IDM-Kenya'
        climate_demog = self.get_test_demo(self.test_output, location=[[-2.53, 36.12], [-2.48, 36.15]])
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project=climate_project,
                              start_year='2008', num_years='1',
                              resolution='0',
                              project_root='v2017'
                              )
        rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
        for climate_file in [rain_fn, tempfn, humidity_fn]:
            self.assertEqual(climate_file.split('_')[0], climate_project.split('-')[-1])

    def test_year(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts climate data it not from the correct years.
        Returns:

        """
        start_year = '2007'
        num_years = '2'
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year=start_year, num_years=num_years,
                              resolution='0',
                              project_root='v2017')
        files = cg.generate_climate_files()
        for file in files:
            with open(os.path.join(self.test_output, file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["OriginalDataYears"],
                                 start_year + '-' + str(int(start_year)+int(num_years)-1))

    def test_resolution_0(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts resolution of climate data is not 30arcsec.
        Returns:

        """
        expected_resolution = '30arcsec'
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2008', num_years='1',
                              resolution='0',
                              project_root='v2017')
        rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
        for climate_file in [rain_fn, tempfn, humidity_fn]:
            self.assertEqual(climate_file.split('_')[1], expected_resolution)
            with open(os.path.join(self.test_output, climate_file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["Resolution"], expected_resolution)

    def test_resolution_150(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts resolution of climate data is not 2.5arcmin.
        Returns:

        """
        expected_resolution = '2.5arcmin'
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2008', num_years='1',
                              resolution='150',
                              project_root='v2017')
        rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
        for climate_file in [rain_fn, tempfn, humidity_fn]:
            self.assertEqual(climate_file.split('_')[1], expected_resolution)
            with open(os.path.join(self.test_output, climate_file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["Resolution"], expected_resolution)

    def test_resolution_30(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts resolution of climate data is not 30arcsec.
        Returns:

        """
        expected_resolution = '30arcsec'
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2008', num_years='1',
                              resolution='30',
                              project_root='v2017')
        rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
        for climate_file in [rain_fn, tempfn, humidity_fn]:
            self.assertEqual(climate_file.split('_')[1], expected_resolution)
            with open(os.path.join(self.test_output, climate_file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["Resolution"], expected_resolution)

    def test_idRef(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts climate data doesn't match correct IdReference.
        Returns:

        """
        idRef = "test idRef"
        climate_demog = self.get_test_demo(self.test_output, idref=idRef)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2007', num_years='1',
                              resolution='0',
                              project_root='v2017')
        files = cg.generate_climate_files()
        for file in files:
            with open(os.path.join(self.test_output, file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["IdReference"], idRef)

    def test_default_idRef(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts climate data doesn't match default IdReference in demo file.
        Returns:

        """
        default_idRef = "Gridded world grump2.5arcmin"
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2007', num_years='1',
                              resolution='30',
                              project_root='v2017')
        files = cg.generate_climate_files()
        for file in files:
            with open(os.path.join(self.test_output, file + ".json"), 'r') as json_file:
                json_metadata = json.load(json_file)["Metadata"]
                self.assertEqual(json_metadata["IdReference"], default_idRef)

    def test_output_path_generation(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts climate data is not stored in correct output_path.
        Returns:

        """
        output_path = os.path.join(self.test_output, 'output_path')
        # delete output_path if it already exists
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        climate_demog = self.get_test_demo(self.test_output)
        # the ClimateGenerator will create output_path folder and download the climate files in this folder
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(self.test_output, 'wo.json'),
                              climate_files_output_path=output_path,
                              climate_project='IDM-Zambia',
                              start_year='2008', num_years='1',
                              resolution='150',
                              project_root='v2017')
        rain_fn, tempfn, humidity_fn = cg.generate_climate_files()
        for climate_file in [rain_fn, tempfn, humidity_fn]:
            self.assertTrue(os.path.isfile(os.path.join(output_path, climate_file)))
            self.assertTrue(os.path.isfile(os.path.join(output_path, climate_file + '.json')))

    def test_work_order_path_generation(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts wo.json is not stored in correct work_order_path.
        Returns:

        """
        climate_demog = self.get_test_demo(self.test_output)
        # delete work_order_path if it already exists
        work_order_path = "test"
        if os.path.isdir(work_order_path):
            shutil.rmtree(work_order_path)
        # the ClimateGenerator will create work_order_path folder and put wo.json in this folder
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=os.path.join(work_order_path, 'wo.json'),
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2007', num_years='1',
                              resolution='30',
                              project_root='v2017')
        cg.generate_climate_files()
        self.assertTrue(os.path.isfile(os.path.join(work_order_path, 'wo.json')))

    def test_work_order_generation(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts wo_file("something") is not stored in working directory.
        Returns:

        """
        wo_file = 'something'
        if os.path.isfile(wo_file):
            os.remove(wo_file)
        climate_demog = self.get_test_demo(self.test_output)
        # the ClimateGenerator will put "something" in working directory and the work item should runs fine(use
        # "something" as "WorkOrder.json".)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path=wo_file,
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2007', num_years='1',
                              resolution='30',
                              project_root='v2017')
        cg.generate_climate_files()
        self.assertTrue(os.path.isfile(wo_file))

    def test_empty_work_order_path(self):
        """
        Tests constructor and generate_climate_files() of ClimateGenerator.
        Asserts 'wo.json' is not stored in working directory when user inputs an empty string.
        Returns:

        """
        wo_file = 'wo.json'
        if os.path.isfile(wo_file):
            os.remove(wo_file)
        climate_demog = self.get_test_demo(self.test_output)
        cg = ClimateGenerator(demographics_file_path=climate_demog,
                              work_order_path='',
                              climate_files_output_path=self.test_output,
                              climate_project='IDM-Zambia',
                              start_year='2007', num_years='1',
                              resolution='30',
                              project_root='v2017')
        cg.generate_climate_files()
        self.assertTrue(os.path.isfile('wo.json'))
    # endregion

    # region Bad cases
    def test_bad_project_name(self):
        """
        Tests constructor of ClimateGenerator.
        Asserts it doesn't throw the expected ValueError when user input a bad project name.
        Returns:

        """
        with self.assertRaises(ValueError) as cm:
            climate_demog = self.get_test_demo(self.test_output)
            cg = ClimateGenerator(demographics_file_path=climate_demog,
                                  work_order_path=os.path.join(self.test_output, 'wo.json'),
                                  climate_files_output_path=self.test_output,
                                  climate_project='IDM-Unknown')
        # Assert that we let the error message contains the project name requested
        self.assertIn('IDM-Unknown', str(cm.exception))

    def test_bad_start_year(self):
        """
        Tests constructor of ClimateGenerator.
        Asserts it doesn't throw the expected ValueError when user input a bad year.
        Returns:

        """
        with self.assertRaises(ValueError) as cm:
            climate_demog = self.get_test_demo(self.test_output)
            cg = ClimateGenerator(demographics_file_path=climate_demog,
                                  work_order_path=os.path.join(self.test_output, 'wo.json'),
                                  climate_files_output_path=self.test_output,
                                  climate_project='IDM-Zambia',
                                  start_year='1996', num_years='1',
                                  resolution='0',
                                  project_root='v2017')
        # the requested year range
        self.assertIn('1996 to 1997', str(cm.exception))
        # the supported range
        self.assertIn('2001 to 201', str(cm.exception))

    def test_bad_resolution(self):
        """
        Tests constructor of ClimateGenerator.
        Asserts it doesn't throw the expected ValueError when user input a bad resolution.
        Returns:

        """
        with self.assertRaises(ValueError) as cm:
            climate_demog = self.get_test_demo(self.test_output)
            cg = ClimateGenerator(demographics_file_path=climate_demog,
                                  work_order_path=os.path.join(self.test_output, 'wo.json'),
                                  climate_files_output_path=self.test_output,
                                  climate_project='IDM-Zambia',
                                  start_year='2007', num_years='1',
                                  resolution='15',
                                  project_root='v2017')

        # ensure supported resolutions are listed in error message
        self.assertIn('0, 150, 30', str(cm.exception))
        # ensure user's bad input is in the error message
        self.assertIn('15', str(cm.exception))

    # retire this test since we update the script to not allow mismatched idRef
    # def test_unmatched_idref(self):
    #     """
    #     Asserts ClimateGenerator.generate_climate_files() doesn't throw a proper error when work item is failed.
    #     Returns:
    #
    #     """
    #     with self.assertRaises(ValueError) as cm:
    #         climate_demog = self.get_test_demo(self.test_output)
    #         with open(climate_demog, "r") as climate_demog_file:
    #             cd = json.load(climate_demog_file)
    #             cd["Metadata"]["IdReference"] = 'unmatched idref'
    #         with open(climate_demog, "w") as climate_demog_file:
    #             json.dump(cd, climate_demog_file,indent=4)
    #         cg = ClimateGenerator(demographics_file_path=climate_demog,
    #                               work_order_path=os.path.join(self.test_output, 'wo.json'),
    #                               climate_files_output_path=self.test_output,
    #                               climate_project='IDM-Zambia',
    #                               start_year='2007', num_years='1',
    #                               resolution='0',
    #                               project_root='v2017',
    #                               idRef="Gridded world grump2.5arcmin")
    #         cg.generate_climate_files()
    #     # ensure work item not succeeded in error message
    #     self.assertIn('Work item status is not Succeeded', str(cm.exception))

    def test_bad_demo_path(self):
        """
        Tests constructor of ClimateGenerator.
        Asserts it doesn't throw the expected ValueError when user input a bad demographics_file_path.
        Returns:

        """
        with self.assertRaises(FileNotFoundError) as cm:
            with tempfile.TemporaryDirectory() as output_path:
                cg = ClimateGenerator(demographics_file_path=os.path.join(output_path, 'demo.json'),
                                      work_order_path=os.path.join(self.test_output, 'wo.json'),
                                      climate_files_output_path=self.test_output,
                                      climate_project='IDM-Zambia',
                                      start_year='2007', num_years='1',
                                      resolution='30',
                                      project_root='v2017')

        self.assertIn('No such file or directory', str(cm.exception))
    # endregion

