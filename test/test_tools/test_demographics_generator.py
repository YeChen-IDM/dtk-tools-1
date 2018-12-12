import os
import tempfile
import unittest

from dtk.tools.demographics.DemographicsFile import DemographicsFile
from dtk.tools.demographics.DemographicsGenerator import DemographicsGenerator as DemographicsGeneratorRE
from dtk.tools.demographics.generator.DemographicsGeneratorConcern import demographics_generator_concern
from dtk.tools.demographics.generator.DemographicsNodeGeneratorConcern import WorldBankBirthRateNodeConcern
from simtools.SetupParser import SetupParser


@demographics_generator_concern(larval_habitat_multiplier=1.0, calib_single_node_pop=1000.0)
def larval_habitat_multiplier(demographics, larval_habitat_multiplier=1.0, calib_single_node_pop: float = 1000.0):
    for node_item in demographics['Nodes']:
        pop_multiplier = float(node_item['NodeAttributes']['InitialPopulation']) / calib_single_node_pop

        if 'LarvalHabitatMultiplier' not in node_item['NodeAttributes']:
            node_item['NodeAttributes']['LarvalHabitatMultiplier'] = larval_habitat_multiplier
        # Copy the larval param dict handed to this node
        node_item['NodeAttributes']['LarvalHabitatMultiplier'] *= pop_multiplier


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
            grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')
            DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                   demographics_filename=demo_fp,
                                                   demographics_concern=larval_habitat_multiplier
                                                   )
            self.assertTrue(os.path.exists(demo_fp))

    def test_from_invalid_country(self):
        with self.assertRaises(ValueError) as cm:
            WorldBankBirthRateNodeConcern(country='Test1')
        self.assertIn('Cannot locate country Test1', str(cm.exception))

    def test_alt_columns(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_alt_columns_grid.csv')
            DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                   demographics_filename=demo_fp,
                                                   latitude_column_name='latitude',
                                                   longitude_column_name='lo',
                                                   population_column_name='population')
            self.assertTrue(os.path.exists(demo_fp))

    def test_alt_lat_missing(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_alt_columns_grid.csv')
            with self.assertRaises(ValueError) as cm:
                DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                       demographics_filename=demo_fp,
                                                       latitude_column_name='lat',
                                                       longitude_column_name='lo',
                                                       population_column_name='population')
            self.assertIn('Column lat is required', str(cm.exception))

    def test_alt_lon_missing(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_alt_columns_grid.csv')
            with self.assertRaises(ValueError) as cm:
                DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                       demographics_filename=demo_fp,
                                                       latitude_column_name='latitude',
                                                       longitude_column_name='lon',
                                                       population_column_name='population')
            self.assertIn('Column lon is required', str(cm.exception))

    def test_compare_default_pop_vs_provided(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_default_pop_fp = os.path.join(demo_files_dir, "demographics1.json")
            demo_provided_pop_fp = os.path.join(demo_files_dir, "demographics2.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')
            # since our population field is called population and not the default pop
            # we will fallback to using country pop
            DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                   demographics_filename=demo_default_pop_fp)
            self.assertTrue(os.path.exists(demo_default_pop_fp))

            DemographicsGeneratorRE.from_grid_file(population_input_file=grid_file,
                                                   demographics_filename=demo_provided_pop_fp,
                                                   population_column_name='population')

            self.assertTrue(os.path.exists(demo_provided_pop_fp))

            demo_default = DemographicsFile.from_file(demo_default_pop_fp)
            demo_provided = DemographicsFile.from_file(demo_provided_pop_fp)

            # now look over the nodes
            for nodeid, prop in demo_default.nodes.items():
                # ensure all nodes exist in both
                self.assertIn(nodeid, demo_provided.nodes)
                # ignore population, length and extra attributes(birth rate is calculated
                ignore_keys = ['pop', '__len__', 'extra_attributes']
                self.assertTrue({k: v for k, v in prop.__dict__.items() if k not in ignore_keys} == \
                                {k: v for k, v in demo_provided.nodes[nodeid].__dict__.items() if k not in ignore_keys})
                # for one without, use default population
                self.assertEquals(prop.pop, 1000)
