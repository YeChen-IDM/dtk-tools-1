import json
import os
import random
import tempfile
import unittest
from collections import defaultdict

import numpy as np

from dtk.tools.climate.WeatherNode import WeatherNode
from dtk.tools.demographics.DemographicsFile import DemographicsFile
from dtk.tools.migration.GravityModelRatesGenerator import GravityModelRatesGenerator
from dtk.tools.migration.MigrationGenerator import MigrationGenerator, MigrationTypes
from dtk.tools.migration.StaticGravityModelRatesGenerator import StaticGravityModelRatesGenerator
from dtk.tools.migration.StaticLinkRatesModelGenerator import StaticLinkRatesModelGenerator
from simtools.SetupParser import SetupParser

demographics_file = os.path.join(os.path.dirname(__file__), 'migration_generator_test_demographics.json')


class MigrationGeneratorTests(unittest.TestCase):
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

    def get_demographics_from_grid_file(self):
        return demographics_file

    def test_graph_required_for_gravity_model(self):
        with self.assertRaises(ValueError) as cm:
            m = GravityModelRatesGenerator(None)
        self.assertIn('A Graph Generator is required for the GravityModelRatesGenerator', str(cm.exception))

    def test_static_gravity_rates_generator_not_enough_gravity_params(self):
        with self.assertRaises(ValueError) as cm:
            m = StaticGravityModelRatesGenerator(demographics_file,
                                                 np.array([7.50395776e-06, 9.65648371e-01, 9.65648371e-01])
                                                 )
        self.assertIn('You must provide all 4 gravity params', str(cm.exception))

    # TODO Add tests for gravity rates model generator with migration generator. One test using SmallWorldGrid GeoGraph
    # and the other using GeoGraphGenerator

    def test_static_link_rates_generator(self):
        link_rates = defaultdict(dict)
        for x in range(1, 29):
            for y in range(1, 29):
                if x != y:
                    if y not in link_rates[x]:
                        weight = random.random()
                        link_rates[x][y] = weight
                        link_rates[y][x] = weight

        model = StaticLinkRatesModelGenerator(link_rates)
        result = model.generate()

        self.assertEqual(json.dumps(link_rates, sort_keys=True), json.dumps(result, sort_keys=True))

        with tempfile.TemporaryDirectory() as output_path:
            migration_file_name = os.path.join(output_path, 'migration.bin')

            m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                                   link_rates_model=model)
            m.generate_migration(True, demographics_file_path=demographics_file)
            # TODO verify output file
            # Check that the binary file exists
            self.assertTrue(os.path.exists(migration_file_name))
            # Check that the header exists
            self.assertTrue(migration_file_name.replace('.bin', '.json'))
            # Check that the human readable form exists
            self.assertTrue(migration_file_name.replace('.bin', '.txt'))



    def test_static_gravity_rates_generator(self):
        with tempfile.TemporaryDirectory() as output_path:
            migration_file_name = os.path.join(output_path, 'migration.bin')

            static_gravity_params = np.array([7.50395776e-06, 9.65648371e-01, 9.65648371e-01, -1.10305489e+00])

            # Build our StaticGravityModelRatesGenerator
            link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

            # Now use the migratio generator to produce the binary file
            m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                                   link_rates_model=link_rates_model)
            m.generate_migration(True, demographics_file_path=demographics_file)

            # TODO verify output file
            # Check that the binary file exists
            self.assertTrue(os.path.exists(migration_file_name))
            # Check that the header exists
            self.assertTrue(migration_file_name.replace('.bin', '.json'))
            # Check that the human readable form exists
            self.assertTrue(migration_file_name.replace('.bin', '.txt'))
