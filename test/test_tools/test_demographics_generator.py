import os
import tempfile
import unittest

from dtk.generic.demographics import distribution_types
from dtk.tools.demographics.DemographicsFile import DemographicsFile
from dtk.tools.demographics.DemographicsGenerator import DemographicsGenerator
from dtk.tools.demographics.DemographicsGeneratorConcern import WorldBankBirthRateConcern, \
    DefaultsDictionaryGeneratorConcern, DemographicsGeneratorConcernChain, DefaultIndividualAttributesConcern, \
    EquilibriumAgeDistributionConcern, StaticLevelBirthRateConcern, demographics_generator_node_concern
from dtk.tools.demographics.Node import Node
from simtools.SetupParser import SetupParser


@demographics_generator_node_concern(larval_habitat_multiplier=1.0, calib_single_node_pop=1000.0)
def larval_habitat_multiplier(defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict,
                              larval_habitat_multiplier=1.0, calib_single_node_pop: float = 1000.0):
    pop_multiplier = node_attributes['InitialPopulation'] / calib_single_node_pop
    if 'LarvalHabitatMultiplier' not in node_attributes:
        node_attributes['LarvalHabitatMultiplier'] = larval_habitat_multiplier

    node_attributes['LarvalHabitatMultiplier'] *= pop_multiplier


class DemographicsGeneratorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        SetupParser.init()

    def test_using_dict(self):
        # This test shows using just a simple set of dictionary values you want to be applied to demographics
        # In this case, the dictionary will only update demographics default
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')

            exponential_age_param = 0.0001068
            population_removal_rate = 23

            # Build our individual_attributes and node attributes we want to be given as part of the defaults

            mod_mortality = {
                "NumDistributionAxes": 2,
                "AxisNames": ["gender", "age"],
                "AxisUnits": ["male=0,female=1", "years"],
                "AxisScaleFactors": [1, 365],
                "NumPopulationGroups": [2, 1],
                "PopulationGroups": [
                    [0, 1],
                    [0]
                ],
                "ResultUnits": "annual deaths per 1000 individuals",
                "ResultScaleFactor": 2.74e-06,
                "ResultValues": [
                    [population_removal_rate],
                    [population_removal_rate]
                ]
            }

            individual_attributes = {
                "MortalityDistribution": mod_mortality,
                "AgeDistributionFlag": distribution_types["EXPONENTIAL_DISTRIBUTION"],
                "AgeDistribution1": exponential_age_param,
                "RiskDistribution1": 1,
                "PrevalenceDistributionFlag": 1,
                "AgeDistribution2": 0,
                "PrevalenceDistribution1": 0.13,
                "PrevalenceDistribution2": 0.15,
                "RiskDistributionFlag": 0,
                "RiskDistribution2": 0,
                "MigrationHeterogeneityDistribution1": 1,
                "ImmunityDistributionFlag": 0,
                "MigrationHeterogeneityDistributionFlag": 0,
                "ImmunityDistribution1": 1,
                "MigrationHeterogeneityDistribution2": 0,
                "ImmunityDistribution2": 0
            }

            node_attributes = {
                "Urban": 0,
                "AbovePoverty": 0.5,
                "Region": 1,
                "Seaport": 0,
                "Airport": 0,
                "Altitude": 0
            }

            d = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                     demographics_filename=demo_fp,
                                                     concerns=DefaultsDictionaryGeneratorConcern(
                                                         individual_attributes, node_attributes)
                                                     )


    def test_chain(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')
            pop_removal_rate = 31.2
            concerns = DemographicsGeneratorConcernChain.from_list(
                [
                    DefaultIndividualAttributesConcern(prevalence1=0.19, population_removal_rate=pop_removal_rate),
                    StaticLevelBirthRateConcern(pop_removal_rate),
                    EquilibriumAgeDistributionConcern(default_birth_rate=31.2)
                ]

            )
            d = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                     demographics_filename=demo_fp,
                                                     concerns=concerns
                                                     )


    def test_from_grid_file(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_grid.csv')
            DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                 demographics_filename=demo_fp,
                                                 concerns=[larval_habitat_multiplier]
                                                 )
            self.assertTrue(os.path.exists(demo_fp))

    def test_from_invalid_country(self):
        with self.assertRaises(ValueError) as cm:
            WorldBankBirthRateConcern(country='Test1')
        self.assertIn('Cannot locate country Test1', str(cm.exception))

    def test_alt_columns(self):
        with tempfile.TemporaryDirectory() as output_dir:
            demo_files_dir = os.path.join(output_dir, 'Demographics')
            if not os.path.exists(demo_files_dir):
                os.mkdir(demo_files_dir)

            demo_fp = os.path.join(demo_files_dir, "demographics.json")
            grid_file = os.path.join(os.path.dirname(__file__), 'test_alt_columns_grid.csv')
            DemographicsGenerator.from_grid_file(population_input_file=grid_file,
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
                DemographicsGenerator.from_grid_file(population_input_file=grid_file,
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
                DemographicsGenerator.from_grid_file(population_input_file=grid_file,
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
            DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                 demographics_filename=demo_default_pop_fp)
            self.assertTrue(os.path.exists(demo_default_pop_fp))

            DemographicsGenerator.from_grid_file(population_input_file=grid_file,
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
                self.assertEqual(prop.pop, 1000)
