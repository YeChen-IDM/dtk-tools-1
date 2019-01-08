import os
import tempfile
import unittest
import csv
import json
import copy

from dtk.generic.demographics import distribution_types
from dtk.tools.demographics.DemographicsFile import DemographicsFile
from dtk.tools.demographics.DemographicsGenerator import DemographicsGenerator, InvalidResolution
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
    def __init__(self, *args, **kwargs):
        self.temp_files = []
        super(DemographicsGeneratorTest, self).__init__(*args, **kwargs)


    @classmethod
    def setUpClass(cls):
        SetupParser.init()

    def tearDown(self):
        # delete any persistent temp files that have been created
        for temp_file in self.temp_files:
            os.remove(temp_file)

    def diff_nodelists(self, first, second, lat_keys=[], long_keys=[], pop_keys=[]):
        """
        Compare two lists of nodes or node-like objects. A "node" is expected to be a dictionary containing
        latitude, longitude, and population fields (either in a 'NodeAttributes' sub-dictionary or at the root level).

        :param first: first list of nodes
        :param second: second list of nodes (to compare to the first)
        :param lat_keys: custom latitude key names
        :param long_keys: custom longitude key names
        :param pop_keys: custom population key names
        :return: list of node representations only in the 1st list, and a list of node representations only in the 2nd
        """
        lat_names = ['lat', 'Latitude'] + lat_keys
        long_names = ['lon', 'Longitude'] + long_keys
        pop_names = ['pop', 'InitialPopulation'] + pop_keys

        def canonicalize_node_rep(node):
            """
            Nodes are canonicalized into string representations of the form: "lat:12.0263 long:-1.6071 pop:1500" to make
            comparison and set differences easier to compute.

            :param node: node or node-like dictionary object
            :return: string representation of the node
            """
            # for actual nodes start with the NodeAttributes dictionary
            if 'NodeAttributes' in node:
                node = node['NodeAttributes']
            # pick the value for the first key that matches any of the possible lat/long/pop keys
            latitude = next((node[name] for name in lat_names if name in node), '')
            longitude = next((node[name] for name in long_names if name in node), '')
            pop = next((node[name] for name in pop_names if name in node), '')
            # round lat/long to 4 decimal digits (roughly 11 meter accuracy)
            return f'lat:{float(latitude):0.4f} long:{float(longitude):0.4f} pop:{pop}'

        first_nodes = set(map(canonicalize_node_rep, first))
        second_nodes = set(map(canonicalize_node_rep, second))

        return (first_nodes - second_nodes), (second_nodes - first_nodes)

    def assert_nodelists_equal(self, first, second, lat_keys=[], long_keys=[], pop_keys=[]):
        """
        Assert that two lists of nodes are identical.

        :param first: first list of nodes
        :param second: second list of nodes (to compare to the first)
        :param lat_keys: custom latitude key names
        :param long_keys: custom longitude key names
        :param pop_keys: custom population key names
        """
        only_first, only_second = self.diff_nodelists(first, second, lat_keys, long_keys, pop_keys)
        self.assertEqual(len(only_first) + len(only_second), 0,
                     f'Expected nodelists to be equal, only in first: ({only_first}), only in second: ({only_second})')

    def dump_json_to_temp_csv(self, json_data):
        """
        Create a temp file containing the same contents in csv form as a json blob.

        :param json_data: json style data blob
        :return: csv filename
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_tempfile:
            csv_writer = csv.writer(csv_tempfile)
            csv_writer.writerow(json_data[0].keys())
            for data in json_data:
                csv_writer.writerow(data.values())
            # keep track of temp files written so we can clean them up at the end of the test
            self.temp_files.append(csv_tempfile.name)
            return csv_tempfile.name

    def test_missing_lat(self):
        raw_nodes = '[{"id": "0", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            with self.assertRaises(ValueError) as cm:
                DemographicsGenerator.from_grid_file(population_input_file=grid_file, demographics_filename=demo_fp)
            self.assertIn('Column lat is required', str(cm.exception))

    def test_missing_long(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            with self.assertRaises(ValueError) as cm:
                DemographicsGenerator.from_grid_file(population_input_file=grid_file, demographics_filename=demo_fp)
            self.assertIn('Column lon is required', str(cm.exception))

    def test_custom_lat(self):
        raw_nodes = '[{"id": "0", "wye": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        latitude_column_name='wye'
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], lat_keys=['wye'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_custom_lat_missing(self):
        raw_nodes = '[{"id": "0", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            with self.assertRaises(ValueError) as cm:
                demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                            demographics_filename=demo_fp,
                                                            latitude_column_name='wye'
                                                            )
            self.assertIn('Column wye is required', str(cm.exception))

    def test_custom_long(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "eks": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        longitude_column_name='eks'
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], long_keys=['eks'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_custom_long_missing(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            with self.assertRaises(ValueError) as cm:
                demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                            demographics_filename=demo_fp,
                                                            latitude_column_name='eks'
                                                            )
            self.assertIn('Column eks is required', str(cm.exception))

    def test_custom_pop(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "peeps": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        population_column_name='peeps'
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], pop_keys=['peeps'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_all_custom_columns(self):
        raw_nodes = '[{"id": "0", "wye": "12.0263354369855", "eks": "-1.60712345544773", "gcid": "1", "peeps": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        latitude_column_name='wye',
                                                        longitude_column_name='eks',
                                                        population_column_name='peeps'
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], lat_keys=['wye'], long_keys=['eks'], pop_keys=['peeps'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_res_30(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        res_in_arcsec=30
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], pop_keys=['peeps'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_res_250(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp,
                                                        res_in_arcsec=250
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'], pop_keys=['peeps'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_res_invalid(self):
        raw_nodes = '[{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}]'
        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            with self.assertRaises(InvalidResolution) as cm:
                demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                            demographics_filename=demo_fp,
                                                            res_in_arcsec=100
                                                            )
            self.assertIn('is not a valid arcsecond resolution', str(cm.exception))

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
        raw_nodes = '''
            [{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1", "pop": "1500"}, 
            {"id": "1", "lat": "12.0308553124896", "lon": "-1.60712345544773", "gcid": "2", "pop": "1763"}, 
            {"id": "2", "lat": "12.0353751865335", "lon": "-1.60712345544773", "gcid": "3", "pop": "343"}, 
            {"id": "3", "lat": "12.0398950591167", "lon": "-1.60712345544773", "gcid": "4", "pop": "12"}, 
            {"id": "4", "lat": "12.0444149302386", "lon": "-1.60712345544773", "gcid": "5", "pop": "989"}, 
            {"id": "5", "lat": "12.0489347998988", "lon": "-1.60712345544773", "gcid": "6", "pop": "1567"}, 
            {"id": "6", "lat": "12.0534546680966", "lon": "-1.60712345544773", "gcid": "7", "pop": "1785"}, 
            {"id": "7", "lat": "12.0579745348316", "lon": "-1.60712345544773", "gcid": "8", "pop": "2242"}, 
            {"id": "8", "lat": "12.0624944001034", "lon": "-1.60712345544773", "gcid": "9", "pop": "343"}, 
            {"id": "9", "lat": "12.0670142639113", "lon": "-1.60712345544773", "gcid": "10", "pop": "12"}, 
            {"id": "10", "lat": "12.0715341262548", "lon": "-1.60712345544773", "gcid": "11", "pop": "146"}, 
            {"id": "11", "lat": "12.0760539871335", "lon": "-1.60712345544773", "gcid": "12", "pop": "158"}, 
            {"id": "12", "lat": "12.0805738465468", "lon": "-1.60712345544773", "gcid": "13", "pop": "777"}, 
            {"id": "13", "lat": "12.0850937044942", "lon": "-1.60712345544773", "gcid": "14", "pop": "86"}, 
            {"id": "14", "lat": "12.0896135609752", "lon": "-1.60712345544773", "gcid": "15", "pop": "40"}, 
            {"id": "15", "lat": "12.0941334159893", "lon": "-1.60712345544773", "gcid": "16", "pop": "35"}, 
            {"id": "16", "lat": "12.0986532695359", "lon": "-1.60712345544773", "gcid": "17", "pop": "39"}, 
            {"id": "17", "lat": "12.1031731216146", "lon": "-1.60712345544773", "gcid": "18", "pop": "765"}, 
            {"id": "18", "lat": "12.1076929722248", "lon": "-1.60712345544773", "gcid": "19", "pop": "123"}, 
            {"id": "19", "lat": "12.1122128213661", "lon": "-1.60712345544773", "gcid": "20", "pop": "654"}, 
            {"id": "20", "lat": "12.1167326690378", "lon": "-1.60712345544773", "gcid": "21", "pop": "543"}, 
            {"id": "21", "lat": "12.1212525152395", "lon": "-1.60712345544773", "gcid": "22", "pop": "452"}, 
            {"id": "22", "lat": "12.1257723599707", "lon": "-1.60712345544773", "gcid": "23", "pop": "456"}, 
            {"id": "23", "lat": "12.1302922032309", "lon": "-1.60712345544773", "gcid": "24", "pop": "78"}, 
            {"id": "24", "lat": "12.1348120450195", "lon": "-1.60712345544773", "gcid": "25", "pop": "12"}, 
            {"id": "25", "lat": "12.139331885336", "lon": "-1.60712345544773", "gcid": "26", "pop": "15"}, 
            {"id": "26", "lat": "12.14385172418", "lon": "-1.60712345544773", "gcid": "27", "pop": "16"}, 
            {"id": "27", "lat": "12.1483715615508", "lon": "-1.60712345544773", "gcid": "28", "pop": "19"}]
        '''

        nodes = json.loads(raw_nodes)
        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp
                                                        )
            self.assert_nodelists_equal(nodes, demo['Nodes'])
            self.assertTrue(os.path.exists(demo_fp))

    def test_from_invalid_country(self):
        with self.assertRaises(ValueError) as cm:
            WorldBankBirthRateConcern(country='Test1')
        self.assertIn('Cannot locate country Test1', str(cm.exception))

    def test_default_pop(self):
        raw_nodes = '''
            [{"id": "0", "lat": "12.0263354369855", "lon": "-1.60712345544773", "gcid": "1"}, 
            {"id": "1", "lat": "12.0308553124896", "lon": "-1.60712345544773", "gcid": "2"}, 
            {"id": "2", "lat": "12.0353751865335", "lon": "-1.60712345544773", "gcid": "3"}, 
            {"id": "3", "lat": "12.0398950591167", "lon": "-1.60712345544773", "gcid": "4"}, 
            {"id": "4", "lat": "12.0444149302386", "lon": "-1.60712345544773", "gcid": "5"}, 
            {"id": "5", "lat": "12.0489347998988", "lon": "-1.60712345544773", "gcid": "6"}, 
            {"id": "6", "lat": "12.0534546680966", "lon": "-1.60712345544773", "gcid": "7"}]
             '''

        nodes = json.loads(raw_nodes)
        expected_nodes = copy.deepcopy(nodes)
        for node in expected_nodes:
            node['pop'] = 1000

        grid_file = self.dump_json_to_temp_csv(nodes)

        with tempfile.TemporaryDirectory() as output_dir:
            demo_fp = os.path.join(output_dir, "demographics.json")
            demo = DemographicsGenerator.from_grid_file(population_input_file=grid_file,
                                                        demographics_filename=demo_fp
                                                        )
            self.assert_nodelists_equal(expected_nodes, demo['Nodes'])
            self.assertTrue(os.path.exists(demo_fp))

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

if __name__ == '__main__':
    unittest.main()
