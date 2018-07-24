import json
import unittest

from dtk.utils.Campaign.utils.SchemaParser import SchemaParser

import warnings
# warnings.simplefilter("ignore", ResourceWarning)


class TestSchema(unittest.TestCase):

    test_dict = {
        "Built_in_list": r"../tests/Built_in_list.json",
        "complex_list": r"../tests/complex_list.json",
        "complex_list_2": r"../tests/complex_list_2.json",
        "default": r"../tests/default.json",
        "enum": r"../tests/enum.json",
        "idmType_1": r"../tests/idmType_1.json",
        "idmType_2": r"../tests/idmType_2.json",
        "idmType_3": r"../tests/idmType_3.json",
        "idmType_WaningEffect": r"../tests/idmType_WaningEffect.json",
        "key_value": r"../tests/key_value.json",
        "key_value_1": r"../tests/key_value_1.json",
        "key_value_2": r"../tests/key_value_2.json",
        "list_dict": r"../tests/list_dict.json",
        "list_dict_1": r"../tests/list_dict_1.json",
        "list_dict_2": r"../tests/list_dict_2.json",
        "Sim_Types": r"../tests/Sim_Types.json",
        "simple_dict": r"../tests/simple_dict.json",
        "simple_list_1": r"../tests/simple_list_1.json",
        "simple_list_2": r"../tests/simple_list_2.json",
        "times_values": r"../tests/times_values.json",
        "times_values_1": r"../tests/times_values_1.json",
        "times_values_2": r"../tests/times_values_2.json",
        "SimpleVaccine": r"../tests/SimpleVaccine.json",
        "UsageDependentBednet": r"../tests/UsageDependentBednet.json",
        "vector_species_name_goes_here": r"../tests/vector_species_name_goes_here.json",
        "SimpleVaccine_Waning_Config": r"../tests/SimpleVaccine_Waning_Config.json",
        "ScaleLarvalHabitat": r"../tests/ScaleLarvalHabitat.json",
        "DynamicStringSet": r"../tests/DynamicStringSet.json",
        "CalendarEventCoordinator": r"../tests/CalendarEventCoordinator.json",
        "LarvalHabitatMultiplier": r"../tests/LarvalHabitatMultiplier.json",
        "DoseMap": r"../tests/DoseMap.json",
        "MutiNodeInterventionDistributor": r"../tests/MutiNodeInterventionDistributor.json"
    }

    @classmethod
    def select_test_file(cls, file_name):
        schema_file = cls.test_dict[file_name]
        return schema_file

    @staticmethod
    def load_json_data(schema_file):
        # http://www.neuraldump.net/2017/06/how-to-suppress-python-unittest-warnings/
        warnings.simplefilter("ignore", ResourceWarning)

        try:
            return json.load(open(schema_file, 'rb'))
        except IOError:
            raise Exception('Unable to read file: %s' % schema_file)

    def setUp(self):
        # print('setUp')
        # self.schema_parser = SchemaParser(r'../tests/schema_post.json')
        self.schema_parser = SchemaParser(r'../tests/new_schema_post.json')

    def tearDown(self):
        # print('tearDown')
        pass

    # @unittest.skip('Skip: test_DoseMap, succeed')
    # def test_DoseMap(self):
    #     test_file = self.select_test_file("DoseMap")
    #
    #     test_schema = self.load_json_data(test_file)
    #     json_str = json.dumps(test_schema, indent=3)
    #     print(json_str)
    #
    #     schema_output = self.schema_parser.parse(test_file)
    #     print(json.dumps(schema_output, indent=3))


    # @unittest.skip('Skip: test_MultiNodeInterventionDistributor, succeed')
    def test_MultiNodeInterventionDistributor(self):
        test_file = self.select_test_file("MutiNodeInterventionDistributor")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))


    @unittest.skip('Skip: test_LarvalHabitatMultiplier, succeed')
    def test_LarvalHabitatMultiplier(self):
        test_file = self.select_test_file("LarvalHabitatMultiplier")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Target_Demographic"]["enum"], list))
        # self.assertTrue(isinstance(schema_output["Target_Demographic"]["enum"], list))
        # self.assertListEqual(test_schema["Target_Demographic"]["enum"], schema_output["Target_Demographic"]["enum"])

    @unittest.skip('Skip: test_demo')
    def test_demo(self):
        keys = list(self.test_dict.keys())
        print('\n'.join(keys))

    @unittest.skip('Skip: test_default, no special handling, TBD...')
    def test_default(self):
        test_file = self.select_test_file("default")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_enum, succeed')
    def test_enum(self):
        test_file = self.select_test_file("enum")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["Target_Demographic"]["enum"], list))
        self.assertTrue(isinstance(schema_output["Target_Demographic"]["enum"], list))
        self.assertListEqual(test_schema["Target_Demographic"]["enum"], schema_output["Target_Demographic"]["enum"])

    @unittest.skip('Skip: test_Sim_Types, succeed')
    def test_Sim_Types(self):
        test_file = self.select_test_file("Sim_Types")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["SimpleVaccine"]["Sim_Types"], list))
        self.assertTrue(isinstance(schema_output["SimpleVaccine"]["Sim_Types"], list))
        self.assertListEqual(test_schema["SimpleVaccine"]["Sim_Types"], schema_output["SimpleVaccine"]["Sim_Types"])

    @unittest.skip('Skip: test_simple_dict, succeed')
    def test_simple_dict(self):
        test_file = self.select_test_file("simple_dict")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["SimpleBednet"]["Cost_To_Consumer"], dict))
        self.assertTrue(isinstance(schema_output["SimpleBednet"]["Cost_To_Consumer"], dict))
        self.assertDictEqual(test_schema["SimpleBednet"]["Cost_To_Consumer"], schema_output["SimpleBednet"]["Cost_To_Consumer"])

    @unittest.skip('Skip: test_simple_list_1, succeed')
    def test_simple_list_1(self):
        test_file = self.select_test_file("simple_list_1")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Using_Event"]["test"], list))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["test"], dict))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["test"]["default"], list))
        # self.assertTrue(len(test_schema["Using_Event"]["test"]), len(schema_output["Using_Event"]["test"]["default"]))
        # self.assertTrue(schema_output["Using_Event"]["test"]["type"], "list")
        # self.assertListEqual(test_schema["Using_Event"]["test"], schema_output["Using_Event"]["test"]["default"])

    @unittest.skip('Skip: test_simple_list_2, succeed')
    def test_simple_list_2(self):
        test_file = self.select_test_file("simple_list_2")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Using_Event"]["test"], list))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["test"], dict))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["test"]["default"], list))
        # self.assertTrue(len(test_schema["Using_Event"]["test"]), len(schema_output["Using_Event"]["test"]["default"]))
        # self.assertTrue(schema_output["Using_Event"]["test"]["type"], "list")
        # self.assertListEqual(test_schema["Using_Event"]["test"], schema_output["Using_Event"]["test"]["default"])

    @unittest.skip('Skip: test_list_dict, succeed')
    def test_list_dict(self):
        test_file = self.select_test_file("list_dict")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Times"], list))
        # self.assertTrue(isinstance(schema_output["Times"], dict))
        # self.assertTrue(len(test_schema["Times"]), 1)
        # self.assertTrue(test_schema["Times"][0], dict)
        # self.assertDictEqual(test_schema["Times"][0], schema_output["Times"])

    @unittest.skip('Skip: test_list_dict_1')
    def test_list_dict_1(self):
        test_file = self.select_test_file("list_dict_1")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Times"], list))
        # self.assertTrue(isinstance(schema_output["Times"], dict))
        # self.assertTrue(len(test_schema["Times"]), 1)
        # self.assertTrue(test_schema["Times"][0], dict)
        # self.assertDictEqual(test_schema["Times"][0], schema_output["Times"])

    @unittest.skip('Skip: test_list_dict_2')
    def test_list_dict_2(self):
        test_file = self.select_test_file("list_dict_2")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Times"], list))
        # self.assertTrue(isinstance(schema_output["Times"], dict))
        # self.assertTrue(len(test_schema["Times"]), 1)
        # self.assertTrue(test_schema["Times"][0], dict)
        # self.assertDictEqual(test_schema["Times"][0], schema_output["Times"])

    @unittest.skip('Skip: test_Built_in_list, succeed')
    def test_Built_in_list(self):
        test_file = self.select_test_file("Built_in_list")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["Using_Event"]["Built-in"], list))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["Built-in"], dict))
        # self.assertTrue(isinstance(schema_output["Using_Event"]["Built-in"]["default"], list))
        # self.assertEqual(len(test_schema["Using_Event"]["Built-in"]), len(schema_output["Using_Event"]["Built-in"]["default"]))
        # self.assertListEqual(test_schema["Using_Event"]["Built-in"],
        #                      schema_output["Using_Event"]["Built-in"]["default"])

    @unittest.skip('Skip: test_complex_list')
    def test_complex_list(self):
        test_file = self.select_test_file("complex_list")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_complex_list_2')
    def test_complex_list_2(self):
        test_file = self.select_test_file("complex_list_2")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_key_value, succeed')
    def test_key_value(self):
        test_file = self.select_test_file("key_value")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["idmType:NodePropertyRestrictions"], list))
        self.assertTrue(len(test_schema["idmType:NodePropertyRestrictions"]), 1)
        self.assertTrue(test_schema["idmType:NodePropertyRestrictions"][0], dict)
        self.assertTrue(set(test_schema["idmType:NodePropertyRestrictions"][0].keys()), ("<key>", "<value>"))

    @unittest.skip('Skip: test_key_value_1')
    def test_key_value_1(self):
        test_file = self.select_test_file("key_value_1")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertTrue(isinstance(test_schema["idmType:NodePropertyRestrictions"], list))
        # self.assertTrue(len(test_schema["idmType:NodePropertyRestrictions"]), 1)
        # self.assertTrue(test_schema["idmType:NodePropertyRestrictions"][0], dict)
        # self.assertTrue(set(test_schema["idmType:NodePropertyRestrictions"][0].keys()), ("<key>", "<value>"))

    # @unittest.skip('Skip: test_key_value_2, succeed')
    # def test_key_value_2(self):
    #     test_file = self.select_test_file("key_value_2")
    #
    #     test_schema = self.load_json_data(test_file)
    #     json_str = json.dumps(test_schema, indent=3)
    #     print(json_str)
    #
    #     schema_output = self.schema_parser.parse(test_file)
    #     print(json.dumps(schema_output, indent=3))
    #
    #     # self.assertTrue(test_schema["CalendarEventCoordinator"]["Node_Property_Restrictions"]["type"],
    #     #                            "idmType:NodePropertyRestrictions")
    #     # self.assertTrue(schema_output["CalendarEventCoordinator"]["Node_Property_Restrictions"]["type"], "list")
    #     # self.assertTrue(isinstance(schema_output["CalendarEventCoordinator"]["Node_Property_Restrictions"]["item_type"],
    #     #                 dict))
    #     # self.assertEqual(schema_output["CalendarEventCoordinator"]["Node_Property_Restrictions"]["item_type"]["type"], "dict")
    #     # self.assertTrue(set(schema_output["CalendarEventCoordinator"]["Node_Property_Restrictions"]["item_type"].keys()),
    #     #                 ("type", "key_type", "value_type"))

    @unittest.skip('Skip: test_times_values, succeed')
    def test_times_values(self):
        test_file = self.select_test_file("times_values")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))
    #
    #     self.assertTrue(isinstance(test_schema["idmType:InterpolatedValueMap"], dict))
    #     self.assertTrue(set(test_schema["idmType:InterpolatedValueMap"].keys()), ("Times", "Values"))
    #     self.assertTrue(isinstance(test_schema["idmType:InterpolatedValueMap"]["Times"], list))
    #     self.assertTrue(isinstance(test_schema["idmType:InterpolatedValueMap"]["Values"], list))
    #     self.assertTrue(len(test_schema["idmType:InterpolatedValueMap"]["Times"]), 1)
    #     self.assertTrue(len(test_schema["idmType:InterpolatedValueMap"]["Values"]), 1)
    #     self.assertTrue(isinstance(test_schema["idmType:InterpolatedValueMap"]["Times"][0], dict))
    #     self.assertTrue(isinstance(test_schema["idmType:InterpolatedValueMap"]["Values"][0], dict))
    #
    #     self.assertTrue(isinstance(schema_output["idmType:InterpolatedValueMap"], dict))
    #     self.assertTrue(set(schema_output["idmType:InterpolatedValueMap"].keys()), ("Times", "Values"))
    #     self.assertTrue(isinstance(schema_output["idmType:InterpolatedValueMap"]["Times"], dict))
    #     self.assertTrue(isinstance(schema_output["idmType:InterpolatedValueMap"]["Values"], dict))

    @unittest.skip('Skip: test_times_values_1, succeed')
    def test_times_values_1(self):
        print(111)
        test_file = self.select_test_file("times_values_1")
        print(test_file)

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["WaningEffectMapCount"]["Durability_Map"], dict))
        self.assertEqual(test_schema["WaningEffectMapCount"]["Durability_Map"]["type"], "idmType:InterpolatedValueMap")

        self.assertTrue(isinstance(schema_output["WaningEffectMapCount"]["Durability_Map"], dict))
        self.assertTrue(set(schema_output["WaningEffectMapCount"]["Durability_Map"].keys()), ("Times", "Values"))
        self.assertTrue(isinstance(schema_output["WaningEffectMapCount"]["Durability_Map"]["Times"], dict))
        self.assertTrue(isinstance(schema_output["WaningEffectMapCount"]["Durability_Map"]["Values"], dict))

    @unittest.skip('Skip: test_times_values_2, succeed')
    def test_times_values_2(self):
        print(333)
        test_file = self.select_test_file("times_values_2")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["test:InterpolatedValueMap"], dict))
        self.assertTrue(set(test_schema["test:InterpolatedValueMap"].keys()), ("Times", "Values", "Tests"))
        self.assertTrue(isinstance(test_schema["test:InterpolatedValueMap"]["Times"], list))
        self.assertTrue(isinstance(test_schema["test:InterpolatedValueMap"]["Values"], list))
        self.assertTrue(isinstance(test_schema["test:InterpolatedValueMap"]["Tests"], list))
        self.assertTrue(len(test_schema["test:InterpolatedValueMap"]["Times"]), 1)
        self.assertTrue(len(test_schema["test:InterpolatedValueMap"]["Values"]), 1)
        self.assertTrue(len(test_schema["test:InterpolatedValueMap"]["Tests"]), 1)

        self.assertTrue(isinstance(schema_output["test:InterpolatedValueMap"], dict))
        self.assertTrue(set(schema_output["test:InterpolatedValueMap"].keys()), ("Times", "Values", "Tests"))
        self.assertTrue(isinstance(schema_output["test:InterpolatedValueMap"]["Times"], dict))
        self.assertTrue(isinstance(schema_output["test:InterpolatedValueMap"]["Values"], dict))
        self.assertTrue(isinstance(schema_output["test:InterpolatedValueMap"]["Tests"], dict))



    @unittest.skip('Skip: [TODO] test_idmType_2, succeed')
    def test_idmType_2(self):
        test_file = self.select_test_file("idmType_2")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertTrue(isinstance(test_schema["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"], dict))
        self.assertEqual(test_schema["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"]["type"], "idmType:LarvalHabitatMultiplier")

        self.assertTrue(isinstance(schema_output["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"], dict))
        self.assertEqual(schema_output["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"]["type"], "list")
        self.assertTrue("item_type" in schema_output["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"])
        self.assertTrue(isinstance(schema_output["ScaleLarvalHabitat"]["Larval_Habitat_Multiplier"]["default"], list))

    @unittest.skip('Skip: test_idmType_3, succeed')
    def test_idmType_3(self):
        test_file = self.select_test_file("idmType_3")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertEqual(test_schema["vector_species_name_goes_here"]["Larval_Habitat_Types"]["type"], "idmType:LarvalHabitats")
        self.assertEqual(schema_output["vector_species_name_goes_here"]["Larval_Habitat_Types"]["type"], "object")
        self.assertEqual(schema_output["vector_species_name_goes_here"]["Larval_Habitat_Types"]["subclasses"], "LarvalHabitats")

    @unittest.skip('Skip: test_idmType_WaningEffect, succeed')
    def test_idmType_WaningEffect(self):
        test_file = self.select_test_file("idmType_WaningEffect")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertEqual(test_schema["SimpleVaccine"]["Waning_Config"]["type"], "idmType:WaningEffect")
        self.assertEqual(schema_output["SimpleVaccine"]["Waning_Config"]["type"], "idmType:WaningEffect")
        self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_SimpleVaccine')
    def test_SimpleVaccine(self):
        test_file = self.select_test_file("SimpleVaccine")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_UsageDependentBednet')
    def test_UsageDependentBednet(self):
        test_file = self.select_test_file("UsageDependentBednet")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_vector_species_name_goes_here')
    def test_vector_species_name_goes_here(self):
        test_file = self.select_test_file("vector_species_name_goes_here")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_ScaleLarvalHabitat')
    def test_ScaleLarvalHabitat(self):
        test_file = self.select_test_file("ScaleLarvalHabitat")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])

    @unittest.skip('Skip: test_DynamicStringSet')
    def test_DynamicStringSet(self):
        test_file = self.select_test_file("DynamicStringSet")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))


    @unittest.skip('Skip: test_CalendarEventCoordinator')
    def test_CalendarEventCoordinator(self):
        test_file = self.select_test_file("CalendarEventCoordinator")

        test_schema = self.load_json_data(test_file)
        json_str = json.dumps(test_schema, indent=3)
        print(json_str)

        schema_output = self.schema_parser.parse(test_file)
        print(json.dumps(schema_output, indent=3))

        # self.assertDictEqual(test_schema["SimpleVaccine"]["Waning_Config"], schema_output["SimpleVaccine"]["Waning_Config"])