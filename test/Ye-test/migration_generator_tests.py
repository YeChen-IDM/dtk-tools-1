import os
import unittest
import numpy as np
import shutil
import random
import json
from typing import Union

from dtk.tools.migration.GravityModelRatesGenerator import GravityModelRatesGenerator
from dtk.tools.migration.MigrationGenerator import MigrationGenerator
from dtk.tools.migration.MigrationFile import MigrationTypes
from dtk.tools.migration.StaticGravityModelRatesGenerator import StaticGravityModelRatesGenerator
from simtools.SetupParser import SetupParser
from generate_migration_file import generate_txt_from_demo, generate_txt_from_bin, \
    generate_migration_files_from_txt
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.AssetManager.FileList import FileList
from simtools.AssetManager.AssetCollection import AssetCollection
from dtk.utils.reports.CustomReport import *

demographics_file = os.path.join(os.path.dirname(__file__), 'migration_files',
                                 'migration_generator_test_demographics.json')


static_gravity_params = np.array([7.50395776e-06, 9.65648371e-01, 9.65648371e-01, -1.10305489e+00])


class MigrationGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "migration_generator_test"
        if not os.path.isdir(cls.output_path):
            os.mkdir(cls.output_path)

    def setUp(self):
        self.test_output = os.path.join(self.output_path, self._testMethodName)
        if os.path.isdir(self.test_output):
            shutil.rmtree(self.test_output)
        os.mkdir(self.test_output)

    # region test methods
    def convert_bin_file_and_verify(self, migration_file_name: str):
        """
        verify migration binary file is correct by converting it into text file using generate_txt_from_bin(). Compare
        this converted file with expected migration text file.
        :param migration_file_name:
        :return:
        """
        # Convert binary to text file
        migration_file_converted_from_bin = os.path.join(self.test_output, 'converted_migration.txt')
        generate_txt_from_bin(migration_file_name, demographics_file, migration_file_converted_from_bin, 27)
        # Compare
        self.verify_txt_file(migration_file_name, migration_file_converted_from_bin)

    def verify_txt_file_based_on_demo(self, migration_file_name: str, exclude_nodes: Union[list, None]=None):
        """
        verify migration text file is correct by comparing it with expected migration text file generated with
        generate_txt_from_demo() which uses the same math as Static Gravity Model(StaticGravityModelRatesGenerator).
        :param migration_file_name:
        :param exclude_nodes:
        :return:
        """
        # Generate expected migration text file
        migration_file_calculated_SGM = os.path.join(self.test_output, 'expected_migration.txt')
        generate_txt_from_demo(demographics_file,
                               static_gravity_params,
                               migration_file_calculated_SGM,
                               exclude_nodes)
        # Compare
        self.verify_txt_file(migration_file_name, migration_file_calculated_SGM)

    def verify_txt_file(self, migration_file_name: str, expected_migration_txt_file: str):
        """
        Verify that both migration text files have the same FromNode and DestNode. The corresponding migration rates
        are almost equal.
        :param migration_file_name:
        :param expected_migration_txt_file:
        :return:
        """
        with open(expected_migration_txt_file, 'r') as expected_rates_file:
            expected_rates = expected_rates_file.readlines()
        with open(migration_file_name.replace('.bin', '.txt'), 'r') as rates_file:
            rates = rates_file.readlines()
        self.assertEqual(len(expected_rates), len(rates))

        # Compare migration.txt with expected_migration.txt
        for i in range(len(expected_rates)):
            expected_values = expected_rates[i].split()
            values = rates[i].split()
            # [FromNode, DestNode] match the expected values
            self.assertEqual(expected_values[:2], values[:2])
            # Migration rates are almost equal.
            self.assertAlmostEqual(float(expected_values[2]), float(values[2]), places=3,
                                   msg=f"Migration rate from node {expected_values[0]} to node {expected_values[1]} is "
                                       f"{values[2]}, expeted {expected_values[2]}.")

    def verify_migration_files_generated(self, migration_file_name: str):
        """
        Verify that the three migration files(.txt, .bin.json and .bin) exist.
        :param migration_file_name:
        :return:
        """
        # Check that the binary file exists
        self.assertTrue(os.path.exists(migration_file_name))
        # Check that the header exists
        self.assertTrue(os.path.exists(migration_file_name + '.json'))
        # Check that the human readable form exists
        self.assertTrue(os.path.exists(migration_file_name.replace('.bin', '.txt')))
    # endregion

    # region Bad cases
    def test_graph_required_for_gravity_model(self):
        with self.assertRaises(ValueError) as cm:
            m = GravityModelRatesGenerator("")
        self.assertIn('A Graph Generator is required for the GravityModelRatesGenerator', str(cm.exception))

    def test_static_gravity_rates_generator_not_enough_gravity_params(self):
        with self.assertRaises(ValueError) as cm:
            m = StaticGravityModelRatesGenerator(demographics_file,
                                                 np.array([7.50395776e-06, 9.65648371e-01, 9.65648371e-01])
                                                 )
        self.assertIn('You must provide all 4 gravity params', str(cm.exception))

    def test_static_gravity_rates_generator_no_demog_file(self):
        with self.assertRaises(ValueError) as cm:
            m = StaticGravityModelRatesGenerator("test_demo_file.json",
                                                 static_gravity_params)

        self.assertIn('A demographics file is required', str(cm.exception))

    def test_link_rates_model_required_for_migration_generator(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')
        with self.assertRaises(ValueError) as cm:
            link_rates_model = "foo"
            m = MigrationGenerator(migration_file_name,
                                   migration_type=MigrationTypes.local,
                                   link_rates_model=link_rates_model)

        self.assertIn('A Link Rates Model Generator is required', str(cm.exception))

    def test_migration_type_for_migration_generator(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        with self.assertRaises(ValueError) as cm:
            m = MigrationGenerator(migration_file_name,
                                   migration_type='local',
                                   link_rates_model=link_rates_model)

        self.assertIn('A MigrationTypes is required', str(cm.exception))

    def test_migration_generator_no_idRef(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)

        with self.assertRaises(ValueError) as cm:
            m.generate_migration(True)

        self.assertIn('An idRef is required', str(cm.exception))

    # region Happy cases
    def test_migration_generator_local(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        self.verify_migration_files_generated(migration_file_name)

    def test_migration_generator_regional(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.regional,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        self.verify_migration_files_generated(migration_file_name)

    def test_migration_generator_sea(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.sea,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        self.verify_migration_files_generated(migration_file_name)

    def test_migration_generator_air(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.air,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        self.verify_migration_files_generated(migration_file_name)

    def test_migration_generator_in_comps(self):
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(False, demographics_file_path=demographics_file)

        # Now runs in Comps
        # Setup Comps connection
        default_block = 'HPC'
        SetupParser.init(selected_block=default_block)

        cb = DTKConfigBuilder.from_files(config_name=os.path.join("migration_files", "config.json"),
                                         campaign_name=os.path.join("migration_files", "campaign.json"),
                                         Enable_Local_Migration=1,
                                         Local_Migration_Filename='migration.bin')

        # Save all required files into asset collection
        fl = FileList()
        fl.add_path("migration_files", recursive=True)
        fl.add_file(migration_file_name)
        fl.add_file(migration_file_name + '.json')
        collection = AssetCollection(local_files=fl)
        collection.prepare(location=default_block)
        cb.set_collection_id(collection.collection_id)

        # Add custom reports
        add_human_migration_tracking_report(cb)
        add_node_demographics_report(cb)
        # cb.set_dll_root("migration_files") # no need for this line if report_plugins is inside Assets

        exp_manager = ExperimentManagerFactory.init()
        run_sim_args = {
            'exp_name': 'Migration_Generator_' + str(self._testMethodName),
            'config_builder': cb
        }
        exp_manager.run_simulations(**run_sim_args)

        exp_manager.wait_for_finished(verbose=True)

        # Verify the simulation runs successfully.
        self.assertTrue(exp_manager.succeeded(), "SimulationState is not Succeeded.")

        print("SimulationState is Succeeded, please see sim output in Comps.\n")

    def test_migration_generator_verify_txt_file(self):
        """
        Testing migration text file from MigrationGenerator with text file generated from generate_txt_from_demo()
        in generate_migration_file.py.
        :return:
        """
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name,
                               migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        # Now test the migration.txt file
        self.verify_txt_file_based_on_demo(migration_file_name)

    def test_migration_generator_verify_txt_file_no_compiled_demog(self):
        """
        Similar to test_migration_generator_verify_txt_file(). this test passes IdRef instead of demograohics file into
        MigrationGenerator.generate_migration()
        :return:
        """
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, idRef="Custom user")

        # Now test the migration.txt file
        self.verify_txt_file_based_on_demo(migration_file_name)

    def test_migration_generator_verify_txt_file_with_exclude_nodes(self):
        """
        Similar to test_migration_generator_verify_txt_file(). this test has exclude_nodes.
        :return:
        """
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        exclude_nodes = [1, 3, 25]

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params,
                                                            exclude_nodes=exclude_nodes)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        # Now test the migration.txt file
        self.verify_txt_file_based_on_demo(migration_file_name, exclude_nodes)

    def test_migration_generator_verify_bin_file(self):
        """
        Verify that the binary file generated from MigrationGenerator converts to the correct migration nodes and rates
        configuration.
        :return:
        """
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        # Now test the binary file
        self.convert_bin_file_and_verify(migration_file_name)

    def test_migration_generator_verify_json_file(self):
        """
        Verify that the json header file has the correct IdRef, NodeCount, DatavalueCount and NodeOffsets
        :return:
        """
        migration_file_name = os.path.join(self.test_output, 'migration.bin')

        # Build our StaticGravityModelRatesGenerator
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        # Now use the migration generator to produce the binary file
        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        with open(demographics_file, 'r') as demo_file:
            demo_json = json.load(demo_file)
        id_ref = demo_json["Metadata"]["IdReference"]

        # Generate migration json from migration text file.
        generate_migration_files_from_txt(migration_file_name.replace('bin', 'txt'),
                                          migration_file_name.replace("migration.bin", "converted_migration.bin"),
                                          MigrationTypes.local, id_ref)

        with open(migration_file_name.replace('migration.bin', 'converted_migration.bin.json'), 'r') as converted_json_file:
            converted_json_file = json.load(converted_json_file)
        with open(migration_file_name + ".json", 'r') as test_json_file:
            json_file = json.load(test_json_file)

        # Check the IdReference
        self.assertEqual(converted_json_file["Metadata"]["IdReference"], json_file["Metadata"]["IdReference"])

        # Check NodeCount
        self.assertEqual(converted_json_file["Metadata"]["NodeCount"], json_file["Metadata"]["NodeCount"])

        # Check DatavalueCount
        self.assertEqual(converted_json_file["Metadata"]["DatavalueCount"], json_file["Metadata"]["DatavalueCount"])

        # Check NodeOffsets
        self.assertEqual(converted_json_file["NodeOffsets"], json_file["NodeOffsets"])

    def test_migration_file_path_generation_for_migration_generator(self):
        migration_file_name = os.path.join(self.output_path, self._testMethodName + str(random.random()), 'migration.bin')
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        self.verify_migration_files_generated(migration_file_name)

    def test_empty_migration_file_path_for_migration_generator(self):
        migration_file_name = ""
        link_rates_model = StaticGravityModelRatesGenerator(demographics_file, static_gravity_params)

        m = MigrationGenerator(migration_file_name, migration_type=MigrationTypes.local,
                               link_rates_model=link_rates_model)
        m.generate_migration(True, demographics_file_path=demographics_file)

        default_migration_file_name = "migration.bin"

        self.verify_migration_files_generated(default_migration_file_name)
    # endregion

