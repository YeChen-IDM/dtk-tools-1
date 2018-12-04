import unittest
from dtk.interventions.novel_vector_control import add_ATSB
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.utils.Campaign.CampaignClass import *


class ParamNames():
    MALARIA_SIM = "MALARIA_SIM"
    Vector_Species_Names = "Vector_Species_Names"
    Vector_Species_Params = "Vector_Species_Params"
    Vector_Sugar_Feeding_Frequency = "Vector_Sugar_Feeding_Frequency"
    VECTOR_SUGAR_FEEDING_EVERY_DAY = "VECTOR_SUGAR_FEEDING_EVERY_DAY"
    VECTOR_SUGAR_FEEDING_NONE = "VECTOR_SUGAR_FEEDING_NONE"
    Species = "Species"
    Killing_Config = "Killing_Config"
    WaningEffectConstant = "WaningEffectConstant"
    WaningEffectMapLinear = "WaningEffectMapLinear"
    Initial_Effect = "Initial_Effect"
    Event_Coordinator_Config = "Event_Coordinator_Config"
    Intervention_Config = "Intervention_Config"
    Killing_Config_Per_Species = "Killing_Config_Per_Species"


class ATSBTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ATSBTest, self).__init__(*args, **kwargs)

    def setUp(self):
        print(f"running {self._testMethodName}:")
        pass

    def tearDown(self):
        print("end of test\n")
        pass

    def test_atsb_value_error_1(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        species_names = cb.params[ParamNames.Vector_Species_Names]
        killing = 0.0337

        with self.assertRaises(ValueError) as context:
            add_ATSB(cb,
                     kill_cfg=[{ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                            ParamNames.Initial_Effect: killing}
                                } for sp in species_names])
        print(f"\t{context.exception}")
        self.assertTrue('Each config in SugarTrap killing config list must contain species name' in str(context.exception))

    def test_atsb_value_error_2(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        species_names = cb.params[ParamNames.Vector_Species_Names]
        killing = 0.0337
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][
                ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_NONE
        with self.assertRaises(ValueError) as context:
            add_ATSB(cb,
                     kill_cfg=[{ParamNames.Species: sp,
                                ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                            ParamNames.Initial_Effect: killing}
                                } for sp in species_names])
        print(f"\t{context.exception}")
        self.assertTrue('A targeted SugarTrap species is not a sugar-feeding species in config' in str(context.exception))

    def test_atsb_value_error_3(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        species_names = cb.params[ParamNames.Vector_Species_Names]

        with self.assertRaises(ValueError) as context:
            add_ATSB(cb,
                     kill_cfg=[{ParamNames.Species: sp
                                } for sp in species_names])
        print(f"\t{context.exception}")
        self.assertTrue('Each config in SugarTrap killing config list must contain Killing_Config' in str(context.exception))

    def test_atsb_value_error_4(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        species_names = cb.params[ParamNames.Vector_Species_Names]
        sp = species_names[0]
        with self.assertRaises(ValueError) as context:
            add_ATSB(cb,
                     kill_cfg={ParamNames.Species: sp
                                } )
        print(f"\t{context.exception}")
        self.assertTrue('Each config in SugarTrap killing config list must contain Killing_Config' in str(context.exception))

    def test_atsb_target_species_list(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        species_names_kill_cfg = species_names[:]
        species_names_kill_cfg.pop()
        killing = 0.0337
        add_ATSB(cb,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names_kill_cfg])
        self.assertTrue(len(cb.campaign.Events) == 1)
        killing_configs = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        self.assertTrue(len(killing_configs)==len(species_names_kill_cfg))
        for killing_config in killing_configs:
            species = killing_config[ParamNames.Species]
            print(f"\ttarget species is: {species}")
            if species in species_names_kill_cfg:
                species_names_kill_cfg.remove(species)
            else:
                raise AssertionError("A targeted SugarTrap species is not in kill_cfg")
        self.assertTrue(len(species_names_kill_cfg) == 0)

    def test_atsb_config_species_not_in_killing_list(self, debug=False):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM,
                                            Demographics_Filenames=["Garki_single_demographics.json"],
                                            Climate_Model="CLIMATE_CONSTANT",
                                            Simulation_Duration=180)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        species_names_kill_cfg = species_names[:]
        species_names_kill_cfg.pop()
        killing = 0.0337
        add_ATSB(cb, start=10,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names_kill_cfg])

        # run simulation in Comps when debug is True
        if debug:
            from simtools.SetupParser import SetupParser
            from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
            from simtools.AssetManager.FileList import FileList
            from simtools.AssetManager.AssetCollection import AssetCollection

            default_block = 'HPC'
            SetupParser.init(selected_block=default_block)

            fl = FileList()
            exe_path = "Eradication.exe"
            fl.add_file(exe_path)
            fl.add_file("Garki_single_demographics.json")
            collection = AssetCollection(local_files=fl)
            collection.prepare(location='HPC')
            cb.set_collection_id(collection.collection_id)

            exp_manager = ExperimentManagerFactory.init()
            run_sim_args = {
                'exp_name': 'ATSB_' + str(self._testMethodName),
                'config_builder': cb
            }
            exp_manager.run_simulations(**run_sim_args)

            exp_manager.wait_for_finished(verbose=True)
            self.assertTrue(exp_manager.succeeded(), "SimulationState is not Succeeded.")

    def test_atsb_target_species_dict(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][
                ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        sp = species_names[0]
        killing = 0.0337
        add_ATSB(cb,
                 kill_cfg={ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            })
        self.assertTrue(len(cb.campaign.Events) == 1)
        killing_configs = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        self.assertTrue(len(killing_configs) == 1)
        species = killing_configs[0][ParamNames.Species]
        print(f"\ttarget species is: {species}")
        self.assertEqual(species, sp)

    def test_atsb_target_species_default(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        species_names_kill_cfg = species_names[:]
        killing = 0.0337
        add_ATSB(cb,
                 kill_cfg={
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            })
        self.assertTrue(len(cb.campaign.Events) == 1)
        killing_configs = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        self.assertTrue(len(killing_configs)==len(species_names))
        for killing_config in killing_configs:
            species = killing_config[ParamNames.Species]
            print(f"\ttarget species is: {species}")
            if species in species_names_kill_cfg:
                species_names_kill_cfg.remove(species)
            else:
                raise AssertionError("A targeted SugarTrap species is not in kill_cfg")
        self.assertTrue(len(species_names_kill_cfg) == 0)

    def test_atsb_initial_effect(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killings = [0.0337] * len(species_names)
        for i in range(len(killings)):
            killings[i] += 0.02 * i
        coverage = 0.66
        add_ATSB(cb,
                 coverage=coverage,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp, killing in zip(species_names, killings)])
        killing_configs = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        for killing_config in killing_configs:
            initial_effect = killing_config[ParamNames.Killing_Config][ParamNames.Initial_Effect]
            species = killing_config[ParamNames.Species]
            killing = killings[species_names.index(species)]
            print(f"\tinitial effect is: {initial_effect}")
            self.assertAlmostEqual(initial_effect, killing * coverage, 3)

    def test_atsb_start_day(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killing = 0.0337
        start = 5
        add_ATSB(cb, start=start,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names])
        start_day = cb.campaign.Events[0].Start_Day
        self.assertEqual(start_day, start)

    def test_atsb_expiration(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killing = 0.0337
        duration = 3 * 365
        duration_std_dev = 30
        add_ATSB(cb,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names],
                 duration=duration,
                 duration_std_dev=duration_std_dev)
        expiration_period_mean = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Expiration_Period_Mean
        expiration_period_std_dev = cb.campaign.Events[0].Event_Coordinator_Config.Intervention_Config.Expiration_Period_Std_Dev
        print(f"\tExpiration_Period_Mean is: {expiration_period_mean}")
        print(f"\tExpiration_Period_Std_Dev is: {expiration_period_std_dev}")
        self.assertEqual(expiration_period_mean, duration)
        self.assertEqual(expiration_period_std_dev, duration_std_dev)

    def test_atsb_nodeIDs(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killing = 0.0337
        nodeIDs = [111, 222, 333]
        add_ATSB(cb,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names],
                 nodeIDs=nodeIDs)
        node_list = cb.campaign.Events[0].Nodeset_Config.Node_List
        print(f"\tnode_list is: {node_list}")
        self.assertEqual(node_list, nodeIDs)

    def test_atsb_NodeSetAll(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killing = 0.0337
        nodeIDs = []
        add_ATSB(cb,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names],
                 nodeIDs=nodeIDs)
        nodeset_class = cb.campaign.Events[0].Nodeset_Config._definition['class']
        print(f"\tnodeset_class is: {nodeset_class}")
        self.assertEqual(nodeset_class, "NodeSetAll")

    def test_atsb_property_restrictions(self):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][
                ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        killing = 0.0337
        node_property_restrictions = ["Risk:High"]
        add_ATSB(cb,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names],
                 node_property_restrictions=node_property_restrictions)
        restrictions = cb.campaign.Events[0].Node_Property_Restrictions
        print(f"\tNode_Property_Restrictions is: {restrictions}")
        self.assertEqual(restrictions, node_property_restrictions)

    def test_atsb_multiple_traps(self, debug=False):
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM,
                                            Demographics_Filenames=["Garki_single_demographics.json"],
                                            Climate_Model="CLIMATE_CONSTANT",
                                            Simulation_Duration=180)
        cb.campaign.Events = []
        species_names = cb.params[ParamNames.Vector_Species_Names]
        species_params = cb.params[ParamNames.Vector_Species_Params]
        for species_name in species_names:
            species_params[species_name][
                ParamNames.Vector_Sugar_Feeding_Frequency] = ParamNames.VECTOR_SUGAR_FEEDING_EVERY_DAY
        species_names_kill_cfg = species_names[:]
        killing = 0.0337
        add_ATSB(cb, start=180,
                 coverage=0.9)
        add_ATSB(cb,start=1,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names_kill_cfg[:2]],
                 coverage=1, node_property_restrictions=["Risk:High"])
        add_ATSB(cb,start=180,
                 kill_cfg=[{ParamNames.Species: sp,
                            ParamNames.Killing_Config: {"class": ParamNames.WaningEffectConstant,
                                                        ParamNames.Initial_Effect: killing}
                            } for sp in species_names_kill_cfg[1:]],
                 coverage=0.9, nodeIDs=[340461476])
        add_ATSB(cb, start=180,
                 coverage=0.9)
        self.assertTrue(len(cb.campaign.Events) == 4)
        killing_configs_0 = cb.campaign.Events[
            0].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        killing_configs_1 = cb.campaign.Events[
            1].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        killing_configs_2 = cb.campaign.Events[
            2].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species
        killing_configs_3 = cb.campaign.Events[
            3].Event_Coordinator_Config.Intervention_Config.Killing_Config_Per_Species

        # testing species names in killing configs
        self.assertTrue([x[ParamNames.Species] for x in killing_configs_0] == species_names_kill_cfg)
        self.assertTrue([x[ParamNames.Species] for x in killing_configs_1] == species_names_kill_cfg[:2])
        self.assertTrue([x[ParamNames.Species] for x in killing_configs_2] == species_names_kill_cfg[1:])
        self.assertTrue([x[ParamNames.Species] for x in killing_configs_3] == species_names_kill_cfg)

        # testing waning effect is configured correctly.
        self.waning_effect_test(killing_configs_0, WaningEffectBoxExponential)

        self.waning_effect_test(killing_configs_1, dict)
        for x in killing_configs_1:
            self.assertEqual(x[ParamNames.Killing_Config]["class"], ParamNames.WaningEffectConstant)

        self.waning_effect_test(killing_configs_2, dict)
        for x in killing_configs_2:
            self.assertEqual(x[ParamNames.Killing_Config]["class"], ParamNames.WaningEffectConstant)

        self.waning_effect_test(killing_configs_3, WaningEffectBoxExponential)

        # run simulation in Comps when debug is True
        if debug:
            from simtools.SetupParser import SetupParser
            from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
            from simtools.AssetManager.FileList import FileList
            from simtools.AssetManager.AssetCollection import AssetCollection

            default_block = 'HPC'
            SetupParser.init(selected_block=default_block)

            fl = FileList()
            exe_path = "Eradication.exe"
            fl.add_file(exe_path)
            fl.add_file("Garki_single_demographics.json")
            collection = AssetCollection(local_files=fl)
            collection.prepare(location='HPC')
            cb.set_collection_id(collection.collection_id)

            exp_manager = ExperimentManagerFactory.init()
            run_sim_args = {
                'exp_name': 'ATSB_' + str(self._testMethodName),
                'config_builder': cb
            }
            exp_manager.run_simulations(**run_sim_args)

            exp_manager.wait_for_finished(verbose=True)
            self.assertTrue(exp_manager.succeeded(), "SimulationState is not Succeeded.")

    def waning_effect_test(self, killing_configs, instance_name):
        for x in killing_configs:
            self.assertIsInstance(x[ParamNames.Killing_Config], instance_name)
