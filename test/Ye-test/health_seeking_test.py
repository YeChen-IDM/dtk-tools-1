import unittest
# import health_seeking.py from dtk and malaria to compare their functionality
# in each unit test, we make sure health_seeking.py from malaria has the same functionality as the one from dtk
from dtk.interventions.health_seeking import add_health_seeking as add_health_seeking_dtk
from malaria.interventions.health_seeking import add_health_seeking as add_health_seeking_malaria
from simtools.SetupParser import SetupParser
from simtools.ExperimentManager.ExperimentManagerFactory import ExperimentManagerFactory
from simtools.AssetManager.FileList import FileList
from simtools.AssetManager.AssetCollection import AssetCollection
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.utils.Campaign.CampaignClass import *
import json
import os
import copy
import shutil
from enum import Enum

empty_campaign = copy.deepcopy(Campaign(
    Campaign_Name="Empty Campaign",
    Use_Defaults=True,
    Events=[]
))


class result_type(Enum):
    EQUAL = 0
    NOT_EQUAL = 1
    NOT_PRESENT = 2


class ParamNames():
    MALARIA_SIM = "MALARIA_SIM"
    Demographics_Filenames = "Demographics_Filenames"
    Climate_Model = "Climate_Model"
    CLIMATE_CONSTANT = "CLIMATE_CONSTANT"
    Simulation_Duration = "Simulation_Duration"
    Report_Event_Recorder = "Report_Event_Recorder"
    Report_Event_Recorder_Events = "Report_Event_Recorder_Events"
    Report_Event_Recorder_Ignore_Events_In_List = "Report_Event_Recorder_Ignore_Events_In_List"
    Event_Coordinator_Config = "Event_Coordinator_Config"
    Intervention_Config = "Intervention_Config"
    Actual_IndividualIntervention_Config = "Actual_IndividualIntervention_Config"
    Actual_IndividualIntervention_Configs = "Actual_IndividualIntervention_Configs"
    Demographic_Coverage = "Demographic_Coverage"
    Target_Age_Min = "Target_Age_Min"
    Target_Age_Max = "Target_Age_Max"
    PropertyValueChanger = "PropertyValueChanger"
    Target_Property_Key = "Target_Property_Key"
    Target_Property_Value = "Target_Property_Value"
    Revert = "Revert"
    Property_Restrictions_Within_Node = "Property_Restrictions_Within_Node"
    Node_Property_Restrictions = "Node_Property_Restrictions"


class HealthSeekingTest(unittest.TestCase):
    runInComps = True
    debug = True
    # get full diff output
    maxDiff = None

    def __init__(self, *args, **kwargs):
        super(HealthSeekingTest, self).__init__(*args, **kwargs)

    # region unittest setup and teardown method
    @classmethod
    def setUpClass(cls):
        if cls.runInComps:
            default_block = 'HPC'
            SetupParser.init(selected_block=default_block)

    def setUp(self):
        print(f"running {self._testMethodName}:")
        pass

    def tearDown(self):
        print("end of test\n")
        pass
    # endregion

    # region class helper methods
    def is_subdict(self, small: dict, big: dict):
        """
        compare two dictionaries with nested structure, return if small is a sub dictionary of big
        Args:
            small:
            big:

        Returns:

        """
        if isinstance(small, dict) and isinstance(big, dict):
            for key in small:
                if key in big:
                    if (isinstance(small[key], dict) and isinstance(big[key], dict)) or \
                    (isinstance(small[key], list) and isinstance(big[key], list)):
                        if not self.is_subdict(small[key], big[key]):
                            return False
                    else:
                        if not small[key] == big[key]:
                            return False
                else:
                    return False
        elif isinstance(small, list) and isinstance(big, list):
            if len(small) != len(big):
                return False
            for i in range(len(small)):
                if not self.is_subdict(small[i], big[i]):
                    return False

        return True
        # this will not work for nested dictionaries
        # return dict(big, **small) == big

    def to_test_is_subdict(self):
        a = {"test":[{"a":1},{"b":2}]}
        aa = {"test":[{"a":1},{"b":2}], "test2": 3}
        self.assertTrue(self.is_subdict(a, aa))
        a = {"test": [[{"a": 1}], [{"b": 2}]]}
        aa = {"test": [[{"a": 1}], [{"b": 2}]], "test2": 3}
        self.assertTrue(self.is_subdict(a, aa))
        a = {"test": [[{"a": 1}], [{"b": 4}]]}
        aa = {"test": [[{"a": 1}], [{"b": 2}]], "test2": 3}
        self.assertFalse(self.is_subdict(a, aa))

    def is_valueequal(self, test_dict: dict, test_key: str, test_value):
        """
        return True if test_key is a key in test_dict and its value is equal to test_value
        Args:
            test_dict:
            test_key:
            test_value:

        Returns:

        """
        count = []
        if self.is_valueequal_internal(test_dict, test_key, test_value, count) == result_type.EQUAL:
            return True
        else:
            return False

    def is_valueequal_internal(self, test_dict: dict, test_key: str, test_value, count:list=[]):
        """

        Args:
            test_dict:
            test_key:
            test_value:
            count:

        Returns:

        """
        if test_key in test_dict:
            if test_value == test_dict[test_key]:
                count.append("Y")
                return result_type.EQUAL
            else:
                return result_type.NOT_EQUAL
        else:
            for key, value in test_dict.items():
                if isinstance(value, dict):
                    if self.is_valueequal_internal(value, test_key, test_value, count) == result_type.NOT_EQUAL:
                        return result_type.NOT_EQUAL

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            if self.is_valueequal_internal(item, test_key, test_value, count) == result_type.NOT_EQUAL:
                                return result_type.NOT_EQUAL

        if not count:
            return result_type.NOT_PRESENT
        else:
            return result_type.EQUAL

    def to_test_is_valueequal(self):
        key = "test_key"
        value="test_value"
        dict = {"a lit": [{key: value}]}
        self.assertTrue(self.is_valueequal(dict, key, value))
        self.assertTrue(self.is_valueequal({key:value}, key, value))
        dict_2 = {"a lit": [{key: value}], "antoher list":[{'a':1}, {key: value + "1"}]}
        self.assertFalse(self.is_valueequal(dict_2, key, value))

    def run_in_comps(self, cb, broadcast_event_name='Received_Treatment',
                     demographics_filenames=["Garki_single_demographics.json"], enable_property_report=False):
        """
        Run simulation in Comps
        Args:
            cb: config builder
            broadcast_event_name:
            base_demo_file:

        Returns:

        """

        cb.set_param("Disable_IP_Whitelist", 1)

        if enable_property_report:
            cb.enable("Property_Output")
        else:
            cb.disable("Property_Output")

        cb.params["Demographics_Filenames"] = demographics_filenames
        cb.set_param(ParamNames.Climate_Model, ParamNames.CLIMATE_CONSTANT)
        cb.set_param(ParamNames.Simulation_Duration, 180)
        cb.set_param(ParamNames.Report_Event_Recorder, 1)
        cb.set_param(ParamNames.Report_Event_Recorder_Events, [broadcast_event_name])
        cb.set_param(ParamNames.Report_Event_Recorder_Ignore_Events_In_List, 0)

        fl = FileList()
        fl.add_path("asset")
        collection = AssetCollection(local_files=fl)
        collection.prepare(location='HPC')
        cb.set_collection_id(collection.collection_id)

        exp_manager = ExperimentManagerFactory.init()
        run_sim_args = {
            'exp_name': 'Health_Seeking_' + str(self._testMethodName),
            'config_builder': cb
        }
        exp_manager.run_simulations(**run_sim_args)

        exp_manager.wait_for_finished(verbose=True)
        self.assertTrue(exp_manager.succeeded(), "SimulationState is not Succeeded.")

        print("SimulationState is Succeeded, please see sim output in Comps.\n")

    @staticmethod
    def save_varaibles_to_json_files(variable_dict={}, path_to_save=""):
        if os.path.isdir(path_to_save):
            shutil.rmtree(path_to_save)
        os.mkdir(path_to_save)
        for variable_name in variable_dict:
            with open(os.path.join(path_to_save, f"{variable_name}.json"), "w") as outfile:
                json.dump(variable_dict[variable_name], outfile, indent=4)

    @staticmethod
    def get_variable_name(locals, variable):
        variable_name = None
        for k, v in list(locals.items()):
            if v == variable:
                variable_name = k
        return variable_name

    def generate_variable_dict(self, locals, variables):
        variable_dict = {}
        for variable in variables:
            variable_name = self.get_variable_name(locals, variable)
            variable_dict[variable_name] = variable
        return variable_dict

    def save_json_files(self, locals, variables=[], path_to_save=""):
        """
        Save a list of variables as json files, named by the original variable name.
        Args:
            locals:
            variables:
            path_to_save:

        Returns:

        """
        variable_dict = self.generate_variable_dict(locals, variables)
        if not os.path.isdir("health_seaking"):
            os.mkdir("health_seaking")
        path_to_save = os.path.join("health_seaking", path_to_save)
        self.save_varaibles_to_json_files(variable_dict, path_to_save)
    # endregion

    # region unittests
    # each test does the following:
    #   Creates two campaigns that add a BroadcastEventName using both "add health seeking" methods.
    #   Asserts if campaign from dtk method is not a sub-dictionary of campaign from malaria method(when using default option)

    def test_broadcast_event_name(self):
        """
        see comment in region unittests
        Asserts if this event is not in the config Listed_Events or not in campaign Broadcast_Event.

        Returns:

        """
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)

        broadcast_event_name = 'test_event'#'Received_Treatment'
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, broadcast_event_name=broadcast_event_name)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, broadcast_event_name=broadcast_event_name)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                 campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)

        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))

        self.assertTrue(broadcast_event_name in cb.config["parameters"]['Listed_Events'])

        self.assertTrue(self.is_valueequal(campaign_malaria, "Broadcast_Event", broadcast_event_name))

        if self.runInComps:
            self.run_in_comps(cb, broadcast_event_name)

    def test_ind_property_with_disqualifying_properties(self):
        """
        see comment in region unittests
        Asserts if any of ind_property_restrictions and disqualifying_properties is not in campaign.

        Returns:

        """
        node_property_restrictions = None
        ind_property_restrictions = [{"DrugStatus": "RecentDrug"}]
        disqualifying_properties = ["Risk:Low"]
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, ind_property_restrictions=ind_property_restrictions,
                               disqualifying_properties=disqualifying_properties)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, ind_property_restrictions=ind_property_restrictions,
                                   disqualifying_properties=disqualifying_properties)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                 campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)
        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))

        self.assertTrue(self.is_valueequal(campaign_malaria, "Disqualifying_Properties", disqualifying_properties))

        self.assertTrue(self.is_valueequal(campaign_malaria, ParamNames.Property_Restrictions_Within_Node,
                                           ind_property_restrictions))

        if self.runInComps:
            self.run_in_comps(cb, demographics_filenames=["Garki_single_demographics.json", "ip_overlay.json"])

    def test_ind_property(self):
        """
        see comment in region unittests
        Asserts if ind_property_restrictions is not in campaign.

        Returns:

        """
        ind_property_restrictions = [{"Risk": "High"}]
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, ind_property_restrictions=ind_property_restrictions)

        if self.runInComps:
            self.run_in_comps(cb, demographics_filenames=["Garki_single_demographics.json", "ip_overlay.json"])

    def test_node_property(self):
        """
        see comment in region unittests
        Asserts if node_property_restrictions is not in campaign.

        Returns:

        """
        node_property_restrictions = [{"Place": "City"}]
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, node_property_restrictions=node_property_restrictions)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, node_property_restrictions=node_property_restrictions)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                 campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)
        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))

        self.assertTrue(self.is_valueequal(campaign_malaria, ParamNames.Node_Property_Restrictions, node_property_restrictions))

        if self.runInComps:
            self.run_in_comps(cb, demographics_filenames=["Garki_4_nodes_demographics.json", "np_overlay.json"])

    def test_drug_ineligibility_duration(self):
        """
        see comment in region unittests
        Asserts if hardcoded Property_Restrictions_Within_Node is not in campaign.
        Asserts if PropertyValueChanger in not in campaign or not has the correct configuration.

        Returns:

        """
        drug_ineligibility_duration = 14
        target_property_key = "DrugStatus"
        target_property_value = "RecentDrug"
        property_restriction_value = "None"
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, drug_ineligibility_duration=drug_ineligibility_duration)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, drug_ineligibility_duration=drug_ineligibility_duration)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                 campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)
        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))
        default_target_length = 2
        for i in range(default_target_length):
            ic = campaign_malaria["Events"][i][ParamNames.Event_Coordinator_Config][ParamNames.Intervention_Config]
            self.assertTrue(ic[ParamNames.Property_Restrictions_Within_Node] == [{target_property_key: property_restriction_value}])
            aiics = ic[ParamNames.Actual_IndividualIntervention_Config][
                ParamNames.Actual_IndividualIntervention_Configs][-1]
            self.assertTrue(aiics["class"] == ParamNames.PropertyValueChanger)
            self.assertTrue(aiics[ParamNames.Revert] == drug_ineligibility_duration)
            self.assertTrue(aiics[ParamNames.Target_Property_Key] == target_property_key)
            self.assertTrue(aiics[ParamNames.Target_Property_Value] == target_property_value)

        if self.runInComps:
            self.run_in_comps(cb, demographics_filenames=["Garki_single_demographics.json", "ip_overlay.json"],
                              enable_property_report=True)

    def test_drug_ineligibility_duration_with_ind_property_restrictions(self):
        """
        see comment in region unittests
        Asserts if Property_Restrictions_Within_Node doesn't have the expected configuration in campaign.
        Asserts if PropertyValueChanger in not in campaign or not has the correct configuration.

        Returns:

        """
        drug_ineligibility_duration = 14
        target_property_key = "DrugStatus"
        target_property_value = "RecentDrug"
        property_restriction_value = "None"
        ind_property_restrictions = [{"Risk": "High"}]
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, drug_ineligibility_duration=drug_ineligibility_duration,
                               ind_property_restrictions=ind_property_restrictions)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, drug_ineligibility_duration=drug_ineligibility_duration,
                                   ind_property_restrictions=ind_property_restrictions)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                 campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)
        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))
        default_target_length = 2
        for i in range(default_target_length):
            ic = campaign_malaria["Events"][i][ParamNames.Event_Coordinator_Config][ParamNames.Intervention_Config]
            self.assertTrue(ic[ParamNames.Property_Restrictions_Within_Node] == [
                    {**{target_property_key: property_restriction_value}, **x} for x in ind_property_restrictions])
            aiics = ic[ParamNames.Actual_IndividualIntervention_Config][
                ParamNames.Actual_IndividualIntervention_Configs][-1]
            self.assertTrue(aiics["class"] == ParamNames.PropertyValueChanger)
            self.assertTrue(aiics[ParamNames.Revert] == drug_ineligibility_duration)
            self.assertTrue(aiics[ParamNames.Target_Property_Key] == target_property_key)
            self.assertTrue(aiics[ParamNames.Target_Property_Value] == target_property_value)

        if self.runInComps:
            self.run_in_comps(cb, demographics_filenames=["Garki_single_demographics.json", "ip_overlay.json"],
                              enable_property_report=True)

    def test_target_drugs_dosing(self):
        """
        see comment in region unittests
        Asserts if Intervention_Config doesn't have the expected configuration in campaign.
        Asserts if Drug config doesn't have the expected configuration in campaign.

        Returns:

        """
        targets = [{'trigger': 'NewSevereCase', 'coverage': 0.67, 'agemin': 10, 'agemax': 40,
                    'seek': 0.78, 'rate': 0.4},
                   {'trigger': 'NewClinicalCase', 'coverage': 0.45, 'seek': 0.9, 'rate': 0.8},
                   {'trigger': 'Births', 'coverage': 0.13, 'seek': 0.4, 'rate': 0.3}]
        drugs = ['Vehicle', 'DHA', 'Sulfadoxine']
        dosing = 'SingleDoseWhenSymptom'
        cb = DTKConfigBuilder.from_defaults(sim_type=ParamNames.MALARIA_SIM)
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_dtk(cb, targets=targets, drug=drugs, dosing=dosing)
        campaign_dtk = json.loads(cb.campaign.to_json(True))
        campaign_dtk_with_default = json.loads(cb.campaign.to_json(False))
        cb.campaign = copy.deepcopy(empty_campaign)
        add_health_seeking_malaria(cb, targets=targets, drug=drugs, dosing=dosing)
        campaign_malaria = json.loads(cb.campaign.to_json(True))
        campaign_malaria_with_default = json.loads(cb.campaign.to_json(False))
        if self.debug:
            self.save_json_files(locals(),
                                 [campaign_dtk, campaign_dtk_with_default,
                                  campaign_malaria, campaign_malaria_with_default],
                                 self._testMethodName)
        # compare both campaign object without comparing default values, see imports
        self.assertTrue(self.is_subdict(small=campaign_dtk, big=campaign_malaria))

        # compare both campaign object including default values
        self.assertTrue(self.is_subdict(small=campaign_malaria_with_default, big=campaign_dtk_with_default))

        for i in range(len(targets)):
            ic =campaign_malaria["Events"][i][ParamNames.Event_Coordinator_Config][ParamNames.Intervention_Config]
            self.assertTrue(ic[ParamNames.Demographic_Coverage] == targets[i]["coverage"] * targets[i]["seek"])
            if all([k in targets[i].keys() for k in ['agemin', 'agemax']]):
                self.assertTrue(ic[ParamNames.Target_Age_Max] == targets[i]["agemax"])
                self.assertTrue(ic[ParamNames.Target_Age_Min] == targets[i]["agemin"])
            aiic = ic[ParamNames.Actual_IndividualIntervention_Config]
            self.assertTrue(aiic["Delay_Period"] == 1.0 / targets[i]["rate"])
        for j in range(len(drugs)):
                aiics = aiic[ParamNames.Actual_IndividualIntervention_Configs][j]
                drug_config = {"Cost_To_Consumer": 1,
                               "Drug_Type": drugs[j],
                               "Dosing_Type": dosing,
                               "class": "AntimalarialDrug"}
                self.assertTrue(aiics==drug_config)

        if self.runInComps:
            self.run_in_comps(cb)
    # endregion

