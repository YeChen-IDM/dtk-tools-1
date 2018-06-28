import json
import unittest

# from dtk.interventions_new import *

from dtk.interventions.biting_risk import change_biting_risk
from dtk.interventions_new.biting_risk import change_biting_risk as change_biting_risk_new

from dtk.interventions.habitat_scale import scale_larval_habitats
from dtk.interventions_new.habitat_scale import scale_larval_habitats as scale_larval_habitats_new

from dtk.interventions.health_seeking import add_health_seeking
from dtk.interventions_new.health_seeking import add_health_seeking as add_health_seeking_new

from dtk.interventions.heg import heg_release
from dtk.interventions_new.heg import heg_release as heg_release_new

from dtk.interventions.incidence_counter import add_incidence_counter
from dtk.interventions_new.incidence_counter import add_incidence_counter as add_incidence_counter_new

from dtk.interventions.input_EIR import add_InputEIR
from dtk.interventions_new.input_EIR import add_InputEIR as add_InputEIR_new

from dtk.interventions.intervention_states import setup_recent_drug_states
from dtk.interventions_new.intervention_states import setup_recent_drug_states as setup_recent_drug_states_new

from dtk.interventions.irs import add_IRS
from dtk.interventions_new.irs import add_IRS as add_IRS_New

from dtk.interventions.itn import add_ITN
from dtk.interventions_new.itn import add_ITN as add_ITN_New

from dtk.interventions.itn_age_season import add_ITN_age_season
from dtk.interventions_new.itn_age_season import add_ITN_age_season as add_ITN_age_season_new

from dtk.interventions.ivermectin import add_ivermectin
from dtk.interventions_new.ivermectin import add_ivermectin as add_ivermectin_new

from dtk.interventions.larvicides import add_larvicides
from dtk.interventions_new.larvicides import add_larvicides as add_larvicides_new

from dtk.interventions.migrate_to import add_migration_event
from dtk.interventions_new.migrate_to import add_migration_event as add_migration_event_new

from dtk.interventions.mosquito_release import add_mosquito_release
from dtk.interventions_new.mosquito_release import add_mosquito_release as add_mosquito_release_new

from dtk.interventions.novel_vector_control import add_ATSB_individual_vector
from dtk.interventions_new.novel_vector_control import add_ATSB_individual_vector as add_ATSB_individual_vector_new

from dtk.interventions.outbreakindividual import recurring_outbreak
from dtk.interventions_new.outbreakindividual import recurring_outbreak as recurring_outbreak_new

from dtk.interventions.outbreakindividualdengue import add_OutbreakIndividualDengue
from dtk.interventions_new.outbreakindividualdengue import add_OutbreakIndividualDengue as add_OutbreakIndividualDengue_new

from dtk.interventions.property_change import change_node_property
from dtk.interventions_new.property_change import change_node_property as change_node_property_new
from dtk.utils.Campaign.CampaignHelper import CampaignEncoder

from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.utils.core.DTKConfigBuilderNew import DTKConfigBuilderNew


class TestInterventions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skip('Skip: test_dict, succeed')
    def test_dict(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"b": 2, "a": 1}
        self.assertDictEqual(d1, d2)

    @unittest.skip('Skip: test_dict2, failed')
    def test_dict2(self):
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"b": 2, "a": 1}
        self.assertDictEqual(d1, d2)

    @unittest.skip('Skip: test_int32, succeed')
    def test_int32(self):
        o = {'Campaign_Name': 'Empty Campaign', 'Use_Defaults': 1, 'Events': [{'Event_Coordinator_Config': {'Intervention_Config': {'Intervention_List': [{'Bednet_Type': 'ITN', 'Blocking_Config': {'Decay_Time_Constant': 730, 'Initial_Effect': 0.9, 'class': 'WaningEffectExponential'}, 'Cost_To_Consumer': 5, 'Killing_Config': {'Decay_Time_Constant': 1460, 'Initial_Effect': 0.6, 'class': 'WaningEffectExponential'}, 'Usage_Config_List': [{'class': 'WaningEffectMapLinearAge', 'Initial_Effect': 1.0, 'Durability_Map': {'Times': [0, 125], 'Values': [1, 1]}}, {'class': 'WaningEffectMapLinearSeasonal', 'Initial_Effect': 1.0, 'Durability_Map': {'Times': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 365], 'Values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}], 'class': 'UsageDependentBednet', 'Received_Event': 'Bednet_Got_New_One', 'Using_Event': 'Bednet_Using', 'Discard_Event': 'Bednet_Discarded', 'Expiration_Distribution_Type': 'DUAL_TIMESCALE_DURATION', 'Expiration_Period_1': 14600, 'Expiration_Period_2': 14600, 'Expiration_Percentage_Period_1': 1}], 'class': 'MultiInterventionDistributor'}, 'class': 'StandardInterventionDistributionEventCoordinator', 'Target_Demographic': 'Everyone', 'Demographic_Coverage': 1, 'Property_Restrictions_Within_Node': [{'BitingRisk': 'High'}, {'IsCool': 'Yes'}], 'Node_Property_Restrictions': [], 'Duration': -1}, 'Nodeset_Config': {'class': 'NodeSetAll'}, 'Start_Day': 100, 'class': 'CampaignEvent'}]}
        print(o)
        print(type(o))
        print(json.dumps(o, indent=3))

    @unittest.skip('Skip: test_change_biting_risk')
    def test_change_biting_risk(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        change_biting_risk(cb=cb,
                           coverage=1,
                           target_group={"agemin": 5, "agemax": 60}
                           )

        print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        change_biting_risk_new(cb=cb2,
                               coverage=1,
                               target_group={"agemin": 5, "agemax": 60}
                               )

        print(cb2.campaign.to_json())

        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        # cb2_dict = cb2.__dict__
        # print(cb2_dict)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_habitat_scale, ScaleLarvalHabitatLHM not found in schema')
    def test_habitat_scale(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        scale_larval_habitats(cb=cb,
                              scales=7
                              )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        scale_larval_habitats_new(cb=cb2,
                                  scales=7
                                  )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_health_seeking')
    def test_health_seeking(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_health_seeking(config_builder=cb,
                           start_day=7
                           )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_health_seeking_new(config_builder=cb2,
                               start_day=7
                               )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_heg')
    def test_heg(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        heg_release(cb=cb,
                    released_number=7
                    )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        heg_release_new(cb=cb2,
                        released_number=7
                        )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_incidence_counter')
    def test_incidence_counter(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_incidence_counter(cb=cb)

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_incidence_counter_new(cb=cb2)

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_input_EIR')
    def test_input_EIR(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_InputEIR(cb=cb,
                     monthlyEIRs=[3, 10, 2, 4, 1, 1, 2, 2, 2, 6, 3, 9]
                     )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_InputEIR_new(cb=cb2,
                         monthlyEIRs=[3, 10, 2, 4, 1, 1, 2, 2, 2, 6, 3, 9]
                         )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_intervention_states, SetInterventionState not found in schema')
    def test_intervention_states(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        setup_recent_drug_states(cb=cb)

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        setup_recent_drug_states_new(cb=cb2)

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_ITN')
    def test_ITN(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_ITN(config_builder=cb,
                start=100,
                coverage_by_ages=[{"coverage": 1, "min": 1, "max": 10}, {"coverage": 1, "min": 11, "max": 50}])

        print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_ITN_New(config_builder=cb2,
                start=100,
                coverage_by_ages=[{"coverage": 1, "min": 1, "max": 10}, {"coverage": 1, "min": 11, "max": 50}])

        print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        # cb2_dict = cb2.__dict__
        # print(cb2_dict)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_irs')
    def test_irs(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_IRS(config_builder=cb,
                start=100,
                coverage_by_ages=[{"coverage": 1, "min": 1, "max": 10}, {"coverage": 1, "min": 11, "max": 50}])

        print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_IRS_New(config_builder=cb2,
                start=100,
                coverage_by_ages=[{"coverage": 1, "min": 1, "max": 10}, {"coverage": 1, "min": 11, "max": 50}])

        print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        # cb2_dict = cb2.__dict__
        # print(cb2_dict)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    # @unittest.skip('Skip: test_ITN_age_season')
    def test_ITN_age_season(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_ITN_age_season(config_builder=cb,
                           start=100,
                           ind_property_restrictions=[{"BitingRisk": "High"}, {"IsCool": "Yes"}]
                           )

        # print(type(cb.campaign))
        # print(cb.campaign)
        # print(json.dumps(cb.campaign, indent=3))    # TypeError: Object of type 'int32' is not JSON serializable

        # print(json.dumps(cb.campaign, indent=3, cls=CampaignEncoder))   # good if no __init__!
        # print(json.dumps(cb.campaign, indent=3, cls=CampaignEncoder))  # not working!

        # encoder = CampaignEncoder(use_default=1)
        # s = encoder.encode(cb.campaign)
        # # print(s)
        # # print(type(s))
        # obj = json.loads(s)
        # # print(type(obj))
        # print(json.dumps(obj, indent=3))


        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_ITN_age_season_new(config_builder=cb2,
                               start=100,
                               ind_property_restrictions=[{"BitingRisk": "High"}, {"IsCool": "Yes"}]
                               )

        # print(cb2.campaign.to_json())

        # encoder = CampaignEncoder(use_defaults=1)      # test
        # encoder = CampaignEncoder(use_defaults=0)      # test
        encoder = CampaignEncoder(use_defaults=cb2.campaign.Use_Defaults)
        s = encoder.encode(cb2.campaign)
        # print(s)
        # print(type(s))
        obj = json.loads(s)
        # print(type(obj))
        print(json.dumps(obj, indent=3))

        # cb2_json = json.loads(cb2.campaign.to_json())
        # # print(cb2_json)
        #
        # cb_events = cb.campaign["Events"]
        # cb2_events = cb2_json["Events"]
        #
        # self.assertEqual(len(cb_events), len(cb2_events))
        # # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_ivermectin')
    def test_ivermectin(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_ivermectin(config_builder=cb,
                       start_days=[5, 10],
                       drug_code='DAY',
                       coverage='Demographic_Coverage',
                       listening_duration=[2, 6]
                       )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_ivermectin_new(config_builder=cb2,
                           start_days=[5, 10],
                           drug_code='DAY',
                           coverage='Demographic_Coverage',
                           listening_duration=[2, 6]
                           )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_larvicides')
    def test_larvicides(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_larvicides(config_builder=cb,
                       start=10
                       )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_larvicides_new(config_builder=cb2,
                           start=10
                           )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_migrate_to')
    def test_migrate_to(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_migration_event(cb=cb,
                            nodeto='test',
                            start_day=10
                            )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_migration_event_new(cb=cb2,
                                nodeto='test',
                                start_day=10
                                )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: mosquito_release')
    def test_mosquito_release(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_mosquito_release(cb=cb,
                             species='test',
                             start_day=10
                             )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_mosquito_release_new(cb=cb2,
                                 species='test',
                                 start_day=10
                                 )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: novel_vector_control')
    def test_novel_vector_control(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_ATSB_individual_vector(config_builder=cb,
                                   start=10
                                   )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_ATSB_individual_vector_new(config_builder=cb2,
                                       start=10
                                       )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_outbreakindividual')
    def test_outbreakindividual(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        recurring_outbreak(cb=cb)

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        recurring_outbreak_new(cb=cb2)

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_outbreakindividualdengue')
    def test_outbreakindividualdengue(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        add_OutbreakIndividualDengue(config_builder=cb,
                                     start=10,
                                     strain_id_name='test',
                                     coverage_by_age={"min": 1, "max": 10})

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        add_OutbreakIndividualDengue_new(config_builder=cb2,
                                         start=10,
                                         strain_id_name='test',
                                         coverage_by_age={"min": 1, "max": 10})

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])

    @unittest.skip('Skip: test_property_change')
    def test_property_change(self):
        cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
        change_node_property(cb=cb,
                             target_property_name='test',
                             target_property_value=10
                             )

        # print(json.dumps(cb.campaign, indent=3))

        cb2 = DTKConfigBuilderNew.from_defaults("MALARIA_SIM")
        change_node_property_new(cb=cb2,
                                 target_property_name='test',
                                 target_property_value=10
                                 )

        # print(cb2.campaign.to_json())

        # cb2_dict = json.loads(cb2.campaign.to_json())
        cb2_json = json.loads(cb2.campaign.to_json())
        # print(cb2_json)

        cb_events = cb.campaign["Events"]
        cb2_events = cb2_json["Events"]

        self.assertEqual(len(cb_events), len(cb2_events))
        # self.assertDictEqual(cb_events[0], cb2_events[0])



if __name__ == '__main__':
    unittest.main()
