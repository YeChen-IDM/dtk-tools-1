from dtk.interventions.triggered_campaign_delay_event import triggered_campaign_delay_event

from dtk.utils.Campaign.CampaignClass import *
from dtk.utils.Campaign.CampaignEnum import *


def change_node_property(cb, target_property_name, target_property_value, start_day=0, daily_prob=1,
                         max_duration=0, revert=0, nodeIDs=[], node_property_restrictions=[], triggered_campaign_delay=0,
                        trigger_condition_list=[], listening_duration=-1):

    node_cfg = NodeSetAll()
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    node_property_value_changer = NodePropertyValueChanger(
        Target_NP_Key_Value="%s:%s" % (target_property_name, target_property_value),
        Daily_Probability=daily_prob,
        Maximum_Duration=max_duration,
        Revert=revert
    )

    if trigger_condition_list:
        if triggered_campaign_delay:
            trigger_condition_list = [str(triggered_campaign_delay_event(cb, start_day, nodeIDs, triggered_campaign_delay,
                        trigger_condition_list, listening_duration))]

        changer_event = CampaignEvent(
            Start_Day=int(start_day),
            Nodeset_Config=node_cfg,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Blackout_Event_Trigger="Node_Property_Change_Blackout",     # [TODO]: enum??
                    # we don't care about this, just need something to be here so the blackout works at all
                    Blackout_Period=1,  # so we only distribute the node event(s) once
                    Blackout_On_First_Occurrence=1,
                    Duration=listening_duration,
                    Trigger_Condition_List=trigger_condition_list,
                    Actual_IndividualIntervention_Config=node_property_value_changer
                )
            )
        )

        if node_property_restrictions:
            changer_event.Event_Coordinator_Config.Intervention_Config.Node_Property_Restrictions = node_property_restrictions

        cb.add_event(changer_event)

    else:
        prop_ch_config = StandardInterventionDistributionEventCoordinator(Intervention_Config=node_property_value_changer)

        if node_property_restrictions:
            prop_ch_config.Intervention_Config.Node_Property_Restrictions = node_property_restrictions

        event = CampaignEvent(
            Start_Day=start_day,
            Event_Coordinator_Config=prop_ch_config,
            Nodeset_Config=node_cfg
        )

        cb.add_event(event)


def change_individual_property_at_age(cb, target_property_name, target_property_value, change_age_in_days, start_day=0,
                                      duration=-1, coverage=1, daily_prob=1, max_duration=0, revert=0, nodeIDs=[],
                                      node_property_restrictions=[]):

    actual_config = PropertyValueChanger(
        Target_Property_Key=target_property_name,
        Target_Property_Value=target_property_value,
        Daily_Probability=daily_prob,
        Maximum_Duration=max_duration,
        Revert=revert
    )

    birth_triggered_intervention = BirthTriggeredIV(
        Duration=duration,  # default to forever if  duration not specified
        Demographic_Coverage=coverage,
        Actual_IndividualIntervention_Config=DelayedIntervention(
            Coverage=1,
            Delay_Distribution=DelayedIntervention_Delay_Distribution_Enum.FIXED_DURATION,
            Delay_Period=change_age_in_days,
            Actual_IndividualIntervention_Configs=[actual_config]
        )
    )

    prop_ch_config = StandardInterventionDistributionEventCoordinator(
        Intervention_Config=birth_triggered_intervention
    )

    if node_property_restrictions:
        prop_ch_config.Intervention_Config.Node_Property_Restrictions = node_property_restrictions

    node_cfg = NodeSetAll()
    if nodeIDs :
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    event = CampaignEvent(
        Start_Day=start_day,
        Event_Coordinator_Config=prop_ch_config,
        Nodeset_Config=node_cfg
    )

    cb.add_event(event)


def change_individual_property(cb, target_property_name, target_property_value, target='Everyone', start_day=0,
                               coverage=1, daily_prob=1, max_duration=0, revert=0, nodeIDs=[],
                               node_property_restrictions=[], ind_property_restrictions=[], triggered_campaign_delay=0,
                               trigger_condition_list=[], listening_duration=-1
                               ):

    node_cfg = NodeSetAll()
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    property_value_changer = PropertyValueChanger(
        Target_Property_Key=target_property_name,
        Target_Property_Value=target_property_value,
        Daily_Probability=daily_prob,
        Maximum_Duration=max_duration,
        Revert=revert
    )

    if trigger_condition_list:
        if triggered_campaign_delay:
            trigger_condition_list = [triggered_campaign_delay_event(cb, start_day,
                                                                     nodeIDs=nodeIDs,
                                                                     triggered_campaign_delay=triggered_campaign_delay,
                                                                     trigger_condition_list=trigger_condition_list,
                                                                     listening_duration=listening_duration)]

        changer_event = CampaignEvent(
            Start_Day=start_day,
            Nodeset_Config=node_cfg,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Blackout_Event_Trigger="Ind_Property_Blackout",
                    # we don't care about this, just need something to be here so the blackout works at all
                    Blackout_Period=1,
                    # so we only distribute the node event(s) once
                    Blackout_On_First_Occurrence=True,
                    Target_Residents_Only=False,
                    Duration=listening_duration,
                    Trigger_Condition_List=trigger_condition_list,
                    # Target_Residents_Only=1,          # [ZDU]: duplicated
                    Demographic_Coverage=coverage,
                    Actual_IndividualIntervention_Config=property_value_changer
                )
            )
        )

        if isinstance(target, dict) and all([k in target for k in ['agemin', 'agemax']]):
             changer_event.Event_Coordinator_Config.Intervention_Config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.ExplicitAgeRanges
             changer_event.Event_Coordinator_Config.Intervention_Config.Target_Age_Min = target['agemin']
             changer_event.Event_Coordinator_Config.Intervention_Config.Target_Age_Max = target['agemax']
        else:
             changer_event.Event_Coordinator_Config.Intervention_Config.Target_Demographic = NodeLevelHealthTriggeredIV_Target_Demographic_Enum[target]  # default is Everyone

        if node_property_restrictions:
             changer_event.Event_Coordinator_Config.Intervention_Config.Node_Property_Restrictions = node_property_restrictions

        if ind_property_restrictions:
             changer_event.Event_Coordinator_Config.Intervention_Config.Property_Restrictions_Within_Node = ind_property_restrictions
        cb.add_event(changer_event)


    else:
        prop_ch_config = StandardInterventionDistributionEventCoordinator(
            Demographic_Coverage=coverage,
            Intervention_Config=property_value_changer
        )

        if isinstance(target, dict) and all([k in target for k in ['agemin','agemax']]) :
            prop_ch_config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.ExplicitAgeRanges
            prop_ch_config.Target_Age_Min = target['agemin']
            prop_ch_config.Target_Age_Max = target['agemax']
        else:
            prop_ch_config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum[target] # default is Everyone

        if node_property_restrictions:
            prop_ch_config.Intervention_Config.Node_Property_Restrictions = node_property_restrictions

        if ind_property_restrictions:
            prop_ch_config.Intervention_Config.Property_Restrictions_Within_Node = ind_property_restrictions

        event = CampaignEvent(
            Start_Day=start_day,
            Event_Coordinator_Config=prop_ch_config,
            Nodeset_Config=node_cfg
        )

        cb.add_event(event)