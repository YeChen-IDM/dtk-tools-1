from dtk.interventions.triggered_campaign_delay_event import triggered_campaign_delay_event

from dtk.utils.Campaign.CampaignClass import *
from dtk.utils.Campaign.CampaignEnum import *


def change_node_property(cb, target_property_name: str=None, target_property_value: str=None, start_day: int=0,
                         daily_prob: float=1, max_duration: int=3.40282e+38, revert: int=0, nodeIDs: list=None,
                         node_property_restrictions: list=None, triggered_campaign_delay: int=0,
                         trigger_condition_list: list=None, listening_duration: int=-1,
                         disqualifying_properties: list=None):
    """
    Creates an intervention that changes node property <target_property_name>'s value to <target_property_value>,
    on day <start_day> or on <trigger_condition_list> event, you can narrow down which nodes and values get affected
    using <node_property_restrictions>.
    Args:
        cb: the config builder getting the event, default: no default, must pass in
        target_property_name: String NodeProperty whose value we are changing to target_property_value,
            default: no default, must define, example: "Place"
        target_property_value: String value to which we are updating the target_property_name NodeProperty
            default: no default, must define, example: "Urban"
        start_day: Integer day on which the intervention will be distributed or initialized (if triggered or with delay)
            default: 0, example: 90 (about 3 months in)
        daily_prob: The probability each day that the NodeProperty value will move to the target_property_value,
            default: 1 (all on the same day), example: 0.1
        max_duration: The maximum amount of time nodes have to move to a new NodeProperty value. This timing works in
            conjunction with daily_prob (Daily_Probability), nodes not moved to the new value by the end of max_duration
            keep the same value, default: 3.40282e+38, ex: 10
        revert: The number of days before an node moves back to its original NodeProperty value,
            default: 0 (means the new value is kept forever, never reverted), example: 35 (revert after 35 days)
        nodeIDs: list of nodes to which the campaign will be distributed, example:[2384,12,932]
        node_property_restrictions: use this to restrict which NodeProperty Value you want to change FROM.
            Restricts intervention based on list of dictionaries of node properties in
            format: [{"Land":"Swamp", "Roads":"No"}, {"Land": "Forest"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are nodes with (Swamp Land AND No Roads) OR (Forest Land) nodes
        triggered_campaign_delay: number of days campaign is delayed after being triggered,
            default: 0 (no delay), ex: 3 (delay property change by 3 days)
        trigger_condition_list:  list of events that trigger property change, makes property change a triggered event
            that's initially created on the start_day, default: None (not a triggered event)
            example: ["NewClinicalCase", "NewInfection"]
        listening_duration: how many days the triggered campaign will be active for,
            default: -1 (intervention is active forever), example: 60 (only active for 60 days)
        disqualifying_properties: A list of NodeProperty key:value pairs that cause an intervention to be aborted
            (persistent interventions will stop being distributed to nodes with these values).
            default: None, no restrictions, example: ["Place:Swamp"]

    Returns:
        Nothing, a campaign event is added to the campaign.json file

    """
    if not target_property_name or not target_property_value:
        raise ValueError("Please define both:  target_property_name and target_property_value.\n")
    if node_property_restrictions is None:
        node_property_restrictions = []
    if disqualifying_properties is None:
        disqualifying_properties = []
    node_cfg = NodeSetAll()
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    node_property_value_changer = NodePropertyValueChanger(
        Target_NP_Key_Value="%s:%s" % (target_property_name, target_property_value),
        Daily_Probability=daily_prob,
        Disqualifying_Properties=disqualifying_properties,
        Maximum_Duration=max_duration,
        Revert=revert
    )

    if trigger_condition_list:
        if triggered_campaign_delay:
            trigger_condition_list = [str(triggered_campaign_delay_event(cb, start_day, nodeIDs,
                                                                         triggered_campaign_delay,
                                                                         trigger_condition_list,
                                                                         listening_duration))]

        changer_event = CampaignEvent(
            Start_Day=start_day,
            Nodeset_Config=node_cfg,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Blackout_Event_Trigger="Node_Property_Change_Blackout",     # [TODO]: enum??
                    # we don't care about this, just need something to be here so the blackout works at all
                    Blackout_Period=1,  # so we only distribute the node event(s) once on that day in case of
                                        # multiple triggers
                    Blackout_On_First_Occurrence=1,
                    Duration=listening_duration,
                    Trigger_Condition_List=trigger_condition_list,
                    Actual_IndividualIntervention_Config=node_property_value_changer,
                    Node_Property_Restrictions=node_property_restrictions
                )
            )
        )
        cb.add_event(changer_event)

    else:
        changer_event = CampaignEvent(
            Start_Day=start_day,
            Nodeset_Config=node_cfg,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=node_property_value_changer,
                Node_Property_Restrictions=node_property_restrictions
            )
        )
        cb.add_event(changer_event)


def change_individual_property_at_age(cb, target_property_name: str=None, target_property_value: str=None,
                                      change_age_in_days: int=None, start_day: int=0,
                                      listening_duration: int=-1, coverage: float=1, daily_prob: float=1,
                                      max_duration: int=3.40282e+38, revert: int=0, nodeIDs: list=None,
                                      node_property_restrictions: list=None, ind_property_restrictions: list=None,
                                      disqualifying_properties: list=None):
    """
        Creates event that changes individual's individual property at <change_age_in_days> days after birth.
    Args:
        cb: the config builder getting the event, default: no default, must pass in
        target_property_name: String NodeProperty whose value we are changing to target_property_value,
            default: no default, must define, example: "Place"
        target_property_value: String value to which we are updating the target_property_name NodeProperty
            default: no default, must define, example: "Urban"
        change_age_in_days: at what age, in days to enact property change, must define,
            example: 90 (about 3 months after birth)
        start_day: Integer day on which the intervention will be distributed or initialized (if triggered or with delay)
            default: 0, example: 90 (about 3 months in)
        coverage: sets the "Demographic_Coverage", default: 1 (everyone), example: 0.7 (about 70% of otherwise qualifing
            individuals will receive a property change)
        daily_prob: The probability each day that the NodeProperty value will move to the target_property_value,
            default: 1 (all on the same day), example: 0.1
        max_duration: The maximum amount of time nodes have to move to a new NodeProperty value. This timing works in
            conjunction with daily_prob (Daily_Probability), nodes not moved to the new value by the end of max_duration
            keep the same value, default: 3.40282e+38, ex: 10
        revert: The number of days before an node moves back to its original NodeProperty value,
            default: 0 (means the new value is kept forever, never reverted), example: 35 (revert after 35 days)
        nodeIDs: list of nodes to which the campaign will be distributed, example:[2384,12,932]
        node_property_restrictions: use this to restrict which NodeProperty Value you want to change FROM.
            Restricts intervention based on list of dictionaries of node properties in
            format: [{"Land":"Swamp", "Roads":"No"}, {"Land": "Forest"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are nodes with (Swamp Land AND No Roads) OR (Forest Land) nodes
        ind_property_restrictions: Restricts intervention based on list of dictionaries of individual properties in
            format: [{"BitingRisk":"High", "IsCool":"Yes}, {"IsRich": "Yes"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are individuals with (High Biting Risk AND Yes IsCool) OR (IsRich) individuals
        listening_duration: how many days the triggered campaign will be active for,
            default: -1 (intervention is active forever), example: 60 (only active for 60 days)
        disqualifying_properties: A list of NodeProperty key:value pairs that cause an intervention to be aborted
            (persistent interventions will stop being distributed to nodes with these values).
            default: None, no restrictions, example: ["Risk:High"]

    Returns:
        Nothing, adds an intervention event to the campaign.json

    """
    if not target_property_name or not target_property_value or not change_age_in_days:
        raise ValueError("Please define all:  target_property_name and target_property_value and change_age_in_days.\n")
    if node_property_restrictions is None:
        node_property_restrictions = []
    if ind_property_restrictions is None:
        ind_property_restrictions = []
    if disqualifying_properties is None:
        disqualifying_properties = []
    node_cfg = NodeSetAll()
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    property_value_changer = PropertyValueChanger(
        Target_Property_Key=target_property_name,
        Target_Property_Value=target_property_value,
        Disqualifying_Properties=disqualifying_properties,
        Daily_Probability=daily_prob,
        Maximum_Duration=max_duration,
        Revert=revert
    )

    campaign_event = CampaignEvent(
        Start_Day=start_day,
        Nodeset_Config=node_cfg,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Intervention_Config=BirthTriggeredIV(
                Duration=listening_duration,
                Demographic_Coverage=coverage,
                Node_Property_Restrictions=node_property_restrictions,
                Property_Restrictions_Within_Node=ind_property_restrictions,
                Actual_IndividualIntervention_Config=DelayedIntervention(
                    Delay_Distribution=DelayedIntervention_Delay_Distribution_Enum.FIXED_DURATION,
                    Delay_Period_Fixed=change_age_in_days, #old style
                    Delay_Period=change_age_in_days, #new style
                    Actual_IndividualIntervention_Configs=[property_value_changer]
                )
            )
        )
    )

    cb.add_event(campaign_event)


def change_individual_property(cb, target_property_name: str=None, target_property_value: str=None,
                               target_group: any='Everyone', start_day: int=0, coverage: float=1, daily_prob: float=1,
                               max_duration: int=3.40282e+38, revert: int=0, nodeIDs: list=None,
                               node_property_restrictions: list=None, ind_property_restrictions: list=None,
                               triggered_campaign_delay: int=0, trigger_condition_list: list=None,
                               listening_duration: int=-1, blackout_flag: bool=True,
                               disqualifying_properties: list=None, target_residents_only: bool=False):
    """
    Creates campaign that changes IndividualProperty <target_property_name> to a new value <target_property_value>,
    with optional restrictions by <node_property_restrictions> and individual property restrictions
    <ind_property_restrictions>, which is useful when you only want to change a subset of people.
    Args:
        cb: the config builder getting the event, default: no default, must pass in
        target_property_name: String NodeProperty whose value we are changing to target_property_value,
            default: no default, must define, example: "Place"
        target_property_value: String value to which we are updating the target_property_name NodeProperty
            default: no default, must define, example: "Urban"
        target_group:  dictionary of {'agemin' : x, 'agemax' : y, 'gender': "Female"} to target  to individuals between
            x and y years of age, default: 'Everyone', example: {'agemin' : 5, 'agemax' : 15, 'gender': "Female"}
        start_day: Integer day on which the intervention will be distributed or initialized (if triggered or with delay)
            default: 0, example: 90 (about 3 months in)
        coverage: sets the "Demographic_Coverage", default: 1 (everyone), example: 0.7 (about 70% of otherwise qualifing
            individuals will receive a property change)
        daily_prob: The probability each day that the NodeProperty value will move to the target_property_value,
            default: 1 (all on the same day), example: 0.1
        max_duration: The maximum amount of time nodes have to move to a new NodeProperty value. This timing works in
            conjunction with daily_prob (Daily_Probability), nodes not moved to the new value by the end of max_duration
            keep the same value, default: 3.40282e+38, ex: 10
        revert: The number of days before an node moves back to its original NodeProperty value,
            default: 0 (means the new value is kept forever, never reverted), example: 35 (revert after 35 days)
        nodeIDs: list of nodes to which the campaign will be distributed, example:[2384,12,932]
        node_property_restrictions: use this to restrict which NodeProperty Value you want to change FROM.
            Restricts intervention based on list of dictionaries of node properties in
            format: [{"Land":"Swamp", "Roads":"No"}, {"Land": "Forest"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are nodes with (Swamp Land AND No Roads) OR (Forest Land) nodes
        ind_property_restrictions: Restricts intervention based on list of dictionaries of individual properties in
            format: [{"BitingRisk":"High", "IsCool":"Yes}, {"IsRich": "Yes"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are individuals with (High Biting Risk AND Yes IsCool) OR (IsRich) individuals
        triggered_campaign_delay: number of days campaign is delayed after being triggered,
            default: 0 (no delay), ex: 3 (delay property change by 3 days)
        trigger_condition_list:  list of events that trigger property change, makes property change a triggered event
            that's initially created on the start_day, default: None (not a triggered event)
            example: ["NewClinicalCase", "NewInfection"]
        listening_duration: how many days the triggered campaign will be active for,
            default: -1 (intervention is active forever), example: 60 (only active for 60 days)
        blackout_flag: set to true if you don't want triggered intervention to be distributed to the same person more
            than once a day, default: true
        disqualifying_properties: A list of NodeProperty key:value pairs that cause an intervention to be aborted
            (persistent interventions will stop being distributed to nodes with these values).
            default: None, no restrictions, example: ["Risk:High"]
        target_residents_only: if only the people who started out the simulation in this node and are still in that
            node will be affected, default: False
    Returns:
        Nothings, adds an intervention event to the campaign.json
    """
    if not target_property_name or not target_property_value:
        raise ValueError("Please define both:  target_property_name and target_property_value.\n")
    if ind_property_restrictions is None:
        ind_property_restrictions = []
    if node_property_restrictions is None:
        node_property_restrictions = []
    if disqualifying_properties is None:
        disqualifying_properties = []
    node_cfg = NodeSetAll()
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)

    gender = "All"
    age_min = 0
    age_max = 3.40282e+38
    if target_group != "Everyone" and isinstance(target_group, dict):
        try:
            age_min = target_group["agemin"]
            age_max = target_group["agemax"]
            if 'gender' in target_group:
                gender = target_group["gender"]
                target_group = "ExplicitAgeRangesAndGender"
            else:
                target_group = "ExplicitAgeRanges"
        except KeyError:
            raise KeyError("Unknown target_group parameter. Please pass in 'Everyone' or a dictionary of "
                           "{'agemin' : x, 'agemax' : y, 'gender': 'Female'} to target  to individuals between x "
                           "and y years of age, and (optional) gender.\n")

    property_value_changer = PropertyValueChanger(
        Target_Property_Key=target_property_name,
        Target_Property_Value=target_property_value,
        Disqualifying_Properties=disqualifying_properties,
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
                        # Blackout event trigger and period only used when blackout_flag is true
                        Blackout_Event_Trigger="Ind_Property_Blackout",
                        Blackout_Period=1,
                        # so we only change value once per time step
                        Blackout_On_First_Occurrence=blackout_flag,
                        Target_Residents_Only=target_residents_only,
                        Duration=listening_duration,
                        Trigger_Condition_List=trigger_condition_list,
                        Demographic_Coverage=coverage,
                        Target_Demographic=target_group,
                        Target_Age_Min=age_min,
                        Target_Age_Max=age_max,
                        Target_Gender=gender,
                        Node_Property_Restrictions=node_property_restrictions,
                        Property_Restrictions_Within_Node=ind_property_restrictions,
                        Actual_IndividualIntervention_Config=property_value_changer
                    )
                )
            )

        cb.add_event(changer_event)
    else:
        changer_event = CampaignEvent(
            Start_Day=start_day,
            Nodeset_Config=node_cfg,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Demographic_Coverage=coverage,
                Target_Demographic=target_group,
                Target_Age_Min=age_min,
                Target_Age_Max=age_max,
                Target_Gender=gender,
                Property_Restrictions_Within_Node=ind_property_restrictions,
                Node_Property_Restrictions=node_property_restrictions,
                Intervention_Config=property_value_changer
            )
        )
        cb.add_event(changer_event)

