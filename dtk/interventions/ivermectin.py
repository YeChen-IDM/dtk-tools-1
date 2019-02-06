import copy
from dtk.utils.Campaign.CampaignClass import *
from dtk.interventions.triggered_campaign_delay_event import triggered_campaign_delay_event


# Ivermectin parameters
ivermectin_cfg = Ivermectin(
    Killing_Config=WaningEffectBox(
        Box_Duration=7,
        Initial_Effect=0.95
    ),
    Cost_To_Consumer=1.0
)

# set up events to broadcast when receiving campaign drug
receiving_IV_event = BroadcastEvent(Broadcast_Event="Received_Ivermectin")


def ivermectin_config_by_duration(box_duration='WEEK', initial_effect: float=0.95):
    """
    Returns an ivermectin configuration with the Box_Duration set to "drug_code" parameter. Default being 7 days (WEEK)
    Args:
        box_duration: configures the length of ivermectin effect, its Box_Duration. Can be "DAY", "WEEK", "MONTH",
            "90DAYS", or an integer or float of the number of days of the effect.
        initial_effect: Initial strength of the effect, example: 0.9 (about 90% effective)

    Returns:
            Returns a dictionary of ivermectin configuration with the Box_Duration set to "drug_code" parameter.
    """

    cfg = copy.deepcopy(ivermectin_cfg)
    if isinstance(box_duration, str):
        if box_duration == 'DAY':
            cfg.Killing_Config.Box_Duration = 1
        elif box_duration == 'WEEK':
            cfg.Killing_Config.Box_Duration = 7
        elif box_duration == 'MONTH':
            cfg.Killing_Config.Box_Duration = 30
        elif box_duration == '90DAYS':
            cfg.Killing_Config.Box_Duration = 90
        else:
            raise ValueError("Don't recognize box_duration : {} \nPlease pass in 'DAY','WEEK','MONTH', or"
                             " '90DAYS'\n".format(box_duration))
    elif isinstance(box_duration, (int, float)):
        cfg.Killing_Config.Box_Duration = box_duration
    else:
        raise ValueError("Please pass in a valid drug code.\nOptions: Sting - 'DAY','WEKK','MONTH', or "
                         "'90DAYS'.\nInt or Float of number of days for the Box_Duration.\n")

    cfg.Killing_Config.Initial_Effect = initial_effect


    return cfg


def add_ivermectin(config_builder, box_duration: any="WEEK", initial_effect: float=0.95, coverage: float=1.0,
                   start_days: list=None, trigger_condition_list: list=None, triggered_campaign_delay: int=0,
                   listening_duration: int=-1, nodeIDs: list=None, target_group: any="Everyone",
                   target_residents_only: bool=1, node_property_restrictions: list=None,
                   ind_property_restrictions: list=None):
    """
    Adds an ivermectin distribution event
    Args:
        config_builder: the config builder getting the event
        box_duration: configures the length of ivermectin effect for Box_Duration. Can be "DAY", "WEEK", "MONTH",
            "90DAYS", or an integer or float of the number of days of the effect.
        initial_effect: Initial strength of the effect, example: 0.9 (about 90% effective)
        coverage: sets the "Demographic_Coverage", example: 0.7 (about 70% of people will receive ivermectin)
        start_days: list of integer days on which ivermectin will be distributed, example:[1,31,61,91]
        trigger_condition_list: makes ivermectin distribution a triggered event that's distributed on the first of the
            start_days, example: ["NewClinicalCase", "NewInfection"]
        triggered_campaign_delay: number of days campaign is delayed after being triggered, ex: 3
        listening_duration: how many days the triggered campaign will be active for, -1 indicates "indefinitely"
        nodeIDs: list of nodes to which the campaign will be distributed, example:[2384,12,932]
        target_group:  dictionary of {'agemin' : x, 'agemax' : y, 'gender':} to target  to individuals between
            x and y years of age. Default is 'Everyone'
        target_residents_only: if only the people who started out the simulation in this node will be affected
        node_property_restrictions: Restricts intervention based on list of dictionaries of node properties in
            format: [{"Land":"Swamp", "Roads":"No"}, {"Land": "Forest"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are nodes with (Swamp Land AND No Roads) OR (Forest Land) nodes
        ind_property_restrictions: Restricts intervention based on list of dictionaries of individual properties in
            format: [{"BitingRisk":"High", "IsCool":"Yes}, {"IsRich": "Yes"}]; default is no restrictions, with
            restrictions within each dictionary are connected with "and" and within the list are "or", so the
            example restrictions are individuals with (High Biting Risk AND Yes IsCool) OR (IsRich) individuals

    Returns:
        Nothing, Ivermectin campaign is added to the final campaign.json file created by the config_builder

    """

    if node_property_restrictions is None:
        node_property_restrictions = []
    if ind_property_restrictions is None:
        ind_property_restrictions = []
    if not start_days:
        start_days = [0]
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    cfg = [ivermectin_config_by_duration(box_duration, initial_effect), receiving_IV_event]
    intervention_cfg = MultiInterventionDistributor(Intervention_List=cfg)

    if triggered_campaign_delay > 0:
        if not trigger_condition_list:
            raise Exception("When using triggered_campaign_delay, please specify triggered_condition_list, too.\n")
        trigger_condition_list = [triggered_campaign_delay_event(config_builder, start_days[0],
                                                                 nodeIDs=nodeIDs,
                                                                 triggered_campaign_delay=triggered_campaign_delay,
                                                                 trigger_condition_list=trigger_condition_list,
                                                                 listening_duration=listening_duration,
                                                                 ind_property_restrictions=ind_property_restrictions,
                                                                 node_property_restrictions=node_property_restrictions)]

    gender = "All"
    age_min = 0
    age_max = 150
    if target_group != "Everyone" and isinstance(target_group, dict):
        try:
            age_min = target_group["agemin"]
            age_max = target_group["agemax"]
            if 'gender' in target_group:
                gender = target_group["gender"]
                target_group = "ExplicitAgeRangesAndGender"
            else:
                target_group = "ExplicitAgeRanges"
        except:
            raise ValueError("Unknown target_group parameter. Please pass in 'Everyone' or a dictionary of "
                             "{'agemin' : x, 'agemax' : y, 'gender': 'Female'} to target  to individuals between x and "
                             "y years of age, and (optional) gender.\n")

    if trigger_condition_list:
        ivm_event = CampaignEvent(
                    Start_Day=start_days[0],
                    Nodeset_Config=node_cfg,
                    Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                        Intervention_Config=NodeLevelHealthTriggeredIV(
                            Trigger_Condition_List=trigger_condition_list,
                            Target_Residents_Only=target_residents_only,
                            Property_Restrictions_Within_Node=ind_property_restrictions,
                            Node_Property_Restrictions=node_property_restrictions,
                            Duration=listening_duration,
                            Demographic_Coverage=coverage,
                            Target_Demographic=target_group,
                            Target_Age_Min=age_min,
                            Target_Age_Max=age_max,
                            Target_Gender=gender,
                            Actual_IndividualIntervention_Config=intervention_cfg)
                    )
        )
        config_builder.add_event(ivm_event)
    else:
        for start_day in start_days:
                ivm_event = CampaignEvent(
                    Start_Day=start_day,
                    Nodeset_Config=node_cfg,
                    Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                        Target_Residents_Only=target_residents_only,
                        Demographic_Coverage=coverage,
                        Property_Restrictions_Within_Node=ind_property_restrictions,
                        Node_Property_Restrictions=node_property_restrictions,
                        Target_Demographic=target_group,
                        Target_Age_Min=age_min,
                        Target_Age_Max=age_max,
                        Target_Gender=gender,
                        Intervention_Config=intervention_cfg)
                )
                config_builder.add_event(ivm_event)
