import copy
from dtk.utils.Campaign.CampaignClass import *
from dtk.interventions.triggered_campaign_delay_event import triggered_campaign_delay_event


# Ivermectin parameters
ivermectin_cfg = Ivermectin(
    Killing_Config=WaningEffectBox(
        Box_Duration=1,
        Initial_Effect=0.95
    ),
    Cost_To_Consumer=15.0
)

# set up events to broadcast when receiving campaign drug
receiving_IV_event = BroadcastEvent(Broadcast_Event="Received_Ivermectin")


def ivermectin_config_by_duration(drug_code=None):
    """
    Returns the correct ``Killing_Config`` parameter depending on the ``drug_code``

    :param drug_code: Can be ``'DAY'``, ``'WEEK'`` or ``'MONTH'`` or a number of days and drive the ``Killing_config`` (see `Killing_Config in Ivermectin <https://institutefordiseasemodeling.github.io/EMOD/malaria/parameter-campaign.html#iv-ivermectin>`_ for more info).
    :return: a dictionary with the correct ``Killing_Config / Box_Duration`` set.
    """

    if not drug_code:
        return {}
    cfg = copy.deepcopy(ivermectin_cfg)
    if isinstance(drug_code, str):
        if drug_code == 'DAY':
            cfg.Killing_Config.Box_Duration = 1
        elif drug_code == 'WEEK':
            cfg.Killing_Config.Box_Duration = 7
        elif drug_code == 'MONTH':
            cfg.Killing_Config.Box_Duration = 30
        else:
            raise Exception("Don't recognize drug_code" % drug_code)
    elif isinstance(drug_code, (int, float)):
        cfg.Killing_Config.Box_Duration = drug_code
    else:
        raise Exception("Drug code should be the IVM duration in days or a string like 'DAY', 'WEEK', 'MONTH'")

    return cfg


def add_ivermectin(config_builder, drug_code, coverage, start_days,
                   trigger_condition_list=[], triggered_campaign_delay=0,
                   listening_duration=-1, nodeids=[], target_residents_only=1,
                   node_property_restrictions=[], ind_property_restrictions=[]):
    """
    Add an ivermectin event to the config_builder passed.

    :param config_builder: The config builder getting the intervention event
    :param drug_code: Can be 'DAY', 'WEEK' or 'MONTH' and drive the ``Killing_config`` (see `Killing_Config in Ivermectin <https://institutefordiseasemodeling.github.io/EMOD/malaria/parameter-campaign.html#iv-ivermectin>`_ for more info).
    :param coverage: Set the ``Demographic_Coverage``
    :param start_days: list of days when to start the ivermectin distribution
    :param trigger_condition_list: ivermectin will be distributed when it hears the trigger string event, please note the start_days then is used to distribute the NodeLevelHealthTriggeredIV
    :param listening_duration: for how long the NLHTIV will listen for the trigger
    :return: Nothing
    """

    cfg = ivermectin_config_by_duration(drug_code)

    cfg = [cfg] + [receiving_IV_event]

    intervention_cfg = MultiInterventionDistributor(Intervention_List=cfg)

    if triggered_campaign_delay > 0:
        trigger_condition_list = [triggered_campaign_delay_event(config_builder, start_days[0],
                                                                 nodeIDs=nodeids,
                                                                 triggered_campaign_delay=triggered_campaign_delay,
                                                                 trigger_condition_list=trigger_condition_list,
                                                                 listening_duration=listening_duration,
                                                                 node_property_restrictions=node_property_restrictions)]

    if nodeids:
        node_cfg = NodeSetNodeList(Node_List=nodeids)
    else:
        node_cfg = NodeSetAll()

    for start_day in start_days:
        IVM_event = CampaignEvent(
            Start_Day=start_day,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(),
            Nodeset_Config=node_cfg
        )

        if trigger_condition_list:
            IVM_event.Event_Coordinator_Config.Intervention_Config = NodeLevelHealthTriggeredIV(
                Trigger_Condition_List=trigger_condition_list,
                Target_Residents_Only=target_residents_only,
                Property_Restrictions_Within_Node=ind_property_restrictions,
                Node_Property_Restrictions=node_property_restrictions,
                Duration=listening_duration,
                Demographic_Coverage=coverage,
                Actual_IndividualIntervention_Config=intervention_cfg
            )
        else:
            IVM_event.Event_Coordinator_Config.Target_Residents_Only = True if target_residents_only else False
            IVM_event.Event_Coordinator_Config.Demographic_Coverage = coverage
            IVM_event.Event_Coordinator_Config.Property_Restrictions_Within_Node=ind_property_restrictions
            IVM_event.Event_Coordinator_Config.Node_Property_Restrictions=node_property_restrictions
            IVM_event.Event_Coordinator_Config.Intervention_Config = intervention_cfg

        config_builder.add_event(IVM_event)