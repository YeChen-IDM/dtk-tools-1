from dtk.utils.Campaign.CampaignClass import *
import random


def triggered_campaign_delay_event(config_builder, start: int=0,  nodeIDs: list=None,
                                   delay_distribution: str="FIXED_DURATION",
                                   delay_period_mean: float=1, delay_period_std_dev: float=1,
                                   delay_period_max: float=1, coverage: float=1,
                                   triggered_campaign_delay: int=0, trigger_condition_list: list=None,
                                   listening_duration: int=-1, event_to_send_out: str=None,
                                   node_property_restrictions: list=None, ind_property_restrictions: list=None,
                                   only_target_residents: bool=1):
    """
    Creates a triggered campaign that broadcasts an event after a delay. The event it broadcasts can be specified
    or it is randomly generated. You can use either FIXED_DURATION or GAUSSIAN_DURATION for the delay.

    Args:

        config_builder: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` containing the intervention.
        start: the first day of the campaign.
        nodeIDs: The list of node IDs; if empty, defaults to all nodes.
        delay_distribution: The distribution of the length of the delay, possible values are FIXED_DURATION or
            GAUSSIAN_DURATION. FIXED_DURATION is the default
        delay_period_mean: Used with GAUSSIAN_DURATION; the mean time of the duration of the event being sent out.
        delay_period_std_dev: Used with GAUSSIAN_DURATION; the std deviation of the duration of the event being sent out.
        delay_period_max: Used with GAUSSIAN_DURATION; the maximum time of the duration of the event being sent out.
        coverage: The Demographic_Coverage of the distribution.
        triggered_campaign_delay: Used with FIXED_DURATION; the delay time of the event being sent out.
        trigger_condition_list: The list of events that trigger the delayed event broadcast, for
            example: ["HappyBirthday", "Received_Treatment"].
        listening_duration: The duration for which the event will listen for the trigger.
            The default is -1, which indicates "indefinitely/forever."
        event_to_send_out: If specified, this is the event that will be sent out after the delay.
        node_property_restrictions: Restricts the intervention based on a list of dictionaries of node properties in
            format: [{"Land":"Swamp", "Roads":"No"}, {"Land": "Forest"}];  the default is no restrictions. Restrictions
            within each dictionary are connected by "and," and within the list are connected by "or." In the example,
            the restrictions are nodes with (Swamp Land AND No Roads) OR (Forest Land).
        ind_property_restrictions: Restricts intervention based on list of dictionaries of individual properties in
            format: [{"BitingRisk":"High", "IsCool":"Yes}, {"IsRich": "Yes"}]; the default is no restrictions.
            Restrictions within each dictionary are connected by "and," and within the list are connected by "or." In the
            example, the restrictions are individuals with (High Biting Risk AND Yes IsCool) OR (IsRich).
        only_target_residents: The intervention only affects people who started the simulation  in the targeted node,
            and still are in the targeted node.

    Returns:
        The event that will be broadcast after the delay.

    """
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()
    if not node_property_restrictions:
        node_property_restrictions = []
    if not ind_property_restrictions:
        ind_property_restrictions = []
    if not event_to_send_out:
        event_to_send_out = 'Delayed_Event_%d' % random.randrange(100000)
    if not triggered_campaign_delay:
        raise ValueError("Why are you using this with a 0 delay? This is used to add delay to triggered events by "
                         "sending out a delayed event.\n")

    event_cfg = BroadcastEvent(Broadcast_Event=event_to_send_out)

    if delay_distribution == 'FIXED_DURATION':
        intervention = DelayedIntervention(
            Delay_Distribution=delay_distribution,
            Delay_Period=triggered_campaign_delay,
            Delay_Period_Fixed=triggered_campaign_delay, # new style
            Actual_IndividualIntervention_Configs=[event_cfg]
         )
    elif delay_distribution == 'GAUSSIAN_DURATION':
        intervention = DelayedIntervention(
            Delay_Distribution=delay_distribution,
            Delay_Period_Mean=delay_period_mean,
            Delay_Period_Std_Dev=delay_period_std_dev,
            Delay_Period_Max=delay_period_max,
            Delay_Period_Gaussian_Mean=delay_period_mean, # new style
            Delay_Period_Gaussian_Std_Dev=delay_period_std_dev,  #new style
            Delay_Period_Gaussian_Max=delay_period_max, # new style
            Actual_IndividualIntervention_Configs=[event_cfg]
            )
    else:
        raise ValueError("{} is not a recognized delay_distribution. Please use GAUSSIAN_DURATION or FIXED_DURAION.\n")

    triggered_delay = CampaignEvent(
        Start_Day=start,
        Nodeset_Config=node_cfg,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Trigger_Condition_List=trigger_condition_list,
                    Duration=listening_duration,
                    Demographic_Coverage=coverage,
                    Target_Residents_Only=only_target_residents,
                    Node_Property_Restrictions=node_property_restrictions,
                    Property_Restrictions_Within_Node=ind_property_restrictions,
                    Actual_IndividualIntervention_Config=intervention)
        )
        )
    config_builder.add_event(triggered_delay)

    return event_to_send_out
