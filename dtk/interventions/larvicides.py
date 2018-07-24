import copy
from dtk.utils.Campaign.CampaignClass import *


default_larvicides = Larvicides(
    Blocking_Config=WaningEffectBoxExponential(
        Box_Duration=100,
        Decay_Time_Constant=150,
        Initial_Effect=0.4
    ),
    Cost_To_Consumer=1.0,
    Habitat_Target=Larvicides_Habitat_Target_Enum.ALL_HABITATS,
    Killing_Config=WaningEffectBoxExponential(
        Box_Duration=100,
        Decay_Time_Constant=150,
        Initial_Effect=0.2
    )
)


def add_larvicides(config_builder, start, killing=None, reduction=None, habitat_target=None, waning=None, nodesIDs = None):

    event = CampaignEvent(
        Start_Day=start,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator()
    )

    # Take care of specified NodeIDs if any
    if not nodesIDs:
        event.Nodeset_Config = NodeSetAll()
    else:
        event.Nodeset_Config = NodeSetNodeList(Node_List=nodesIDs)

    # Copy the default event
    larvicides_event = copy.deepcopy(default_larvicides)

    # Change according to parameters
    if killing:
        larvicides_event.Killing_Config.Initial_Effect = killing

    if reduction:
        larvicides_event.Blocking_Config.Initial_Effect = reduction

    if habitat_target:
        larvicides_event.Habitat_Target = Larvicides_Habitat_Target_Enum[habitat_target]

    if waning:
        if "blocking" in waning:
            for k, v in waning["blocking"].items():
                setattr(larvicides_event.Blocking_Config, k, v)
        if "killing" in waning:
            for k, v in waning["killing"].items():
                setattr(larvicides_event.Killing_Config, k, v)

    # Add the larvicides to the event coordinator
    event.Event_Coordinator_Config.Intervention_Config = larvicides_event

    # Add to the config builder
    config_builder.add_event(event)
