from dtk.utils.Campaign.CampaignClass import *


def add_InputEIR(cb, monthlyEIRs, age_dependence="SURFACE_AREA_DEPENDENT", start_day=0, nodeIDs=None,
                 ind_property_restrictions=[]):
    """
    Create an intervention introducing new infections (see `InputEIR <https://institutefordiseasemodeling.github.io/EMOD/malaria/parameter-campaign.html#iv-inputeir>`_ for detail)
    If another InputEIR event is distributed to a node with an existing InputEIR event, the second one replaces the
        first (much like a new bednet replaces an old bednet).

    Args:
        cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` containing the campaign parameters
        monthlyEIRs: a list of monthly EIRs (must be 12 items)
        age_dependence: "LINEAR" or "SURFACE_AREA_DEPENDENT"
        start_day: Start day of the introduction of new infections
        nodeIDs: The list of nodes to apply this intervention to (**Node_List** parameter). If not provided, set value of NodeSetAll.

    Returns:
        None
    """

    nodes = NodeSetNodeList(Node_List=nodeIDs) if nodeIDs else NodeSetAll()

    if len(monthlyEIRs) is not 12:
        raise Exception('The input argument monthlyEIRs should have 12 entries, not %d' % len(monthlyEIRs))

    input_EIR_event = CampaignEvent(
        Event_Name="Input EIR intervention",
        Start_Day=start_day,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Number_Repetitions=-1,
            Intervention_Config=InputEIR(
                Age_Dependence=InputEIR_Age_Dependence_Enum[age_dependence],
                Monthly_EIR=monthlyEIRs
            )
        ),
        Nodeset_Config=nodes
    )

    if ind_property_restrictions:
        input_EIR_event.Event_Coordinator_Config.Intervention_Config["Property_Restrictions_Within_Node"] = ind_property_restrictions

    cb.add_event(input_EIR_event)