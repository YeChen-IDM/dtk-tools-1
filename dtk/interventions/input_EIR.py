from dtk.utils.Campaign.CampaignClass import *


def add_InputEIR(cb, monthlyEIRs, age_dependence="SURFACE_AREA_DEPENDENT", start_day=0, nodes={"class": "NodeSetAll"},
                 ind_property_restrictions=[]):
    """
    Create an intervention introducing new infections (see `InputEIR <https://institutefordiseasemodeling.github.io/EMOD/malaria/parameter-campaign.html#iv-inputeir>`_ for detail)

    :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` containing the campaign parameters
    :param monthlyEIRs: a list of monthly EIRs (must be 12 items)
    :param age_dependence: "LINEAR" or "SURFACE_AREA_DEPENDENT"
    :param start_day: Start day of the introduction of new infections
    :return: Nothing
    """
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
        Nodeset_Config=NodeSetAll()
    )

    if ind_property_restrictions:
        input_EIR_event.Event_Coordinator_Config.Intervention_Config["Property_Restrictions_Within_Node"] = ind_property_restrictions

    cb.add_event(input_EIR_event)