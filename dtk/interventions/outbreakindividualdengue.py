from dtk.utils.Campaign.CampaignClass import *
from dtk.utils.Campaign.CampaignEnum import *


# Add dengue outbreak individual event
def add_OutbreakIndividualDengue(config_builder, start, coverage_by_age, strain_id_name, nodeIDs=[]):
    dengue_event = CampaignEvent(
        Start_Day=int(start),
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Intervention_Config=OutbreakIndividualDengue(
                Strain_Id_Name=strain_id_name,  # eg. "Strain_1"
                Antigen=0,
                Genome=0,
                Comment_antigen_genome="See GitHub https://github.com/InstituteforDiseaseModeling/DtkTrunk/issues/1682",
                Incubation_Period_Override=-1
            )
        )
    )

    if all([k in coverage_by_age.keys() for k in ['min', 'max']]):
        dengue_event.Event_Coordinator_Config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.ExplicitAgeRanges
        dengue_event.Event_Coordinator_Config.Target_Age_Min = coverage_by_age["min"]
        dengue_event.Event_Coordinator_Config.Target_Age_Max = coverage_by_age["max"]

    # not sure else is the correct way to do eg.{min: 0} or {max: 1.725}
    else:
        dengue_event.Event_Coordinator_Config.Demographic_Coverage = 0
        dengue_event.Event_Coordinator_Config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.Everyone

    if not nodeIDs:
        dengue_event.Nodeset_Config = NodeSetAll()
    else:
        dengue_event.Nodeset_Config = NodeSetNodeList(Node_List=nodeIDs)

    config_builder.add_event(dengue_event)
