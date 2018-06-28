import random
from dtk.interventions.triggered_campaign_delay_event import triggered_campaign_delay_event
from dtk.utils.Campaign.CampaignClass import *


# the old MigrateTo has now been split into MigrateIndividuals and MigrateFamily.
# add_migration_event adds a MigrateIndividuals event.
def add_migration_event(cb, nodeto, start_day=0, coverage=1, repetitions=1, tsteps_btwn=365,
                        duration_at_node_distr_type='FIXED_DURATION', 
                        duration_of_stay=100, duration_of_stay_2=0, 
                        duration_before_leaving_distr_type='FIXED_DURATION', 
                        duration_before_leaving=0, duration_before_leaving_2=0, 
                        target='Everyone', nodesfrom={"class": "NodeSetAll"},
                        ind_property_restrictions=[], node_property_restrictions=[], triggered_campaign_delay=0,
                        trigger_condition_list=[], listening_duration=-1):

    migration_event = MigrateIndividuals(
        NodeID_To_Migrate_To=nodeto,
        Is_Moving=False
    )

    migration_event = update_duration_type(migration_event, duration_at_node_distr_type,
                                           dur_param_1=duration_of_stay, dur_param_2=duration_of_stay_2,
                                           leaving_or_at='at')
    migration_event = update_duration_type(migration_event, duration_before_leaving_distr_type,
                                           dur_param_1=duration_before_leaving, dur_param_2=duration_before_leaving_2,
                                           leaving_or_at='leaving')

    if trigger_condition_list:
        if repetitions > 1 or triggered_campaign_delay > 0:
            event_to_send_out = random.randrange(100000)
            for x in range(repetitions):
                # create a trigger for each of the delays.
                triggered_campaign_delay_event(cb, start_day, nodesfrom,
                                               triggered_campaign_delay + x * tsteps_btwn,
                                               trigger_condition_list,
                                               listening_duration, event_to_send_out)
            trigger_condition_list = [str(event_to_send_out)]

        event = CampaignEvent(
            Event_Name="Migration Event Triggered",
            Start_Day=start_day,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Intervention_Config=NodeLevelHealthTriggeredIV(
                    Duration=listening_duration,
                    Trigger_Condition_List=trigger_condition_list,
                    Target_Demographic=StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum[target],
                    Target_Residents_Only=True,
                    Node_Property_Restrictions=node_property_restrictions,
                    Property_Restrictions_Within_Node=ind_property_restrictions,
                    Demographic_Coverage=coverage,
                    Actual_IndividualIntervention_Config=migration_event
                )
            ),
            Nodeset_Config=nodesfrom
        )

        if isinstance(target, dict) and all([k in target.keys() for k in ['agemin', 'agemax']]):
            event.Event_Coordinator_Config.Intervention_Config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.ExplicitAgeRanges
            event.Event_Coordinator_Config.Intervention_Config.Target_Age_Min = target['agemin']
            event.Event_Coordinator_Config.Intervention_Config.Target_Age_Max = target['agemax']

    else:
        event = CampaignEvent(
            Event_Name="Migration Event",
            Start_Day=start_day,
            Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                Property_Restrictions_Within_Node=ind_property_restrictions,
                Node_Property_Restrictions=node_property_restrictions,
                Number_Distributions=-1,
                Number_Repetitions=repetitions,
                Target_Residents_Only=True,
                Target_Demographic=StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum[target],
                Timesteps_Between_Repetitions=tsteps_btwn,
                Demographic_Coverage=coverage,
                Intervention_Config=migration_event
            ),
            Nodeset_Config=nodesfrom
        )

        if isinstance(target, dict) and all([k in target for k in ['agemin', 'agemax']]):
            event.Event_Coordinator_Config.Target_Demographic = StandardInterventionDistributionEventCoordinator_Target_Demographic_Enum.ExplicitAgeRanges
            event.Event_Coordinator_Config.Target_Age_Min = target['agemin']
            event.Event_Coordinator_Config.Target_Age_Max = target['agemax']

    cb.add_event(event)


def update_duration_type(migration_event, duration_at_node_distr_type, dur_param_1=0, dur_param_2=0, leaving_or_at='at') :

    if leaving_or_at == 'leaving':
        trip_end = 'Before_Leaving'
        MigrateFamily_Duration_Enum = MigrateIndividuals_Duration_Before_Leaving_Distribution_Type_Enum
    else:
        trip_end = 'At_Node'
        MigrateFamily_Duration_Enum = MigrateIndividuals_Duration_At_Node_Distribution_Type_Enum

    if duration_at_node_distr_type == 'FIXED_DURATION' :
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.FIXED_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Fixed", dur_param_1)
    elif duration_at_node_distr_type == 'UNIFORM_DURATION' :
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.UNIFORM_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Uniform_Min", dur_param_1)
        setattr(migration_event, "Duration_" + trip_end + "_Uniform_Max", dur_param_2)
    elif duration_at_node_distr_type == 'GAUSSIAN_DURATION' :
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.GAUSSIAN_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Gausian_Mean", dur_param_1)
        setattr(migration_event, "Duration_" + trip_end + "_Gausian_StdDev", dur_param_2)
    elif duration_at_node_distr_type == 'EXPONENTIAL_DURATION' :
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.EXPONENTIAL_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Exponential_Period", dur_param_1)
    elif duration_at_node_distr_type == 'POISSON_DURATION' :
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.POISSON_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Poisson_Mean", dur_param_1)
    else:
        print("warning: unsupported duration distribution type, reverting to fixed duration")
        setattr(migration_event, "Duration_" + trip_end + "_Distribution_Type", MigrateFamily_Duration_Enum.FIXED_DURATION)
        setattr(migration_event, "Duration_" + trip_end + "_Fixed", dur_param_1)

    return migration_event
