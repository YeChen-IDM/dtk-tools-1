

def add_event_reporter(cb,
                       event_type="Individual",
                       events_list=[],
                       node_properties=[],
                       individual_properties=[],
                       ignore_events_in_list=0
                       ):
    """
        Adds event reporters/recorders to any type of the event

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param event_type: event type that is being recorded, default is Individual
        :param events_list: list of events you want recorded
        :param node_properties: An array of node property keys. Each key will be a column in the report.
        :param individual_properties: An array of Individual Property keys/names. For each the value of each key,
        there will be two columns - Key:Value:NumIndividuals, Key:Value:NumInfected
        :param ignore_events_in_list: flag that indicates that you want the events in the lists ignored,
        and the inverse recorded
        """
    event_type_l = event_type.lower()
    if event_type_l == "individual":
        event_recorder = {
            "Report_Event_Recorder": 1,
            "Report_Event_Recorder_Events": events_list,
            "Report_Event_Recorder_Individual_Properties":individual_properties,
            "Report_Event_Recorder_Ignore_Events_In_List": ignore_events_in_list
        }
    elif event_type_l == "node":
        event_recorder = {
            "Report_Node_Event_Recorder": 1,
            "Report_Node_Event_Recorder_Events": events_list,
            "Report_Node_Event_Recorder_Node_Properties": node_properties,
            "Report_Node_Event_Recorder_Ignore_Events_In_List": ignore_events_in_list
        }
    elif event_type_l == "coordinator":
        event_recorder = {
            "Report_Coordinator_Event_Recorder": 1,
            "Report_Coordinator_Event_Recorder_Events": events_list,
            "Report_Coordinator_Event_Recorder_Ignore_Events_In_List": ignore_events_in_list
        }
    else:
        raise ValueError("event_type {} unknown".format(event_type))

    cb.update_params(event_recorder)

def add_surveillance_event_recorder(cb,
                       events_list=[],
                       country_resources=[],
                       individual_properties=[],
                       ignore_events_in_list=0
                       ):
    """
        Adds surveillance event recorder.

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param events_list: list of events you want recorded
        :param country_resources: An array of CountryResource names. There will be a column for each name.
        :param individual_properties: An array of Individual Property keys/names. For each the value of each key,
        there will be two columns - Key:Value:NumIndividuals, Key:Value:NumInfected
        :param ignore_events_in_list: flag that indicates that you want the events in the lists ignored,
        and the inverse recorded
        """

    event_recorder = {
        "Report_Surveillance_Event_Recorder": 1,
        "Report_Surveillance_Event_Recorder_Events": events_list,
        "Report_Surveillance_Event_Recorder_Country_Resources": country_resources,
        "Report_Surveillance_Event_Recorder_Stats_By_IPs": individual_properties,
        "Report_Surveillance_Event_Recorder_Ignore_Events_In_List": ignore_events_in_list
    }
    cb.update_params(event_recorder)


def add_coordinator_event(cb,
                          start_day=0,
                          node_ids=None,
                          cost_to_consumer=0,
                          coordinator_name="BroadcastCoordinatorEvent",
                          event=None
                          ):
    """
        Adds a scheduled broadcast event, default is the COORDINATOR event

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param cost_to_consumer: cost to consumer per broadcasted event
        :param coordinator_name: the name of the coordinator (used in reports)

        :param event: a string that is the event to be sent out
        """
    if not event:
        raise ValueError("event needs to be explicitly defined")

    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    broadcaster = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": {
            "class": "BroadcastCoordinatorEvent",
            "Coordinator_Name": coordinator_name,
            "Cost_To_Consumer": cost_to_consumer,
            "Broadcast_Event": event
        }
    }

    cb.add_event(broadcaster)


def add_broadcast_event(cb,
                        start_day=0,
                        node_ids=[],
                        cost_to_consumer=0,
                        node_property_restrictions=[],
                        ind_property_restrictions=[],
                        demographic_coverage=1,
                        target_demographic="Everyone",
                        event_type="COORDINATOR",
                        event=None,
                        ):
    """
        Adds a scheduled broadcast event, default is the COORDINATOR event

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param node_ids: list of node IDs which this will affect
        :param cost_to_consumer: cost to consumer per broadcasted event
        :param node_property_restrictions: only for Node-type events:
        list of dictionaries of node properties to which the campaign  will be
        restricted : [{ "NodeProperty1" : "PropertyValue1" }, {'NodeProperty2': "PropertyValue2"}, ...]
        :param ind_property_restrictions: used with Property_Restrictions_Within_Node. Format: list of dicts:
        [{ "IndividualProperty1" : "PropertyValue1" }, {'IndividualProperty2': "PropertyValue2"}, ...]
        :param demographic_coverage: This is the percentage of individuals that emit an event of interest
        and meet the demographic restrictions.
        :param target_demographic: to restrict monitoring by age, dict of {'agemin' : x, 'agemax' : y}.
        Default is targeting everyone.
        :param event_type: Whether the event is a "coordinator"-type or "node"-type event
        :param event: a string that is the event to be sent out
        """
    if not event:
        raise ValueError("event needs to be explicitly defined")

    # setting up event to broadcast
    event_type_l = event_type.lower()
    if event_type_l == "coordinator":
        broadcast_event = {
            "class": "BroadcastCoordinatorEvent",
            "Coordinator_Name": "BroadcastCoordinatorEvent",
            "Cost_To_Consumer": cost_to_consumer,
            "Broadcast_Event": event
        }
    elif event_type_l == "node":
        broadcast_event = {
            'class': 'StandardInterventionDistributionEventCoordinator',
            "Node_Property_Restrictions": node_property_restrictions,
            "Intervention_Config": {
                    "class": "BroadcastNodeEvent",
                    "Cost_To_Consumer": cost_to_consumer,
                    "Broadcast_Event": event
            }
        }
    elif event_type_l == "individual":
        broadcast_event = {
            'class': 'StandardInterventionDistributionEventCoordinator',
            "Demographic_Coverage": demographic_coverage,
            "Node_Property_Restrictions": node_property_restrictions,
            "Property_Restrictions_Within_Node": ind_property_restrictions,
            "Intervention_Config": {
                "class": "BroadcastEvent",
                "Cost_To_Consumer": cost_to_consumer,
                "Broadcast_Event": event
            }
        }
        # setting up the target demographic
        if target_demographic != 'Everyone':
            target_demographic = {
                "Target_Demographic": "ExplicitAgeRanges",  # Otherwise default is Everyone
                "Target_Age_Min": target_demographic['agemin'],
                "Target_Age_Max": target_demographic['agemax']
            }
        else:
            target_demographic = {"Target_Demographic": "Everyone"}
        broadcast_event = {**broadcast_event, **target_demographic}  # this concatenates the two dictionaries
    else:
        raise ValueError("event_type {} unknown".format(event_type))

    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    broadcaster = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": broadcast_event
    }

    cb.add_event(broadcaster)


def add_triggered_environmental_diagnostic(cb,
                                         start_day=0,
                                         duration=-1,
                                         coordinator_name="TriggeredEnvironmentalDiagnostic",
                                         start_triggers_list=None,
                                         stop_triggers_list=[],
                                         completion_event="NoTrigger",
                                         repetitions=1,
                                         tsteps_btwn_repetitions=365,
                                         node_ids=[],
                                         node_property_restrictions=[],
                                         sample_threshold=1000,
                                         sensitivity=1,
                                         specificity=1,
                                         environment_ip_value="",
                                         positive_diagnostic_event=None,
                                         negative_diagnostic_event=None
                                         ):
    """
        Adds a triggered environmental diagnostic.

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param coordinator_name: the name of the coordinator (used in reports)
        :param duration: duration of time for which the diagnostic exists and listens for the trigger
        :param start_triggers_list: list of coordinator-type events for which the diagnostic listens to start running, cannot be empty
        :param stop_triggers_list: list of coordinator-type events which stop the diagnostic run (can be initiated by diagnostic)
        :param completion_event: coordinator-type event that is sent out when the diagnostic run is done
        :param repetitions: Number of repetitions of the diagnostic test
        :param tsteps_btwn_repetitions: days between repetitions of the diagnostic test
        :param node_ids: list of node IDs which diagnostic will test, there will be a diagnostic per node
        :param node_property_restrictions: list of dictionaries of node properties to which the diagnostic will be restricted
        Format: list of dicts: [{ "NodeProperty1" : "PropertyValue1" }, {'NodeProperty2': "PropertyValue2"}, ...]
        >> below are diagnostic-specific parameters <<<
        :param sample_threshold: a sample GREATER THAN the threshold triggers a positive diagnosis
        :param environment_ip_value: restriction based on property value of the environment, format: "Geographic:VillageA"
        :param positive_diagnostic_event: node-level event sent out when the sample passes the sample_threshold
        :param negative_diagnostic_event: node-level event sent out when the sample is less than or equal to the sample_threshold

        """
    if not start_triggers_list or not positive_diagnostic_event:
        raise ValueError("start_triggers_list, positive_diagnostic_event all need to be explicitly defined")

    environmental_diagnostic = {
                "class": "EnvironmentalDiagnostic",
                "Sample_Threshold": sample_threshold,
                "Environment_IP_Key_Value": environment_ip_value,
                "Base_Sensitivity" : sensitivity,
                "Base_Specificity" : specificity,
                "Positive_Diagnostic_Event": positive_diagnostic_event,
                "Negative_Diagnostic_Event": negative_diagnostic_event
            }
    # if no negative events defined, delete the parameter
    if not negative_diagnostic_event:
        del environmental_diagnostic["Negative_Diagnostic_Event"]

    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    triggered_coordinator = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": {
            "class": "TriggeredEventCoordinator",
            "Coordinator_Name": coordinator_name,
            "Start_Trigger_Condition_List": start_triggers_list,
            "Stop_Trigger_Condition_List": stop_triggers_list,
            "Duration": duration,
            "Number_Repetitions": repetitions,
            "Timesteps_Between_Repetitions": tsteps_btwn_repetitions,
            "Node_Property_Restrictions": node_property_restrictions,
            "Intervention_Config": environmental_diagnostic,
            "Completion_Event": completion_event
        }
    }


    cb.add_event(triggered_coordinator)


def add_triggered_surveillance_coordinator(cb,
                                           start_day=0,
                                           duration=-1,
                                           coordinator_name="SurveillanceEventCoordinator",
                                           start_triggers_list=None,
                                           stop_triggers_list=[],
                                           node_ids=[],
                                           counter_type="PERIODIC",
                                           counter_period=30,
                                           counter_event_type="NODE",
                                           counter_event_list=None,
                                           node_property_restrictions=[],
                                           ind_property_restrictions=[],
                                           target_demographic="Everyone",
                                           demographic_coverage=1,
                                           threshold_type="COUNT",
                                           action_list=None
                                           ):
    """
        Adds a triggered surveillance. Once triggered, surveys indefinitely every count period. To stop after the first
        run: use the Completion_Event as the Stop Trigger event (both would have to be set explicitly)

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param coordinator_name: name of the coordinator (used in reporting)
        :param duration: duration of time for which the diagnostic exists and listens for the trigger
        :param start_triggers_list: list of coordinator-type events for which the diagnostic listens to start running,
        cannot be empty
        :param stop_triggers_list: list of coordinator-type events which stop the diagnostic run (can be initiated by
        diagnostic)
        :param node_ids: list of node IDs which diagnostic will test, there will be a diagnostic per node
        :param counter_type: PERIODIC - Events are counted during the period. At the end of the period, the
        counter will notify the Responder that it is done counting.
        ROLLING_WINDOW - The counter will count events for a user defined period. Once the counter has been active for
        longer than the period, it will decrement counts incurred from the timestep at the beginning of the period and
        add counts for the current timestep. Also, once its active time is >= to the period, it will tell the responder
        to consider responding each timestep.
        :param counter_period: period in timesteps for which to count or the size in timesteps of the rolling window,
         depending on counter_type
        :param counter_event_type: This defines the type of event that the counter will be counting
        :param counter_event_list: list of the events you're counting
        :param node_property_restrictions: list of dictionaries of node properties to which the diagnostic will be
        restricted : [{ "NodeProperty1" : "PropertyValue1" }, {'NodeProperty2': "PropertyValue2"}, ...]
        :param ind_property_restrictions: used with Property_Restrictions_Within_Node. Format: list of dicts:
        [{ "IndividualProperty1" : "PropertyValue1" }, {'IndividualProperty2': "PropertyValue2"}, ...]
        :param target_demographic: to restrict monitoring by age, dict of {'agemin' : x, 'agemax' : y}.
        Default is targeting everyone.
        :param demographic_coverage: This is the percentage of individuals that emit an event of interest
        and meet the demographic restrictions.
        :param threshold_type: COUNT - This is a raw count of events. It should be adjusted based on
        Base_Population_Scale_Factor. If the adjusted value is <= 1, a warning should be given.
        PERCENTAGE - When using this type, the Responder will count the number of individuals that meet the Restrictions
        and then divide the number of events counted by this number. One should note that it is possible that a person
        emitting the event might not be counted in the denominator because their demographic restriction attributes
        changed between the time they emitted the event and the time the denominator was counted.
        :param action_list: a list of lists of responder's actions, will be translated into dictionaries,
                in format ["Threshold", "Event_Type", "Event_To_Broadcast"], so
                [[1,"COORDINATOR", "Small_Campaign], [10,"NODE","Big_Campaign"], etc]

        """
    if not start_triggers_list or not action_list or not counter_event_list:
        raise ValueError("start_triggers_list, counter_event_list, action_list all need to be explicitly defined")

    # setting up the incidence counter
    if target_demographic != 'Everyone':
        target_demographic = {
            "Target_Demographic": "ExplicitAgeRanges",  # Otherwise default is Everyone
            "Target_Age_Min": target_demographic['agemin'],
            "Target_Age_Max": target_demographic['agemax']
        }
    else:
        target_demographic = {"Target_Demographic": "Everyone"}

    incidence_counter = {
                "Counter_Type": counter_type,
                "Counter_Period": counter_period,
                "Counter_Event_Type": counter_event_type,
                "Trigger_Condition_List": counter_event_list,
                "Node_Property_Restrictions": node_property_restrictions,
                "Property_Restrictions_Within_Node": ind_property_restrictions,
                "Demographic_Coverage": demographic_coverage
            }

    incidence_counter = {**incidence_counter, **target_demographic} # this concatenates the two dictionaries

    # setting up the responder
    action_dictionary_list = []
    for action in action_list:
        action_dictionary = {
            "Threshold": action[0],
            "Event_Type": action[1],
            "Event_To_Broadcast": action[2]
        }
        action_dictionary_list.append(action_dictionary)

    responder = {
        "Threshold_Type": threshold_type,
        "Action_List": action_dictionary_list
        }

    # setting up the surveillance coordinator
    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    surveillance_coordinator = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": {
            "class": "SurveillanceEventCoordinator",
            "Coordinator_Name": coordinator_name,
            "Start_Trigger_Condition_List": start_triggers_list,
            "Stop_Trigger_Condition_List": stop_triggers_list,
            "Duration": duration,
            "Incidence_Counter": incidence_counter,
            "Responder": responder
        }
    }

    cb.add_event(surveillance_coordinator)


def add_triggered_vaccination(cb,
                              start_day=0,
                              duration=-1,
                              coordinator_name="The_Vaccinator",
                              start_triggers_list=None,
                              stop_triggers_list=[],
                              completion_event="",
                              node_ids=[],
                              node_property_restrictions=[],
                              ind_property_restrictions=[],
                              target_demographic="Everyone",
                              demographic_coverage=1,
                              repetitions=1,
                              tsteps_btwn_repetitions=365,
                              cost_to_consumer=0,
                              vaccine_take=1,
                              box_duration=730,
                              initial_effect=0.59
                              ):
    """
        This is a triggered vaccination of as AquisitionBlocking SimpleVaccine that uses a BOXDURABILITY profile with a
        WaningEffectBox.

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param duration: duration of time for which the diagnostic exists and listens for the trigger, -1 is forever
        :param coordinator_name: name of the coordinator (used by the reporter)
        :param start_triggers_list: list of coordinator-type events for which the diagnostic listens to start running,
        cannot be empty
        :param stop_triggers_list: list of coordinator-type events which stop the diagnostic run (can be initiated by
        diagnostic)
        :param completion_event: coordinator-type event sent out at the end of the campaign
        :param node_ids: list of node IDs which diagnostic will test, there will be a diagnostic per node
        :param node_property_restrictions: list of dictionaries of node properties to which the diagnostic will be
        restricted : [{ "NodeProperty1" : "PropertyValue1" }, {'NodeProperty2': "PropertyValue2"}, ...]
        :param ind_property_restrictions: used with Property_Restrictions_Within_Node. Format: list of dicts:
        [{ "IndividualProperty1" : "PropertyValue1" }, {'IndividualProperty2': "PropertyValue2"}, ...]
        :param target_demographic: to restrict monitoring by age, dict of {'agemin' : x, 'agemax' : y}.
        Default is targeting everyone.
        :param demographic_coverage: This is the percentage of individuals that emit an event of interest
        and meet the demographic restrictions.
        :param repetitions: Number of repetitions of the vaccination
        :param tsteps_btwn_repetitions: days between repetitions of the vaccination
        :param cost_to_consumer: cost of each dose of vaccine given
        :param vaccine_take: The rate at which delivered vaccines will successfully stimulate an immune response and
        achieve the desired efficacy.
        :param box_duration: The box duration of the effect in days.
        :param initial_effect: Initial strength of the effect.

        """
    if not start_triggers_list:
        raise ValueError("start_triggers_list needs to be explicitly defined")

    vaccine = {
            "Cost_To_Consumer": cost_to_consumer,
            "Durability_Time_Profile": "BOXDURABILITY",
            "Vaccine_Take": vaccine_take,
            "Vaccine_Type": "AcquisitionBlocking",
            "Waning_Config": {
                "Box_Duration": box_duration,
                "Box_Duration__KP_Duration2": "<-- MARKER",
                "Initial_Effect": initial_effect,
                "Initial_Effect__KP_Efficacy2": "<-- MARKER",
                "class": "WaningEffectBox"
            },
            "class": "SimpleVaccine"
        }

    # setting up the triggered coordinator
    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    triggered_coordinator = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": {
            "class": "TriggeredEventCoordinator",
            "Coordinator_Name": coordinator_name,
            "Start_Trigger_Condition_List": start_triggers_list,
            "Stop_Trigger_Condition_List": stop_triggers_list,
            "Duration": duration,
            "Number_Repetitions": repetitions,
            "Demographic_Coverage": demographic_coverage,
            "Timesteps_Between_Repetitions": tsteps_btwn_repetitions,
            "Node_Property_Restrictions": node_property_restrictions,
            "Property_Restrictions_Within_Node": ind_property_restrictions,
            "Intervention_Config": vaccine,
            "Completion_Event": completion_event
        }
    }

    # setting up to add demographic age restrictions to the event coordinator
    if target_demographic != 'Everyone':
        target_demographic = {
            "Target_Demographic": "ExplicitAgeRanges",  # Otherwise default is Everyone
            "Target_Age_Min": target_demographic['agemin'],
            "Target_Age_Max": target_demographic['agemax']
        }
    else:
        target_demographic = {"Target_Demographic": "Everyone"}

    triggered_coordinator["Event_Coordinator_Config"] = {**triggered_coordinator["Event_Coordinator_Config"],
                                                         **target_demographic}  # this concatenates the two dictionaries

    cb.add_event(triggered_coordinator)


def add_triggered_property_value_changer(cb,
                                         start_day=0,
                                         duration=-1,
                                         start_triggers_list=None,
                                         stop_triggers_list=[],
                                         coordinator_name="PropertyValueChanger",
                                         node_ids=[],
                                         node_property_restrictions=[],
                                         ind_property_restrictions=[],
                                         target_demographic="Everyone",
                                         completion_event="",
                                         demographic_coverage=1,
                                         target_property_name=None,
                                         target_property_value=None,
                                         daily_prob=1,
                                         max_duration=7,
                                         revert=False
                                         ):
    """
        This is a triggered vaccination of as AquisitionBlocking SimpleVaccine that uses a BOXDURABILITY profile with a
        WaningEffectBox.

        :param cb: The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` that will receive the risk-changing
        intervention.
        :param start_day: date upon which to change biting risk
        :param duration: duration of time for which the diagnostic exists and listens for the trigger, -1 is forever
        :param start_triggers_list: list of coordinator-type events for which the diagnostic listens to start running,
        cannot be empty
        :param stop_triggers_list: list of coordinator-type events which stop the diagnostic run (can be initiated by
        diagnostic)
        :param coordinator_name: name of the coordinator (used by the reporter)
        :param node_ids: list of node IDs which diagnostic will test, there will be a diagnostic per node
        :param node_property_restrictions: list of dictionaries of node properties to which the diagnostic will be
        restricted : [{ "NodeProperty1" : "PropertyValue1" }, {'NodeProperty2': "PropertyValue2"}, ...]
        :param ind_property_restrictions: used with Property_Restrictions_Within_Node. Format: list of dicts:
        [{ "IndividualProperty1" : "PropertyValue1" }, {'IndividualProperty2': "PropertyValue2"}, ...]
        :param target_demographic: to restrict monitoring by age, dict of {'agemin' : x, 'agemax' : y}.
        Default is targeting everyone.
        :param completion_event: coordinator-type event sent out at the end of the campaign
        :param demographic_coverage: This is the percentage of individuals that emit an event of interest
        and meet the demographic restrictions.
        :param target_property_name: The name of the individual property type whose value will be updated by the
        intervention.
        :param target_property_value: The user-defined value of the individual property that will be assigned to the
        individual.
        :param daily_prob: 	The probability that an individual will move to the Target_Property_Value
        :param max_duration: The maximum amount of time individuals have to move to a new group. This timing works in
        conjunction with Daily_Probability.
        :param revert: 	The number of days before an individual moves back to their original group.
        """
    if not start_triggers_list or not target_property_name or not target_property_value:
        raise ValueError("start_triggers_list, target_property_value, target_property_value all "
                         "need to be explicitly defined")

    property_value_changer = {
        "class": "PropertyValueChanger",
        "Target_Property_Key": target_property_name,
        "Target_Property_Value": target_property_value,
        "Daily_Probability": daily_prob,
        "Maximum_Duration": max_duration,
        "Revert": revert
    }

    # setting up the triggered coordinator
    if not node_ids:
        nodeset_config = {"class": "NodeSetAll"}
    else:
        nodeset_config = {"class": "NodeSetNodeList", "Node_List": node_ids}

    triggered_coordinator = {
        "class": "CampaignEvent",
        "Start_Day": start_day,
        "Nodeset_Config": nodeset_config,
        "Event_Coordinator_Config": {
            "class": "TriggeredEventCoordinator",
            "Coordinator_Name": coordinator_name,
            "Start_Trigger_Condition_List": start_triggers_list,
            "Stop_Trigger_Condition_List": stop_triggers_list,
            "Duration": duration,
            "Demographic_Coverage": demographic_coverage,
            "Node_Property_Restrictions": node_property_restrictions,
            "Property_Restrictions_Within_Node": ind_property_restrictions,
            "Intervention_Config": property_value_changer,
            "Completion_Event": completion_event
        }
    }

    # setting up to add demographic age restrictions to the event coordinator
    if target_demographic != 'Everyone':
        target_demographic = {
            "Target_Demographic": "ExplicitAgeRanges",  # Otherwise default is Everyone
            "Target_Age_Min": target_demographic['agemin'],
            "Target_Age_Max": target_demographic['agemax']
        }
    else:
        target_demographic = {"Target_Demographic": "Everyone"}

    triggered_coordinator["Event_Coordinator_Config"] = {**triggered_coordinator["Event_Coordinator_Config"],
                                                         **target_demographic}  # this concatenates the two dictionaries

    cb.add_event(triggered_coordinator)
