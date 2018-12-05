
def add_mosquito_release(cb, start_day, species, number=100, repetitions=-1, tsteps_btwn=365, gender='VECTOR_FEMALE',
                         released_genome=[['X', 'X']], released_wolbachia="VECTOR_WOLBACHIA_FREE",
                         nodes={"class": "NodeSetAll"}):
    """
    Function to add recurring introduction of new new vectors

    :param cb: Configuration builder holding the interventions
    :param repetitions: Number of repetitions
    :param tsteps_btwn:  Timesteps between repetitions
    :param start_day: Start day for the first release
    """
    release_event = { "class" : "CampaignEvent",
                      "Event_Name" : "Mosquito Release",
                        "Start_Day": start_day,
                        "Event_Coordinator_Config": {
                            "class": "StandardInterventionDistributionEventCoordinator",
                            "Number_Distributions": -1,
                            "Number_Repetitions": repetitions,
                            "Timesteps_Between_Repetitions": tsteps_btwn,
                            "Target_Demographic": "Everyone",
                            "Intervention_Config": {        
                                    "Released_Number": number, 
                                    "Released_Species": species, 
                                    "Released_Gender": gender,
                                    "Released_Wolbachia": "VECTOR_WOLBACHIA_FREE",
                                    "Released_Genome": released_genome,
                                    "class": "MosquitoRelease"
                                } 
                            },
                        "Nodeset_Config": nodes
                        }

    cb.add_event(release_event)
