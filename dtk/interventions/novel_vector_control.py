from dtk.utils.Campaign.CampaignClass import *
import copy

def add_ATSB(cb, start_day: int=0, coverage: float=0.15, kill_cfg: any=None, duration: int=180, duration_std_dev: int=14,
             nodeIDs: list=None, node_property_restrictions: list=None):
    """
    Add an attractive targeted sugar bait (ATSB) intervention (**SugarTrap** class) using the
    **StandardInterventionDistributionEventCoordinator**.

    Args:
        cb: The The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` 
            containing the campaign configuration.
        start_day: The day on which to start distributing the intervention (**Start_Day** parameter).
        coverage: The proportion of the population that will receive the 
            intervention (**Demographic_Coverage** parameter).
        kill_cfg: Dictionary representing configuration for
         **Killing_Config_Per_Species** or a list of such dictionaries.
        duration: The length of time the ATSB is active for, independent
            of the waning profile of killing. This allows the node to prematurely
            get rid of the ATSB, much like **UsageDependentBednet** allows
            bednet users to get rid of good bednets. The expiration time
            is drawn from a Gaussian distribution with (mu, s) = (**duration**,
            **duration_std_dev**). 
        duration_std_dev: Width of the Gaussian distribution from which 
            the ATSB expiration time is drawn.
        nodeIDs: The list of nodes to apply this intervention to 
            (**Node_List** parameter). If not provided, set value of NodeSetAll.
        node_property_restrictions: The NodeProperty key:value pairs that 
            nodes must have to receive the intervention (**Node_Property_Restrictions** 
            parameter).
    Returns:
        None

    Example:
        ::

            cb = DTKConfigBuilder.from_defaults("MALARIA_SIM")
            # this is looking for the Killing_Config_Per_Species configuration OR a list of Killing_Config_Per_Species
            configurations which currently only exist is in Malaria-Ongoing or Killing_Config configuration
            kill_cfg = {"Species":"Arabiensis",
                        "Killing_Config":{
                            "class": "WaningEffectBoxExponential",
                            "Box_Duration": 100,
                            "Decay_Time_Constant": 150
                            "Initial_Effect":0.75
                            }
                        }
            or kill_cfg = [{"Species":"Arabiensis",
                            "Killing_Config":{
                                "class": "WaningEffectBoxExponential",
                                "Box_Duration": 100,
                                "Decay_Time_Constant": 150
                                "Initial_Effect":0.75
                            }
                        }, {"Species":"Gambiae",
                            "Killing_Config":{
                                "class": "WaningEffectConstant",
                                "Initial_Effect":0.75
                            }
                        }]
            or kill_cfg = {"Killing_Config":{
                                "Box_Duration": 3650,
                                "Initial_Effect": 0.93,
                                "class": "WaningEffectBox"}
                           }
            add_ATSB(cb, start=30, coverage=0.15, kill_cfg=kill_cfg,
                     duration=90, duration_std_dev=7,
                     node_property_restrictions= [{"Place": "Rural"}])
    """
    if node_property_restrictions is None:
        node_property_restrictions = []
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    cfg_species = [x for x in cb.get_param('Vector_Species_Names')
                   if cb.get_param('Vector_Species_Params')[x]['Vector_Sugar_Feeding_Frequency'] != 'VECTOR_SUGAR_FEEDING_NONE']
    if not cfg_species:
        raise ValueError("No species found without \"VECTOR_SUGAR_FEEDING_NONE\" setting. "
                         "Please review/update and try again.\n")

    atsb_master = WaningEffectBoxExponential(
        Initial_Effect=0.0337*coverage,
        Box_Duration=180,
        Decay_Time_Constant=30)

    # default killing cfg
    killing_cfg = [{'Species': sp,
                    'Killing_Config': atsb_master} for sp in cfg_species]

    # if user has specified a kill cfg, just use dicts rather than CampaignClasses.
    # If user doesn't specified a kill cfg, use the default killing cfg
    if kill_cfg:
        local_kill_cfg = copy.deepcopy(kill_cfg)
        # if user-inputed killing cfg is dict and species not specified, make a list
        if isinstance(local_kill_cfg, dict):
            if 'Killing_Config' not in local_kill_cfg:
                raise ValueError('Each config in SugarTrap killing config list must contain Killing_Config')
            else:
                local_kill_cfg['Killing_Config']['Initial_Effect'] *= coverage
            if 'Species' not in local_kill_cfg:
                killing_cfg = [{'Species': sp,
                                'Killing_Config': local_kill_cfg['Killing_Config']} for sp in cfg_species]
            else :
                killing_cfg = [local_kill_cfg]
        # if user-inputed killing cfg is list, check if each listed species is sugar-feeding species in config.
        elif isinstance(local_kill_cfg, list):
            for x in local_kill_cfg:
                if 'Species' not in x:
                    raise ValueError('Each config in SugarTrap killing config list must contain species name')
                elif 'Killing_Config' not in x:
                    raise ValueError('Each config in SugarTrap killing config list must contain Killing_Config')
                else:
                    x['Killing_Config']['Initial_Effect'] *= coverage
            listed_sp = [x['Species'] for x in local_kill_cfg]
            if any([x not in cfg_species for x in listed_sp]) :
                raise ValueError('A targeted SugarTrap species is not a sugar-feeding species in config')
            killing_cfg = [x for x in local_kill_cfg if x['Species'] in cfg_species]
        else :
            raise ValueError('Invalid SugarTrap killing config')

    atsb_config = SugarTrap(
        Cost_To_Consumer=3.75,
        Killing_Config_Per_Species=killing_cfg,
        Expiration_Distribution_Type="GAUSSIAN_DURATION",
        Expiration_Period_Mean=duration,
        Expiration_Period_Std_Dev=duration_std_dev
    )

    event = CampaignEvent(
        Start_Day=start_day,
        Nodeset_Config=node_cfg,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Intervention_Config=atsb_config,
            Intervention_Name="Attractive Toxic Sugar Bait",
            Demographic_Coverage=1,
            Node_Property_Restrictions=node_property_restrictions,
        )
    )
    cb.add_event(event)


def add_topical_repellent(config_builder, start_day, coverage_by_ages: list=None, cost: float=0,
                          blocking_initial: float=0.95, blocking_duration=0.3,
                          repetitions: int=1, tsteps_btwn: int=1, nodeIDs: list=None,
                          node_property_restrictions: list=None, ind_property_restrictions: list=None):
    """
    Add a topical insect repellent intervention (**SimpleIndividualRepellent** class)
    using the **StandardInterventionDistributionEventCoordinator** or a **BirthTriggeredIV**

    Args:
        config_builder: The The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` containing the 
            campaign configuration.
        start_day: The day on which to start distributing the intervention (**Start_Day** parameter).
        coverage_by_ages: A list of dictionaries defining the coverage per
            age group or birth-triggered intervention. For example,
            ``[{"birth":1, "duration":365, "coverage":0.9}, {"coverage":1, "age_min": 1, "age_max": 10},
            {"coverage":0.5,"age_min": 11,"age_max": 50}]`` If "birth":0, birth-triggered intervention will not be
            distributed.
        cost: The cost of each individual application (**Cost_To_Consumer** parameter).
        blocking_initial: The initial blocking effect of the repellent
            (**Initial_Effect** parameter).
        blocking_duration: The duration of the effectiveness (**Box_Duration**
            parameter with the **WaningEffectBox** class).
        repetitions: The number of times to repeat the intervention 
            (**Number_Repetitions** parameter).
        tsteps_btwn: The timesteps between repeated distributions
            (**Timesteps_Between_Repetitions** parameter).
        nodeIDs: The list of nodes to apply this intervention to (**Node_List**
            parameter). If not provided, set value of NodeSetAll.
        node_property_restrictions: The NodeProperty key:value pairs that
            nodes must have to receive the intervention
            (**Node_Property_Restrictions** parameter). In the format
            ``[{"Place":"RURAL"}, {"ByALake":"Yes}]``.
        ind_property_restrictions: The IndividualProperty key:value pairs
            that individuals must have to receive the intervention
            (**Property_Restrictions_Within_Node** parameter). In the format
            ``[{"BitingRisk":"High"}, {"IsCool":"Yes}]``.
    Returns:
        None

    Example:
        ::

            config_builder = DTKConfigBuilder.from_defaults("MALARIA_SIM")
            add_topical_repellent(config_builder, start=10,
                                  coverage_by_ages = [{"birth": 1,
                                                       "duration": -1,
                                                       "coverage": 0.75},
                                                       ],
                                  cost=1, initial_blocking=0.86, duration=0.3,
                                  repetitions=2, interval=1, nodeIDs=[1, 4, 19])
    """
    if not coverage_by_ages:
        ValueError('''Please define a list of coverages by age, in format of: 
        "[{"birth":1, "duration":365, "coverage":0.9}, {"coverage":1,"age_min": 1, "age_max": 10},
        {"coverage":0.5,"age_min": 11,"age_max": 50}] ")''' + "\n")
    if node_property_restrictions is None:
        node_property_restrictions = []
    if ind_property_restrictions is None:
        ind_property_restrictions = []
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    repellent = SimpleIndividualRepellent(
        Cost_To_Consumer=cost,
        Event_Name="Individual Repellent",
        Blocking_Config=WaningEffectBox(
            Initial_Effect=blocking_initial,
            Box_Duration=blocking_duration
        )
    )

    for coverage_by_age in coverage_by_ages:
        if 'birth' in coverage_by_age.keys():
            if coverage_by_age['birth']:
                repellent_event = CampaignEvent(
                    Start_Day=start_day,
                    Nodeset_Config=node_cfg,
                    Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                        Intervention_Config=BirthTriggeredIV(
                            Node_Property_Restrictions=node_property_restrictions,
                            Property_Restrictions_Within_Node=ind_property_restrictions,
                            Duration=coverage_by_age.get('duration', -1),
                            Demographic_Coverage=coverage_by_age["coverage"],
                            Actual_IndividualIntervention_Config=repellent
                        )
                    )
                )
                config_builder.add_event(repellent_event)
        else:
            repellent_event = CampaignEvent(
                Start_Day=start_day,
                Nodeset_Config=node_cfg,
                Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                    Node_Property_Restrictions=node_property_restrictions,
                    Property_Restrictions_Within_Node=ind_property_restrictions,
                    Target_Residents_Only=0,
                    Demographic_Coverage=coverage_by_age["coverage"],
                    Target_Demographic="ExplicitAgeRanges",
                    Target_Age_Min=coverage_by_age["age_min"],
                    Target_Age_Max=coverage_by_age["age_max"],
                    Number_Repetitions=repetitions,
                    Timesteps_Between_Repetitions=tsteps_btwn,
                    Intervention_Config=repellent
                )
            )
            config_builder.add_event(repellent_event)


def add_ors_node(config_builder, start_day: int=0, coverage: float=1, killing_initial: float=0.95,
                 killing_decay: int=100, cost: float=1, nodeIDs: list=None, node_property_restrictions: list=None):
    """
    Add an outdoor residential spraying intervention (**SpaceSpraying** class) using
    **StandardInterventionDistributionEventCoordinator**

    Args:
        config_builder: The The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` 
            containing the campaign configuration.
        start_day: The day on which to start distributing the intervention
            (**Start_Day** parameter).
        coverage: multiplication coefficient reducing the **Initial_Effect** of **Killing_Config**
        killing_initial:  The initial killing effect of the outdoor spraying
            (**Initial_Effect** parameter).
        killing_decay: The exponential decay length, in days (**Decay_Time_Constant**
            in **Killing_Config**).
        cost: The cost of each individual application (**Cost_To_Consumer** 
            parameter).
        nodeIDs: The list of nodes to apply this intervention to (**Node_List** 
            parameter). If not provided, set value of NodeSetAll.
        node_property_restrictions: The NodeProperty key:value pairs that
            nodes must have to receive the intervention
            (**Node_Property_Restrictions** parameter). In the format
            ``[{"Place":"RURAL"}, {"ByALake":"Yes}]``.

    Returns:
        None

    Example:
        ::

            config_builder = DTKConfigBuilder.from_defaults("MALARIA_SIM")
            add_ors_node(config_builder, start_day=200, coverage=0.8,
                         killing_initial=0.85, killing_duration=45, cost=1,
                         nodeIDs=[1, 4, 7])
    """
    if not node_property_restrictions:
        node_property_restrictions = []
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    ors_event = CampaignEvent(
                Event_Name="Outdoor Residual Spray",
                Nodeset_Config=node_cfg,
                Start_Day=start_day,
                Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
                    Node_Property_Restrictions=node_property_restrictions,
                    Intervention_Config=SpaceSpraying(
                            Spray_Kill_Target=SpaceSpraying_Spray_Kill_Target_Enum.SpaceSpray_FemalesAndMales,
                            Habitat_Target=SpaceSpraying_Habitat_Target_Enum.ALL_HABITATS,
                            Cost_To_Consumer=cost,
                            Killing_Config=WaningEffectExponential(
                                Initial_Effect=killing_initial * coverage,
                                Decay_Time_Constant=killing_decay
                            ),
                            Reduction_Config=WaningEffectBox(
                                Initial_Effect=0,
                                Box_Duration=365
                            )
                        )
                )
    )
    config_builder.add_event(ors_event)


def add_larvicides(config_builder, start_day: int=0, habitat_target: str="ALL_HABITATS", coverage: float=1,
                   killing_initial: float=1, killing_duration: int=100, killing_decay: int=150,
                   blocking_initial: float=1, blocking_duration: int=100, blocking_decay: int=150, cost: float=1,
                   nodeIDs: list=None, node_property_restrictions: list=None):
    """
    Add a mosquito larvicide intervention to the campaign using the
    **Larvicides** class, please note the Killing and Blocking configurations are using
    **WaningEffectBoxExponential** class.


    Args:
        config_builder: The :py:class:`DTKConfigBuilder
            <dtk.utils.core.DTKConfigBuilder>` containing the campaign
            configuration.
        start_day: The day on which to start distributing the larvicide
            (**Start_Day** parameter).
        habitat_target: The larval habitat type targeted by the larvicide, needs to be all upper_case
            (**Habitat_Target** parameter).
        coverage: The multiplication coefficient reducing the **Initial_Effect** of **Killing_Config**
            and the **Blocking_Config**
        killing_initial: The initial larval killing efficacy (**Initial_Effect** in
            **Killing_Config**).
        killing_duration: The box duration of the effect in days (**Box_Duration**
            in **Killing_Config**).
        killing_decay: The exponential decay length, in days (**Decay_Time_Constant**
            in **Killing_Config**).
        blocking_initial: The initial larval killing efficacy (**Initial_Effect** in
            **Blocking_Config**).
        blocking_duration: The box duration of the effect in days (**Box_Duration**
            in **Blocking_Config**).
        blocking_decay: The exponential decay length, in days (**Decay_Time_Constant**
            in **Blocking_Config**).
        cost: The cost of each individual application (**Cost_To_Consumer**
            parameter).
        nodeIDs: The list of nodes to apply this intervention to (**Node_List**
            parameter). If not provided, set value of NodeSetAll.
        node_property_restrictions: The NodeProperty key:value pairs that
            nodes must have to receive the intervention
            (**Node_Property_Restrictions** parameter). In the format
            ``[{"Place":"RURAL"}, {"ByALake":"Yes}]``.
    Returns:
        None

    Example:
        ::
            config_builder = DTKConfigBuilder.from_defaults("MALARIA_SIM")
            add_larvicides(config_builder, start=725, killing_initial=0.75,
            blocking_initial=0.9, habitat_target="HUMAN_POPULATION",
            nodeIDs=[2, 5, 7])
    """

    if node_property_restrictions is None:
        node_property_restrictions = []
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    event = CampaignEvent(
        Start_Day=start_day,
        Nodeset_Config=node_cfg,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Node_Property_Restrictions=node_property_restrictions,
            Intervention_Config=Larvicides(
                        Habitat_Target=Larvicides_Habitat_Target_Enum[habitat_target],
                        Blocking_Config=WaningEffectBoxExponential(
                            Box_Duration=blocking_duration,
                            Decay_Time_Constant=blocking_decay,
                            Initial_Effect=blocking_initial * coverage
                        ),
                        Cost_To_Consumer=cost,
                        Killing_Config=WaningEffectBoxExponential(
                            Box_Duration=killing_duration,
                            Decay_Time_Constant=killing_decay,
                            Initial_Effect=killing_initial * coverage
                        )
                    )
        )
    )
    config_builder.add_event(event)


def add_eave_tubes(config_builder, start_day: int=0, coverage: float=1, killing_initial: float=1.0,
                   killing_decay: int=180, blocking_initial: float=1.0, blocking_decay: int=730,
                   outdoor_killing_discount: float=0.3, cost: float=0,
                   nodeIDs: list=None, node_property_restrictions: list=None, ind_property_restrictions: list=None):
    """
    Add insecticidal tubes to the eaves of houses (**IRSHousingModification** intervention class) and
    an outdoor residential spraying intervention (**SpaceSpraying** class) using
    **StandardInterventionDistributionEventCoordinator** (see add_ors_node)

    Args:
        config_builder: The The :py:class:`DTKConfigBuilder <dtk.utils.core.DTKConfigBuilder>` 
            containing the campaign configuration.
        start_day: The day on which to start distributing the intervention
            (**Start_Day** parameter).
        coverage: The proportion of the population that will receive the intervention 
            (**Demographic_Coverage** parameter).
        killing_initial:  The initial killing effect of the eave tubes
            (**Initial_Effect** parameter in **Killing_Config**).
        killing_decay: The exponential decay constant of the effectiveness
            (**Decay_Time_Constant** parameter with the **WaningEffectExponential** 
            class).
        blocking_initial:  The initial blocking effect of the eave tubes
            (**Initial_Effect** parameter in **Blocking_Config**).
        blocking_decay: The exponential decay constant of the effectiveness
            (**Decay_Time_Constant** parameter with the **WaningEffectExponential** class).
        outdoor_killing_discount: The value to differentially scale initial 
            killing effect for outdoor vectors vs. indoor vectors.
        cost: The cost of each individual application (**Cost_To_Consumer** parameter).
        nodeIDs: The list of nodes to apply this intervention to (**Node_List** 
            parameter). If not provided, set value of NodeSetAll.
        node_property_restrictions: The NodeProperty key:value pairs that
            nodes must have to receive the intervention
            (**Node_Property_Restrictions** parameter). In the format
            ``[{"Place":"RURAL"}, {"ByALake":"Yes}]``.
        ind_property_restrictions: The IndividualProperty key:value pairs
            that individuals must have to receive the intervention
            (**Property_Restrictions_Within_Node** parameter). In the format
            ``[{"BitingRisk":"High"}, {"IsCool":"Yes}]``.

    Returns:
        None

    Example:
        ::

            config_builder = DTKConfigBuilder.from_defaults('MALARIA_SIM')
            add_eave_tubes(config_builder, start_day=37, coverage=0.65,
                           killing_initial=0.85, killing_decay=90,
                           blocking_initial=0.95, blocking_decay=365,
                           outdoor_killing_discount=0.3, cost=1,
                           nodeIDs=[33, 56, 7])
    """
    if node_property_restrictions is None:
        node_property_restrictions = []
    if ind_property_restrictions is None:
        ind_property_restrictions = []
    if nodeIDs:
        node_cfg = NodeSetNodeList(Node_List=nodeIDs)
    else:
        node_cfg = NodeSetAll()

    indoor_event = CampaignEvent(
        Start_Day=start_day,
        Nodeset_Config=node_cfg,
        Event_Coordinator_Config=StandardInterventionDistributionEventCoordinator(
            Node_Property_Restrictions=node_property_restrictions,
            Property_Restrictions_Within_Node=ind_property_restrictions,
            Demographic_Coverage=coverage,
            Intervention_Config=IRSHousingModification(
                Cost_To_Consumer=cost,
                Killing_Config=WaningEffectExponential(
                    Initial_Effect=killing_initial,
                    Decay_Time_Constant=killing_decay
                ),
                Blocking_Config=WaningEffectExponential(
                    Initial_Effect=blocking_initial,
                    Decay_Time_Constant=blocking_decay
                )
            )
        )
    )
    config_builder.add_event(indoor_event)

    add_ors_node(config_builder, start_day=start_day, coverage=coverage,
                 killing_initial=killing_initial*outdoor_killing_discount,
                 killing_decay=killing_decay, cost=cost,
                 nodeIDs=nodeIDs, node_property_restrictions=node_property_restrictions)
