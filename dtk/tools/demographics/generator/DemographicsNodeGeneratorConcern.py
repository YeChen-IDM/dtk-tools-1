import abc
import os
from copy import deepcopy
from typing import List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from scipy import integrate as spi

from dtk.generic.demographics import distribution_types
from dtk.tools.demographics.Node import Node


class DemographicsNodeGeneratorConcern:
    """
    This is a generator that hooks into the generation of nodes as well as overall defaults since these
    two are often linked. These classes are meant to encapsulate a set of Demographics configuration
    that may span IndividualAttributes, NodeAttributes, and per node properties that are best expressed
    as a single unit.

    As for execution order, update_defaults will be called during the building of the defaults
    objects before node generation and therefore update_node
    """

    @abc.abstractmethod
    def update_node(self, defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict):
        """
        This is the function that is called for each node as its demographics properties are generated

        Args:
            defaults: Our current set of demographics defaults. This is a passed as a dictionary and from the top level
            ie, we should have the 'NodeAttributes' and 'IndividualAttributes' after the generation of most defaults
            node: The actual dtk.node object. This has details for the node like lat, long, population, etc
            node_attributes: The current node's attributes
            node_individual_attributes: The current node's individual attributes

        Returns:
            None. Updates the objects passed in depending on the concern
        """
        pass

    @abc.abstractmethod
    def update_defaults(self, defaults: dict) -> dict:
        """
        Called before update_node as part of building the demographics default configuration
        Args:
            defaults: The defaults to update

        Returns:
            None. Update the object defaults depending on the concern
        """
        pass


class DemographicsNodeGeneratorConcernChain(DemographicsNodeGeneratorConcern):
    """
        Allows chaining of DemographicsNodeGeneratorConcern. Most configuration will require some form of chaining
        to be a full demographics file since each concern is meant to handle one small piece


        """

    def __init__(self, *args: DemographicsNodeGeneratorConcern):
        """
                Chains a set of DemographicsNodeGeneratorConcern together. The concerns should be passed in order
                as arguments to the init function
                Args:
                    *args: A list of DemographicsNodeGeneratorConcern passed as arguments in execution order

                Examples:
                    ```

                    chain_links = [
                        DefaultIndividualAttributesConcern(), # Start with our default setup
                        WorldBankBirthRateConcern(), # and then set our birth rate to data from WorldBank
                        EquilibriumAgeDistributionConcern() # and then set our distribution to be in Equilibrium
                    ]
                    chain = DemographicsNodeGeneratorConcernChain(*chain_links)

                    ```
                """
        if any([not isinstance(arg, DemographicsNodeGeneratorConcern) for arg in args]):
            raise ValueError("All concerns must be of type DemographicsNodeGeneratorConcern")
        self.concerns: List[DemographicsNodeGeneratorConcern] = list(args)

    def update_node(self, defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict):
        """
        Runs all the concerns in order for node's data

        Args:
            defaults: Our current set of demographics defaults. This is a passed as a dictionary and from the top level
            ie, we should have the 'NodeAttributes' and 'IndividualAttributes' after the generation of most defaults
            node: The actual dtk.node object. This has details for the node like lat, long, population, etc
            node_attributes: The current node's attributes
            node_individual_attributes: The current node's individual attributes

        Returns:

        """
        for concern in self.concerns:
            concern.update_node(defaults, node, node_attributes, node_individual_attributes)

    def update_defaults(self, defaults: dict) -> dict:
        """
        Called before update_node as part of building the demographics default configuration
        Args:
            defaults: The defaults to update

        Returns:
            None. Update the object defaults depending on the concern
        """
        for concern in self.concerns:
            concern.update_defaults(defaults)
        return defaults


class DefaultsDictionaryNodeGeneratorConcern(DemographicsNodeGeneratorConcern):

    def __init__(self, individual_attributes: dict, node_attributes: Optional[dict] = None,
                 update_node: Optional[Callable[[DemographicsNodeGeneratorConcern, dict, Node, dict, dict], None]] = None):
        """
        DictionaryDemographicsNodeGeneratorConcern allows us to easily set our defaults for both individual_attributes
        and node_attributes. You can also pass a custom update_node function since the default will just continue

        Args:

            individual_attributes: Dictionary representing the default IndividualAttributes
            node_attributes: Dictionary representing the default IndividualAttributes
            update_node: Optional function that will be called per node.
        """
        self.individual_attributes = individual_attributes
        if node_attributes is None:
            self.node_attributes = {
                "Urban": 0,
                "AbovePoverty": 0.5,
                "Region": 1,
                "Seaport": 0,
                "Airport": 0,
                "Altitude": 0
            }
        else:
            self.node_attributes = node_attributes

        if update_node:
            self.update_node = update_node

    def update_node(self, defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict):
        """
        Default implementation here is to pass without updating anything
        Args:
            defaults: Our current set of demographics defaults. This is a passed as a dictionary and from the top level
            ie, we should have the 'NodeAttributes' and 'IndividualAttributes' after the generation of most defaults
            node: The actual dtk.node object. This has details for the node like lat, long, population, etc
            node_attributes: The current node's attributes
            node_individual_attributes: The current node's individual attributes

        Returns:

        """
        pass

    def update_defaults(self, defaults: dict):
        """
        Updates the defaults with values the user has supplied for individual_attributes and node_attributes
        Args:
            defaults:

        Returns:

        """
        defaults["IndividualAttributes"] = self.individual_attributes
        defaults["NodeAttributes"] = self.node_attributes


class DefaultIndividualAttributesConcern(DefaultsDictionaryNodeGeneratorConcern):
    def __init__(self, prevalence_flag=distribution_types["EXPONENTIAL_DISTRIBUTION"], prevalence1=0.13, prevalence2=0.15, mortality_scale_factor=2.74e-06,
                 population_removal_rate=23.0):
        """

        A most common/catch call default individual and node properties setup.

        Can easily be subclassed to change just default individual_attributes or even to just modify the
         MortalityDistribution through get_mortality_distribution_config

        Args:
            prevalence_flag: Set the prevalence_flag for the default config
            prevalence1:  Set value to be used for PrevalenceDistribution1
            prevalence2:  Set value to be used for PrevalenceDistribution2
            mortality_scale_factor:  Set our mortality_scale_factor
            population_removal_rate:  Set our population removal rate

        See Also:
            DefaultIndividualAttributesConcern.get_individual_attributes
        """
        mod_mortality = self.get_mortality_distribution_config(mortality_scale_factor, population_removal_rate)
        individual_attributes = self.get_individual_attributes(mod_mortality, prevalence1, prevalence2, prevalence_flag)
        super().__init__(individual_attributes)

    @staticmethod
    def get_individual_attributes(mortality_distribution_config: dict, prevalence1: float, prevalence2: float, prevalence_flag: int):
        """
        Called to produce the default minimum set of defaults using the specified parameters
        Args:
            mortality_distribution_config: MortalityDistribution config
            prevalence1:  Set value to be used for PrevalenceDistribution1
            prevalence2:  Set value to be used for PrevalenceDistribution2:
            prevalence_flag: Set the prevalence_flag

        Returns:

        """
        return {
            "MortalityDistribution": mortality_distribution_config,
            "RiskDistribution1": 1,
            "PrevalenceDistributionFlag": prevalence_flag,
            "PrevalenceDistribution1": prevalence1,
            "PrevalenceDistribution2": prevalence2,
            "RiskDistributionFlag": 0,
            "RiskDistribution2": 0,
            "MigrationHeterogeneityDistribution1": 1,
            "ImmunityDistributionFlag": 0,
            "MigrationHeterogeneityDistributionFlag": 0,
            "ImmunityDistribution1": 1,
            "MigrationHeterogeneityDistribution2": 0,
            "ImmunityDistribution2": 0
        }

    @staticmethod
    def get_mortality_distribution_config(mortality_scale_factor, population_removal_rate):
        """
        Builds our default MortalityDistribution config using specifed inputs

        Args:
            mortality_scale_factor:
            population_removal_rate:

        Returns:

        """
        mortality_distribution_config = {
            "NumDistributionAxes": 2,
            "AxisNames": ["gender", "age"],
            "AxisUnits": ["male=0,female=1", "years"],
            "AxisScaleFactors": [1, 365],
            "NumPopulationGroups": [2, 1],
            "PopulationGroups": [
                [0, 1],
                [0]
            ],
            "ResultUnits": "annual deaths per 1000 individuals",
            "ResultScaleFactor": mortality_scale_factor,
            "ResultValues": [
                [population_removal_rate],
                [population_removal_rate]
            ]
        }
        return mortality_distribution_config


class EquilibriumAgeDistributionConcern(DemographicsNodeGeneratorConcern):

    def __init__(self, mortality_scale=2.74e-06, max_age: float = 100.0,
                 default_birth_rate: float = 23.0,
                 max_years: int = 120,
                 prevalence_flag: int = distribution_types["UNIFORM_DISTRIBUTION"],
                 prevalence1: float = 0.13,
                 prevalence2: float = 0.15):
        """
        Creates a config who's initial age distribution is already at equilibrium, allowing you to  burn-ins for
        immunity establishment only.

        Adapted from code originally developed by Kurt Frey

        Notes:
            You must set your config file to
            ```
                Birth_Rate_Dependence="FIXED_BIRTH_RATE" and
                Age_Initialization_Distribution_Type= "DISTRIBUTION_COMPLEX"
            ```
            for this to work.


            Additional improvements could be made to
            1. Supported   age-specific mortality rates
            2. Possibly using the current config to get mortality_scale, etc
            3. Move default_birth_rate storage. Maybe we can move it to defaults node attributes?
        Args:
            mortality_scale: Mortality scale
            max_age: age past which you want everyone to die. In the current implementation you will get *some* mass
                in this bin, but only the amount specified in the equilibrium distribution.
            default_birth_rate: The default birth rate
            max_years: How long to run the equilibrium to run. Default should be sufficient in almost all cases
            prevalence1:  Set value to be used for PrevalenceDistribution1
            prevalence2:  Set value to be used for PrevalenceDistribution2:
            prevalence_flag: Set the prevalence_flag
        """
        self.max_year = max_years
        self.mortality_scale = mortality_scale
        self.max_age = max_age
        self.default_birth_rate = default_birth_rate
        self.prevalence_flag = prevalence_flag
        self.prevalence1 = prevalence1
        self.prevalence2 = prevalence2

        super().__init__()

    def get_node_distribution(self, node_birth_rate: float, overall_birth_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """
                Calculates the EquilibriumAgeDistribution

                Args:
                    node_birth_rate: population birth rate, in units of births/node/day. We take this per node so it
                     can change within our generators

                    overall_birth_rate: annual deaths per 1000, set to mirror birth rates.
                Returns:
                    resval (a list of age bins in days from 0 to self.max_year),
                    and distval (a list of the cumulative proportion of the population in each age bin of resval)
                """
        # define daily mortality probability by age group
        age_vec = [365 * 0, 365 * (self.max_age - 0.001), 365 * self.max_age]
        mort_vec = [self.mortality_scale * overall_birth_rate, self.mortality_scale * overall_birth_rate, 1]

        day_to_year = 365

        # add bounds around age_vec and mort_vec
        mvec_x = [-1] + age_vec + [self.max_year * day_to_year + 1]
        mvec_y = [mort_vec[0]] + mort_vec + [mort_vec[-1]]

        # cumulative monthly survival probabilities
        m_x = np.arange(0, self.max_year * day_to_year, 30)
        mval = (1.0 - np.interp(m_x, xp=mvec_x, fp=mvec_y)) ** 30

        # create normalized population pyramid
        popvec = node_birth_rate * np.cumprod(mval)
        tpop = np.trapz(popvec, x=m_x)
        npvec = popvec / tpop
        # what proportion of  people are in the simulation up to bin i
        cpvec = spi.cumtrapz(npvec, x=m_x, initial=0)

        resval = np.around(np.linspace(0, self.max_year * day_to_year))
        distval = np.interp(resval, xp=m_x, fp=cpvec)

        return resval, distval

    def update_node(self, defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict):
        """

        Currently, to update a node, we need the birth rate for all nodes as a whole
        Args:
            defaults:
            node:
            node_attributes:
            node_individual_attributes:

        Returns:

        """
        # Todo lookup in defaults
        if "DefaultBirthRate" not in node_attributes:
            raise ValueError("Cannot find the birth rate for current node")

        resval, distval = self.get_node_distribution(node_attributes["BirthRate"],
                                                     node_attributes["BirthRate"])

        age_dist_config = {
            "DistributionValues": [distval.tolist()],
            "ResultScaleFactor": 1,
            "ResultValues":
                [
                    resval.tolist()
                ]
        }
        node_individual_attributes.update({"AgeDistribution": age_dist_config})

    def update_defaults(self, defaults: dict):
        mod_mortality = DefaultIndividualAttributesConcern.get_mortality_distribution_config(self.mortality_scale, self.default_birth_rate)
        defaults["IndividualAttributes"] = DefaultIndividualAttributesConcern. \
            get_individual_attributes(mod_mortality, self.prevalence1, self.prevalence2, self.prevalence_flag)


class WorldBankBirthRateNodeConcern(DemographicsNodeGeneratorConcern):
    def __init__(self, country='Sub-Saharan Africa', birthrate_year: int = 2016,
                 update_mortality_distribution: bool = True,
                 population_reference_file: Optional[str] = None):
        """
        This concern updates the birth rates based on World Bank's birth rate data

        Notes:
            You should generate your MortalityDistribution before this Concern to have it update
            the MortalityDistribution to birth rates and also set update_mortality_distribution to True
        Args:
            update_mortality_distribution: If set to true, we will attempt
            country: Country to load birth rate data from
            birthrate_year: Year to load the birth rate data from
            population_reference_file: Alternative source for WorldBank data. This needs to be in the same format as
            world bank with Country, Yar, and then birth rate specified
        """
        self.default_country = country.replace('_', ' ')
        self.birthrate_year = str(birthrate_year)
        if population_reference_file is None:
            population_reference_file = os.path.join(os.path.dirname(__file__),
                                                     'WB_crude_birthrate_by_year_and_country.csv')
        df = pd.read_csv(population_reference_file, encoding='latin1')
        self.birthrate_df = df
        self.update_mortality_distribution = update_mortality_distribution

        # Ensure the country exists
        if len((df[df['Country Name'] == self.default_country])) == 0:
            msg_prefix = f"Cannot locate country {country}"
            if self.default_country != country:
                msg_prefix += f" as {self.default_country}"
            raise ValueError(f"{msg_prefix}. Possible values are{', '.join(df['Country Name'].unique())}")

        self.default_birth_rate = df[df['Country Name'] == self.default_country][self.birthrate_year].values[0]

    @staticmethod
    def get_mortality_distribution_from_defaults(defaults, birth_rate):
        """
        Try to located the default MortalityDistribution config, This is used when we have demographics nodes
        that span countries. We want to then set any nodes that are not the primary country to
        have their own MortalityDistribution
        Args:
            defaults: default dictionary to be used to find the default MortalityDistribution
            birth_rate:

        Returns:

        """
        if any(["IndividualAttributes" not in defaults, "MortalityDistribution" in defaults["IndividualAttributes"]]):
            raise ValueError("Could not locate the mortality distribution config from Demographics defaults")
        config = deepcopy(defaults["IndividualAttributes"]["MortalityDistribution"])
        # TODO validate against different MortalityDistribution's
        config["ResultValues"] = [
                [birth_rate],
                [birth_rate]
            ]
        return config

    def update_node(self, defaults: dict, node: Node, node_attributes: dict, node_individual_attributes: dict):
        """
        Updates each nodes birth rate to that of world banks
        Args:
            defaults:
            node:
            node_attributes:
            node_individual_attributes:

        Returns:

        """
        if "Country" in node_attributes:
            country_condition = self.birthrate_df['Country Name'] == node_attributes["Country"]
            birth_rate = self.birthrate_df[country_condition][self.birthrate_year].values[0]

            # Set the per country dist
            node_individual_attributes.update({
                "MortalityDistribution": self.get_mortality_distribution_from_defaults(defaults, birth_rate)
            })
        else:
            birth_rate = self.default_birth_rate
        per_node_birth_rate = (float(node.pop) / 1000) * birth_rate / 365.0
        node_attributes['BirthRate'] = per_node_birth_rate
        node_attributes['DefaultBirthRate'] = birth_rate

    def update_defaults(self, defaults: dict):
        """
        Update the defaults config to contain the default birth rates for specified country
        Args:
            defaults: defaults dictionary to update

        Returns:

        """
        ia = defaults['IndividualAttributes']
        if all([self.update_mortality_distribution] +
               ["MortalityDistribution" in ia, "ResultValues" in ia["MortalityDistribution"]]):
            ia["MortalityDistribution"]["ResultValues"] = [
                [self.default_birth_rate],
                [self.default_birth_rate]
            ]


class DefaultWorldBankEquilibriumConcern(DemographicsNodeGeneratorConcernChain):
    def __init__(self, prevalence_flag: int = distribution_types["UNIFORM_DISTRIBUTION"], prevalence1: float = 0.13,
                 prevalence2: float = 0.15, country='Sub-Saharan Africa', birthrate_year: int = 2016,
                 max_years: int = 120, mortality_scale=2.74e-06, max_age: float = 100.0,
                 extra_concerns: Optional[List[DemographicsNodeGeneratorConcern]] = None):
        """
        This is a convenience class to allow generating demographics using a common chain of concerns. First,
        we start with our DefaultIndividualAttributesConcern. This takes care of most of the defaults. Then we
        use the WorldBankBirthRateNodeConcern to add world bank birth rates to nodes and defaults
        Lastly, we use EquilibriumAgeDistributionConcern, and the birth rate our EquilibriumAgeDistributionConcern,
        to create a population in equilibrium from the start.
        Args:
            max_years: How long to run the equilibrium to run. Default should be sufficient in almost all cases
            prevalence1:  Set value to be used for PrevalenceDistribution1
            prevalence2:  Set value to be used for PrevalenceDistribution2:
            prevalence_flag: Set the prevalence_flag
            country: Country to load birth rate data from World Bank
            birthrate_year: Year for birth rate data
            mortality_scale: Set our mortality_scale_factor
            max_age: age past which you want everyone to die. In the current implementation you will get *some* mass
                in this bin, but only the amount specified in the equilibrium distribution.
            extra_concerns: Any other concerns to run AFTER the default 3
        """
        wb_concern = WorldBankBirthRateNodeConcern(country=country, birthrate_year=birthrate_year)
        concerns = [DefaultIndividualAttributesConcern(prevalence_flag=prevalence_flag, prevalence1=prevalence1,
                                                       prevalence2=prevalence2),
                    wb_concern,
                    EquilibriumAgeDistributionConcern(mortality_scale=mortality_scale, max_age=max_age,
                                                      max_years=max_years,
                                                      default_birth_rate=wb_concern.default_birth_rate)
                    ]
        if extra_concerns:
            concerns += extra_concerns
        super().__init__(*concerns)
