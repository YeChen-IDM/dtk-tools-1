import csv
import json
import os
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd
import scipy.integrate as sp

from dtk.tools.demographics.Node import Node, nodeid_from_lat_lon
from simtools.Utilities.General import init_logging

logger = init_logging('DemographicsGenerator')


class InvalidResolution(BaseException):
    pass


class DemographicsType(Enum):
    STATIC = 'static'

    def __str__(self):
        return str(self.value)


class DemographicsGenerator:
    """
    Generates demographics file based on population input file.
    The population input file is csv with structure

    node_label*, lat, lon, pop*

    *-ed columns are optional
    """

    # mapping of requested arcsecond resolution -> demographic metadata arcsecond resolution.
    # All Hash values must be integers.
    CUSTOM_RESOLUTION = 'custom'
    DEFAULT_RESOLUTION = 30
    VALID_RESOLUTIONS = {
        30: 30,
        250: 250,
        CUSTOM_RESOLUTION: 30
    }

    def __init__(self, nodes, demographics_type: DemographicsType = DemographicsType.STATIC,
                 res_in_arcsec=CUSTOM_RESOLUTION, update_demographics: Optional[Callable] = None,
                 default_population: int = 1000,
                 country='Sub-Saharan Africa', birthrate_year: int = 2016, prevalence_flag: int = 1,
                 prevalence1: float = 0.13,
                 prevalence2: float = 0.15, population_reference_file: Optional[str] = None, **kwargs):
        """
        Initialize the SpatialManager
        Args:
            nodes: list of nodes
            demographics_type: could be 'static', 'growing' or a different type; currently only static is
                implemented in generate_nodes(self)
            res_in_arcsec: Simulation grid resolution
            update_demographics: provide the user with a chance to update the demographics file before it's written
                via a user-defined function; (e.g. scale larval habitats based on initial population per node in the
                demographics file) see generate_demographics(self)
            default_population: default population for all nodes
            country: Choose country from World Population list of crude birth rates for birth rate
            birthrate_year: Choose year to which birthrate is set
            prevalence_flag: Determines prevalence distribution parameters. Read EMOD documentation.
            prevalence1: The first value in the distribution, the meaning of which depends upon the value set in
                PrevalenceDistributionFlag. Read EMOD documentation.
            prevalence2: The second value in the distribution, the meaning of which depends upon the value set in
                PrevalenceDistributionFlag. Read EMOD documentation.
            population_reference_file: World population reference file. If not specified, we use our default input
            **kwargs: any keyword arguments to be passed to the update_demographics function
        """
        self.nodes = nodes
        self.demographics_type = demographics_type
        self.set_resolution(res_in_arcsec)
        self.update_demographics = update_demographics
        self.kwargs = kwargs

        # demographics data dictionary (working DTK demographics file when dumped as json)
        self.demographics = None
        self.default_pop = default_population
        self.default_country = country.replace('_', ' ')
        self.year = str(birthrate_year)
        if population_reference_file is None:
            population_reference_file = os.path.join(os.path.dirname(__file__),
                                                     'WB_crude_birthrate_by_year_and_country.csv')
        df = pd.read_csv(population_reference_file, encoding='latin1')
        self.birthrate_df = df

        # Ensure the country exists
        if len((df[df['Country Name'] == self.default_country])) == 0:
            raise ValueError(f"Cannot locate country name as {country} as {self.default_country}. Possible values are"
                             f"{', '.join(df['Country Name'].unique())}")

        self.default_birth_rate = df[df['Country Name'] == self.default_country][self.year].values[0]
        self.mort_scale_factor = 2.74e-06
        self.prevalence_flag = prevalence_flag
        self.prevalence1 = prevalence1
        self.prevalence2 = prevalence2

    @staticmethod
    def arcsec_to_deg(arcsec: float) -> float:
        """
        Arc second to degrees
        Args:
            arcsec: arcsecond as float

        Returns:
            arc second converted to degrees
        """
        return arcsec / 3600.0

    @classmethod
    def from_grid_file(cls, population_input_file: str,
                       demographics_filename: str,
                       res_in_arcsec=CUSTOM_RESOLUTION,
                       update_demographics=None, default_pop=1000,
                       latitude_column_name: str = 'lat', longitude_column_name: str = 'lon',
                       population_column_name: str = 'pop'):
        """
        Build demorgraphics from a grid file and a demographics file using
        world pop data

        Args:
            population_input_file: CSV population file. Must contain all the columns specified by latitude_column_name,
                longitude_column_name, and population_column_name
            demographics_filename: Demographics filename
            res_in_arcsec:
            update_demographics:
            default_pop:
            latitude_column_name:
            longitude_column_name:
            population_column_name:

        Returns:

        """
        # demographics type- currently only static is supported so hard code it
        demographics_type: DemographicsType = DemographicsType.STATIC
        nodes_list = list()
        with open(population_input_file, 'r') as pop_csv:
            reader = csv.DictReader(pop_csv)
            for row in reader:
                # Latitude
                if latitude_column_name not in row:
                    raise ValueError(f'Column {latitude_column_name} is required in input population file.')
                lat = float(row[latitude_column_name])

                # Longitude
                if longitude_column_name not in row:
                    raise ValueError(f'Column {longitude_column_name} is required in input population file.')
                lon = float(row[longitude_column_name])

                # Node label
                res_in_deg = cls.arcsec_to_deg(cls.VALID_RESOLUTIONS[res_in_arcsec])
                node_label = row['node_label'] if 'node_label' in row else nodeid_from_lat_lon(lat, lon, res_in_deg)

                # Population
                pop = int(float(row[population_column_name])) if population_column_name in row else default_pop

                # Append the newly created node to the list
                nodes_list.append(Node(lat, lon, pop, node_label))

        demo = cls(nodes_list, demographics_type, res_in_arcsec, update_demographics, default_pop)
        demographics = demo.generate_demographics()

        demo_f = open(demographics_filename, 'w+')
        json.dump(demographics, demo_f, indent=4)
        demo_f.close()

    @classmethod
    def validate_res_in_arcsec(cls, res_in_arcsec):
        """
        Validate that the resolution is calid
        Args:
            res_in_arcsec: Resolution in arsecon. Supprted values can be found in VALID_RESOLUTIONS

        Returns:
            None.
        Raise:
            KeyError: If the resolution is invalid, a key error is raised
        """
        try:
            cls.VALID_RESOLUTIONS[res_in_arcsec]
        except KeyError:
            raise InvalidResolution(f"{res_in_arcsec} is not a valid arcsecond resolution."
                                    f" Must be one of: {cls.VALID_RESOLUTIONS.keys()}")

    @staticmethod
    def equilibrium_age_distribution(birth_rate, mort_scale, mort_value, max_age=100):
        """
        from Kurt Frey-- this function generates an initial age distribution already at equilibrium,
        allowing you to run burn-ins for immunity establishment only.

        NB: You must set your config file to Birth_Rate_Dependence="FIXED_BIRTH_RATE" and
                                    Age_Initialization_Distribution_Type= "DISTRIBUTION_COMPLEX"
        for this to work. If you have age-dependent birth rates, go talk to Kurt for a modified script.

        Args:
            birth_rate: population birth rate, in units of births/node/day
            mort_scale: demographics["Defaults"]["IndividualAttributes"]["MortalityDistribution"]["ResultScaleFactor"]
            mort_value: annual deaths per 1000, set to mirror birth rates.
            max_age: age past which you want everyone to die. In the current implementation you will get *some* mass
                    in this bin, but only the amound specified in the equilibrium distribution. If you have more
                    age-specific mortality rates, you can implement them with only a slight modification to this script.

        Returns:
            resval (a list of age bins in days from 0 to max_yr),
                    and distval (a list of the cumulative proportion of the population in each age bin of resval)
        """
        # define daily mortality probability by age group
        age_vec = [365 * 0, 365 * (max_age - 0.001), 365 * max_age]
        mort_vec = [mort_scale * mort_value, mort_scale * mort_value, 1]

        max_yr = 120
        day_to_year = 365

        # add bounds around age_vec and mort_vec
        mvec_x = [-1] + age_vec + [max_yr * day_to_year + 1]
        mvec_y = [mort_vec[0]] + mort_vec + [mort_vec[-1]]

        # cumulative monthly survival probabilities
        m_x = np.arange(0, max_yr * day_to_year, 30)
        mval = (1.0 - np.interp(m_x, xp=mvec_x, fp=mvec_y)) ** 30

        # create normalized population pyramid
        popvec = birth_rate * np.cumprod(mval)
        tpop = np.trapz(popvec, x=m_x)
        npvec = popvec / tpop
        # what proportion of  people are in the simulation up to bin i
        cpvec = sp.cumtrapz(npvec, x=m_x, initial=0)

        resval = np.around(np.linspace(0, max_yr * day_to_year))
        distval = np.interp(resval, xp=m_x, fp=cpvec)

        return resval, distval

    def set_resolution(self, res_in_arcsec):
        """
        The canonical way to set arcsecond/degree resolutions on a DemographicsGenerator object. Verifies everything
        is set properly

        Args:
            res_in_arcsec: The requested resolution. e.g. 30, 250, 'custom'

        Returns: No return value.

        """
        self.validate_res_in_arcsec(res_in_arcsec)
        self.custom_resolution = True if res_in_arcsec == self.CUSTOM_RESOLUTION else False
        self.res_in_arcsec = self.VALID_RESOLUTIONS[res_in_arcsec]
        self.res_in_degrees = self.arcsec_to_deg(self.res_in_arcsec)
        logger.debug("Setting resolution to %s arcseconds (%s deg.) from selection: %s" %
                     (self.res_in_arcsec, self.res_in_degrees, res_in_arcsec))

    def generate_defaults(self):
        """
        Generate the defaults section of the demographics file

        all of the below can be taken care of by a generic Demographics class
        (see note about refactor in dtk.generic.demographics)
        """

        population_removal_rate = self.default_birth_rate  # Corresponds to a fairly stable population

        mod_mortality = {
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
            "ResultScaleFactor": self.mort_scale_factor,
            "ResultValues": [
                [population_removal_rate],
                [population_removal_rate]
            ]
        }

        individual_attributes = {
            "MortalityDistribution": mod_mortality,
            "RiskDistribution1": 1,
            "PrevalenceDistributionFlag": self.prevalence_flag,
            "PrevalenceDistribution1": self.prevalence1,
            "PrevalenceDistribution2": self.prevalence2,
            "RiskDistributionFlag": 0,
            "RiskDistribution2": 0,
            "MigrationHeterogeneityDistribution1": 1,
            "ImmunityDistributionFlag": 0,
            "MigrationHeterogeneityDistributionFlag": 0,
            "ImmunityDistribution1": 1,
            "MigrationHeterogeneityDistribution2": 0,
            "ImmunityDistribution2": 0
        }

        node_attributes = {
            "Urban": 0,
            "AbovePoverty": 0.5,
            "Region": 1,
            "Seaport": 0,
            "Airport": 0,
            "Altitude": 0
        }

        if self.default_pop:
            node_attributes.update({"InitialPopulation": self.default_pop})

        defaults = {
            'IndividualAttributes': individual_attributes,
            'NodeAttributes': node_attributes,
        }

        return defaults

    def generate_nodes(self):
        """
        generate demographics file nodes
        """

        nodes = []
        for i, node in enumerate(self.nodes):
            # if res_in_degrees is custom assume node_ids are generated for a household-like setup and not based
            # on lat/lon
            if node.forced_id:
                node_id = node.forced_id
            elif self.custom_resolution:
                node_id = i + 1
            else:
                node_id = nodeid_from_lat_lon(float(node.lat), float(node.lon), self.res_in_degrees)
            node_attributes = node.to_dict()
            individual_attributes = {}

            if self.demographics_type != DemographicsType.STATIC:
                print(self.demographics_type)
                raise ValueError("Demographics type " + str(self.demographics_type) + " is not implemented!")

            # if nodes are in different countries, find node-specific mortality rates
            if "Country" in node_attributes.keys():
                country_condition = self.birthrate_df['Country Name'] == node_attributes["Country"]
                birth_rate = self.birthrate_df[country_condition][self.year].values[0]

                mod_mortality = {
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
                    "ResultScaleFactor": self.mort_scale_factor,
                    "ResultValues": [
                        [birth_rate],
                        [birth_rate]
                    ]
                }

                node_attributes.pop("Country", None)

                individual_attributes.update({"MortalityDistribution": mod_mortality})

            else:
                birth_rate = self.default_birth_rate

            # equilibrium age distribution
            per_node_birth_rate = (float(node.pop) / 1000) * birth_rate / 365.0
            node_attributes.update({'BirthRate': per_node_birth_rate})
            resval, distval = self.equilibrium_age_distribution(per_node_birth_rate, self.mort_scale_factor,
                                                                birth_rate)
            mod_age = {
                "DistributionValues":
                    [
                        distval.tolist()
                    ],
                "ResultScaleFactor": 1,
                "ResultValues":
                    [
                        resval.tolist()
                    ]
            }
            individual_attributes.update({"AgeDistribution": mod_age})

            nodes.append({'NodeID': node_id,
                          'NodeAttributes': node_attributes,
                          'IndividualAttributes': individual_attributes})

        return nodes

    def generate_metadata(self):
        """
        generate demographics file metadata
        """
        if self.custom_resolution:
            reference_id = 'Custom user'
        else:
            reference_id = "Gridded world grump%darcsec" % self.res_in_arcsec

        metadata = {
            "Author": "idm",
            "Tool": "dtk-tools",
            "IdReference": reference_id,
            "DateCreated": str(datetime.now()),
            "NodeCount": len(self.nodes),
            "Resolution": int(self.res_in_arcsec)
        }

        return metadata

    # TODO - Move to malaria
    # @staticmethod
    # def add_larval_habitat_multiplier(demographics):
    #     calib_single_node_pop = 1000.
    #     for node_item in demographics['Nodes']:
    #         pop_multiplier = float(node_item['NodeAttributes']['InitialPopulation']) / calib_single_node_pop
    #
    #         # Copy the larval param dict handed to this node
    #         node_item['NodeAttributes']['LarvalHabitatMultiplier'] *= pop_multiplier
    #
    #     return demographics

    def generate_demographics(self):
        """
        return all demographics file components in a single dictionary; a valid DTK demographics file when dumped as json
        """
        self.demographics = {'Nodes': self.generate_nodes(),
                             'Defaults': self.generate_defaults(),
                             'Metadata': self.generate_metadata()}

        if self.update_demographics:
            # update demographics before dict is written to file, via a user defined function and arguments
            # self.update_demographics is a partial object (see python docs functools.partial) and
            # self.update_demographics.func references the user's function the only requirement for the user defined
            # function is that it needs to take a keyword argument demographics

            self.update_demographics(demographics=self.demographics, **self.kwargs)

        return self.demographics
