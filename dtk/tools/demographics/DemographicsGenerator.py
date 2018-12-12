import csv
import json
from datetime import datetime
from typing import Optional

from dtk.tools.demographics.DemographicsGeneratorModel import DemographicsType
from dtk.tools.demographics.Node import Node, nodeid_from_lat_lon
from dtk.tools.demographics.generator.DemographicsGeneratorConcern import DemographicsGeneratorConcern
from dtk.tools.demographics.generator.DemographicsNodeGeneratorConcern import DemographicsNodeGeneratorConcern, \
    DefaultWorldBankEquilibriumConcern
from simtools.Utilities.General import init_logging

logger = init_logging('DemographicsGenerator')


class InvalidResolution(BaseException):
    pass


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

    def __init__(self, nodes, node_concern: Optional[DemographicsNodeGeneratorConcern] = None,
                 demographics_concern: Optional[DemographicsGeneratorConcern] = None,
                 res_in_arcsec=DEFAULT_RESOLUTION):
        """
        Initialize the SpatialManager
        Args:


            nodes: list of nodes
            node_concern (Optional[DemographicsNodeGeneratorConcern]): What DemographicsNodeGeneratorConcern should
            we apply. If not specified, we use the DefaultWorldBankEquilibriumConcern
            demographics_concern (Optional[DemographicsGeneratorConcern]): Any concern generator we need to execute
            after the Demographics object has been generated, but not saved
            res_in_arcsec: Simulation grid resolution
        """
        self.nodes = nodes
        #  currently only static is implemented in generate_nodes(self)
        self.demographics_type = DemographicsType.STATIC # could be 'static', 'growing' or a different type;
        self.set_resolution(res_in_arcsec)
        self.node_concern = node_concern
        self.demographics_concern = demographics_concern

        # demographics data dictionary (working DTK demographics file when dumped as json)
        self.demographics = None

        if node_concern is None:
            self.node_concern = DefaultWorldBankEquilibriumConcern()

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
                       node_concern: Optional[DemographicsNodeGeneratorConcern] = None,
                       demographics_concern: Optional[DemographicsGeneratorConcern] = None,
                       res_in_arcsec=DEFAULT_RESOLUTION, default_population: int = 1000,
                       latitude_column_name: str = 'lat',
                       longitude_column_name: str = 'lon', population_column_name: str = 'pop'):
        """

        Generates a demographics file from a CSV population

        Args:
            population_input_file: CSV population file. Must contain all the columns specified by latitude_column_name,
                longitude_column_name. The population_column_name is optional. If not found, we fall back to default_population
            demographics_filename: demographics file to save the demographics file too
            node_concern (Optional[DemographicsNodeGeneratorConcern]): What DemographicsNodeGeneratorConcern should
            we apply. If not specified, we use the DefaultWorldBankEquilibriumConcern
            demographics_concern (Optional[DemographicsGeneratorConcern]): Any concern generator we need to execute
            after the Demographics object has been generated, but not saved
            res_in_arcsec:
            default_population: Default population. Only used if population_column_name does not exist
            latitude_column_name: Column name to load latitude values from
            longitude_column_name: Column name to load longitude values from
            population_column_name: Column name to load population values from

        Returns:
            DemographicsGenerator used to generate demogrphics file
        """
        nodes_list = list()
        warn_no_pop = False
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
                if not warn_no_pop and population_column_name not in row:
                    warn_no_pop = True
                    logger.warn(f'Could not location population column{population_column_name}. Using the default '
                                f'population value of {default_population}')
                pop = int(float(row[population_column_name])) if population_column_name in row else default_population

                # Append the newly created node to the list
                nodes_list.append(Node(lat, lon, pop, node_label))

        demo = cls(nodes_list, node_concern=node_concern, demographics_concern=demographics_concern,
                   res_in_arcsec=res_in_arcsec)
        demographics = demo.generate_demographics()

        demo_f = open(demographics_filename, 'w+')
        json.dump(demographics, demo_f, indent=4)
        demo_f.close()
        return demographics

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

    def generate_nodes(self, defaults):
        """
        generate demographics file nodes


        The process for generating nodes starts with looping through the loaded demographics nodes. For each node,
        we:
        1. First determine the node's id. If the node has a forced id set, we use that. If we are using a custom
            resolution, we use the index(ie 1, 2, 3...). Lastly, we build the node id from the lat and lon id of the
             node

        2. We then start to populate the node_attributes and individual attributes for the current ndoe. The
            node_attributes will have data loaded from the initial nodes fed into DemographicsGenerator. The individual
            attributes start off as an empty dict.

        3. We next determine the birthrate for the node. If the node attributes contains a Country element, we first
            lookup the birthrate from the World Pop data. We then build a MortalityDistribution configuration with
            country specific configuration elements  and add that to the individual attributes. If there is no Country
            element in the node attributes, we set the birth rate to the default_birth_rate. This value was set in
            initialization of the DemographicsGenerator to the birth rate of the specified country from the world pop data

        4. We then calculate the per_node_birth_rate using get_per_node_birth_rate and then set the birth rate
           on the node attributes
        5. We then calculate the equilibrium_age_distribution and use that to create the AgeDistribution in
            individual_attributes
        6. We then add each new demographic node to a list to end returned at the end of the function

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

            # Run our model through our Concern Set
            self.node_concern.update_node(defaults, node, node_attributes, individual_attributes)
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

    def generate_demographics(self):
        """
        return all demographics file components in a single dictionary; a valid DTK demographics file when dumped as json
        """
        defaults = {}
        self.node_concern.update_defaults(defaults)
        self.demographics = {'Nodes': self.generate_nodes(defaults),
                             'Defaults': defaults,
                             'Metadata': self.generate_metadata()}

        if self.demographics_concern:
            self.demographics_concern.update_demographics(self.demographics)
        return self.demographics
