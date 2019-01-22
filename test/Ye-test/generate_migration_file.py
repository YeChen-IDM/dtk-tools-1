import os
import json
import numpy as np
import collections
import datetime
from typing import Union
from geopy.distance import distance
from struct import pack, unpack
from dtk.tools.migration.MigrationFile import MigrationTypes


def generate_txt_from_demo(demo_file: str = None, gravity_params: list = None,
                           migration_file_path:str = 'migration.txt', exclude_nodes: Union[list, None] = None):
    """
    Generate migration txt file using Static Gravity Migration Model from demographics file.
    Args:
        demo_file: demographics file
        gravity_params: list that contains 4 gravity parameters
        migration_file_path: file path to write the migration text file.
        exclude_nodes: List of nodes to exclude from migration rate generation

    Returns:

    """

    if not os.path.isfile(demo_file):
        raise ValueError(f"A demographic file is required to generate {migration_file_path} file.")

    if len(gravity_params) != 4:
        raise ValueError("You must provide all 4 gravity params")

    if not migration_file_path:
        # if user enter an empty migration_file_path, use default value 'migration.txt'.
        migration_file_path = 'migration.txt'
    else:
        migtation_file_dirname = os.path.dirname(migration_file_path)
        if migtation_file_dirname:
            # Create folder to save migration file if user input a path that doesn't exist
            if not os.path.isdir(migtation_file_dirname):
                os.mkdir(migtation_file_dirname)

    with open(demo_file, 'r') as json_file:
        demo_json = json.load(json_file)

    nodes = demo_json["Nodes"]
    with open(migration_file_path, 'w') as output_file:
        for node in nodes:
            id_1 = node["NodeID"]
            lat_1 = node["NodeAttributes"]["Latitude"]
            lon_1 = node["NodeAttributes"]["Longitude"]
            pop_1 = node["NodeAttributes"]["InitialPopulation"]
            for node_2 in nodes:
                id_2 = node_2["NodeID"]
                if id_1 != id_2:
                    if not exclude_nodes or (exclude_nodes and id_1 not in exclude_nodes and id_2 not in exclude_nodes):
                        lat_2 = node_2["NodeAttributes"]["Latitude"]
                        lon_2 = node_2["NodeAttributes"]["Longitude"]
                        d = distance((lat_1, lon_1), (lat_2, lon_2)).km
                        pop_2 = node_2["NodeAttributes"]["InitialPopulation"]
                        num_trips = gravity_params[0] * pop_1 ** gravity_params[1] * pop_2 ** gravity_params[2] * d ** \
                                    gravity_params[3]
                        prob_trip = np.min([1., num_trips / pop_1])
                        output_file.write(f"{id_1} {id_2} {prob_trip}\n")


def generate_txt_from_bin(migrationfile: str = "migration.bin",
                          demographicsJson: str = "migration_generator_test_demographics.json",
                          outputfile:str = "migration_2.txt",
                          number_destinations: int = 27):
    """
    generate migration text file from binary file, modified based on get_migration_from_bin_file.py from DtkTrunk repo
    Args:
        migrationfile: migration binary file path.
        demographicsJson: demographics json file path.
        outputfile: file path to write the migration text file.
        number_destinations: max number of destinations for a single node.

    Returns:

    """

    ID = []
    rate = []

    # read the demographics file
    with open (demographicsJson, 'r') as demo_json_file:
        flat = json.load(demo_json_file)

    coordinate = {}

    IDlist = []

    for i in range(len(flat["Nodes"])):
        vID = flat["Nodes"][i]["NodeID"]
        vlat = flat["Nodes"][i]["NodeAttributes"]["Latitude"]
        vlon = flat["Nodes"][i]["NodeAttributes"]["Longitude"]
        coordinate[vID] = (vlat, vlon)
        IDlist.append(vID)

    # read the binary migration file
    with open(migrationfile, "rb") as fopen:
        byte = "a"
        while byte:
            for i in range(number_destinations):
                byte = fopen.read(4)
                if byte:
                    ID.append(int(unpack('L', byte)[0]))
            for i in range(number_destinations):
                byte = fopen.read(8)
                if byte:
                    rate.append(float(unpack('d', byte)[0]))

    with open(outputfile, 'w') as fout:
        IDseq = 0
        count = 0
        for i in range(len(ID)):
            if ID[i] != 0:
                fout.write(f'{IDlist[IDseq]} {ID[i]} {rate[i]}\n')
            count += 1
            if count % number_destinations == 0:
                IDseq += 1


def generate_migration_files_from_txt(filename: str = "migration.txt", outfilename: str = "migration.bin",
                                      mig_type: MigrationTypes = MigrationTypes.local, id_ref: str = "Custom user"):
    """
    modified based on convert_txt_to_bin.py in DtkTrunk repo
    -----------------------------------------------------------------------------
    This script converts a txt file to an EMOD binary-formatted migration file.
    It also creates the required metadata file.

    The txt file has three columns
       From_Node_ID To_Node_ID Rate (Average # of Trips Per Day)
    where the node ID's are the external ID's found in the demographics file.
    Each node ID in the migration file must exist in the demographics file.
    One can have node ID's in the demographics that don't exist in the migration file.

    The txt file does not have to have the same number of entries for each From_Node.
    The script will find the From_Node that has the most and use that for the
    DestinationsPerNode.  The binary file will have DestinationsPerNode entries
    per node.
    -----------------------------------------------------------------------------
    Args:
        filename:
        outfilename:
        mig_type:
        id_ref:

    Returns:

    """
    if not isinstance(mig_type, MigrationTypes):
        raise ValueError("Invalid MigrationType = " + mig_type)

    max_destinations_per_node = 0
    destinations_per_node = 0

    # ----------------------------
    # collect data from txt file
    # ----------------------------
    net = {}
    net_rate = {}
    node_id_list = []
    prev_id = -1
    with open(filename) as fopen:
        for line in fopen:
            s = line.strip().split(' ')
            ID1 = int(float(s[0]))
            ID2 = int(float(s[1]))
            rate = float(s[2])
            if ID1 not in net:
                net[ID1] = []
                net_rate[ID1] = []
            net[ID1].append(ID2)
            net_rate[ID1].append(rate)
            if prev_id != ID1:
                if (destinations_per_node > max_destinations_per_node):
                    max_destinations_per_node = destinations_per_node
                node_id_list.append(ID1)
                # print(prev_id, max_destinations_per_node)
                prev_id = ID1
                destinations_per_node = 0
            destinations_per_node += 1

    # ---------------
    # Write bin file
    # ---------------
    with open(outfilename, 'wb') as fout:
        for ID in net:
            ID_write = []
            ID_rate_write = []
            for i in range(max_destinations_per_node):
                ID_write.append(0)
                ID_rate_write.append(0)
            for i in range(len(net[ID])):
                ID_write[i] = net[ID][i]
                ID_rate_write[i] = net_rate[ID][i]
            s_write = pack('L' * len(ID_write), *ID_write)
            s_rate_write = pack('d' * len(ID_rate_write), *ID_rate_write)
            fout.write(s_write)
            fout.write(s_rate_write)

    # -------------------------------------------------------------------
    # Create NodeOffsets string
    # This contains the location of each From Node's data in the bin file
    # -------------------------------------------------------------------
    offset_str = ""
    nodecount = 0

    for ID in net:
        offset_str += '%0.8X' % ID
        offset_str += '%0.8X' % (
                    nodecount * max_destinations_per_node * 12)  # 12 -> sizeof(uint32_t) + sizeof(double)
        nodecount += 1

    # -------------------
    # Write Metadata file
    # -------------------
    migjson = collections.OrderedDict([])
    migjson['Metadata'] = {}
    migjson['Metadata']['Author'] = os.environ['USERNAME']
    migjson['Metadata']['NodeCount'] = len(node_id_list)
    migjson['Metadata']['IdReference'] = id_ref
    migjson['Metadata']['DateCreated'] = datetime.datetime.now().ctime()
    migjson['Metadata']['Tool'] = __name__
    migjson['Metadata']['DatavalueCount'] = max_destinations_per_node
    migjson['Metadata']['MigrationType'] = mig_type.value
    migjson['NodeOffsets'] = offset_str

    with open(outfilename + ".json", 'w') as file:
        json.dump(migjson, file, indent=4)




