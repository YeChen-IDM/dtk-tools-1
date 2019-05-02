from dtk.tools.serialization import Serialization_KS_Testing as skt
from shutil import copyfile
from os import path
from os import getcwd
from simtools.SetupParser import SetupParser
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from multiprocessing import freeze_support

import unittest

def get_config_ready():
    path_to_SEIR = path.join("..","..","..","Scenarios","Generic","03_SEIR")
    files_to_copy = ["param_overrides.json","campaign.json"]
    working_dir = getcwd()
    for f in files_to_copy:
        from_path = path.join(path_to_SEIR, f)
        to_path = path.join(working_dir, f)
        copyfile(from_path, to_path)
    # TODO: Find solution for flattening config file




def do_stuff():
    SetupParser.default_block = 'HPC'
    gen_cb = DTKConfigBuilder.from_files(config_name="Generic_SEIR/config.json", campaign_name="Generic_SEIR/campaign.json")
    gen_cb.params['Simulation_Duration'] = 200
    exp_tags = {}
    exp_tags['role'] = 'serialization_test'
    exp_tags['model'] = 'generic'
    s_ts = [5, 25, 150]
    T = skt.SerializationKsTest(config_builder=gen_cb,
                                experiment_name='Generic serialization test',
                                experiment_tags=exp_tags,
                                timesteps_to_serialize=s_ts,
                                inset_channels=['New Infections',
                                                'Infected',
                                                'Statistical Population'])
    T.run_test()

if __name__ == '__main__':
    freeze_support()
    do_stuff() # It is necessary to call freeze_support() before any of the rest here.