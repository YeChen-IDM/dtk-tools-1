import Serialization_KS_Testing as skt

from simtools.SetupParser import SetupParser
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from multiprocessing import freeze_support

import unittest


def do_stuff():
    SetupParser.default_block = 'HPC'
    gen_cb = DTKConfigBuilder.from_defaults('GENERIC_SIM_SEIR')
    gen_cb.params["Simulation_Duration"] = 200
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