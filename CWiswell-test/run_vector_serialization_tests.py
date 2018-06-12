import Serialization_KS_Testing as skt

from simtools.SetupParser import SetupParser
from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from dtk.vector.study_sites import configure_site
from multiprocessing import freeze_support

import unittest


def do_stuff():
    SetupParser.default_block = 'HPC'
    vec_cb = DTKConfigBuilder.from_defaults('VECTOR_SIM')
    configure_site(vec_cb, 'Namawala')
    exp_tags = {}
    exp_tags['role'] = 'serialization_test'
    exp_tags['model'] = 'vector'
    s_ts = [5, 25, 150]
    T = skt.SerializationKsTest(config_builder=vec_cb,
                                experiment_name='Vector serialization test',
                                experiment_tags=exp_tags,
                                timesteps_to_serialize=s_ts,
                                inset_channels=['Adult Vectors',
                                                'New Infections',
                                                'Infected',
                                                'Statistical Population',
                                                'Daily EIR'])
    T.run_test()

if __name__ == '__main__':
    freeze_support()
    do_stuff() # It is necessary to call freeze_support() before any of the rest here.