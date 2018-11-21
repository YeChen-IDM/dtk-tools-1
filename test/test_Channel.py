import unittest

from dtk.utils.observations.Channel import Channel


class TestDistributions(unittest.TestCase):

    def test_fail_if_invalid_type_passed(self):
        self.assertRaises(Channel.InvalidChannelType,
                          Channel,
                          name='Prevalence',
                          type='X-Wing')

    def test_works_with_valid_types(self):
        for typ in Channel.ALLOWED_TYPES:
            channel = Channel(name='Prevalence', type=typ)
            if typ == 'count':
                self.assertTrue(channel.needs_pop_scaling)
            else:
                self.assertFalse(channel.needs_pop_scaling)

if __name__ == '__main__':
    unittest.main()
