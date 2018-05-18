import unittest

from dtk.utils.observations.Channel import Channel


class TestChannel(unittest.TestCase):

    def setUp(self):
        self.stratified_string = 's:Water Import from Europa'
        self.undecorated_string = 'Noctis City'
        self.arbitrary_decorator_string = 'p:Eos Chasma National Park'

    def test_decorator_detection(self):
        channel = Channel(self.stratified_string)
        self.assertEqual(channel.decorator, 's')

        channel = Channel(self.undecorated_string)
        self.assertEqual(channel.decorator, None)

        channel = Channel(self.arbitrary_decorator_string)
        self.assertEqual(channel.decorator, 'p')

    def test_str_returns_decorated_string(self):
        channel = Channel(self.stratified_string)
        self.assertEqual(str(channel), self.stratified_string)

        channel = Channel(self.undecorated_string)
        self.assertEqual(str(channel), self.undecorated_string)

        channel = Channel(self.arbitrary_decorator_string)
        self.assertEqual(str(channel), self.arbitrary_decorator_string)

    def test_is_stratifier(self):
        channel = Channel(self.stratified_string)
        self.assertTrue(channel.is_stratifier)

        channel = Channel(self.undecorated_string)
        self.assertFalse(channel.is_stratifier)

        channel = Channel(self.arbitrary_decorator_string)
        self.assertFalse(channel.is_stratifier)

    def test_construct_channel_string_regression(self):
        name = 'Asteroid Mining'
        channel_string = Channel.construct_channel_string(name=name)
        self.assertEqual(channel_string, name)

        channel_string = Channel.construct_channel_string(name=name, decorator=None)
        self.assertEqual(channel_string, name)

        name = 'Carbonate Processing'
        decorator = 's'
        channel_string = Channel.construct_channel_string(name=name, decorator=decorator)
        self.assertEqual(channel_string, '%s:%s' % (decorator, name))

        decorator = 'h'
        channel_string = Channel.construct_channel_string(name=name, decorator=decorator)
        self.assertEqual(channel_string, '%s:%s' % (decorator, name))

    def test_deconstruct_channel_string_regression(self):
        channel_string = 'Asteroid Mining'
        parts = Channel.deconstruct_channel_string(channel=channel_string)
        self.assertEqual(parts['name'], channel_string)
        self.assertEqual(parts['decorator'], None)

        channel_string = 's:Asteroid Mining'
        parts = Channel.deconstruct_channel_string(channel=channel_string)
        self.assertEqual(parts['name'], channel_string.split(':')[1])
        self.assertEqual(parts['decorator'], channel_string.split(':')[0])

        channel_string = 'd:Asteroid Mining'
        parts = Channel.deconstruct_channel_string(channel=channel_string)
        self.assertEqual(parts['name'], channel_string.split(':')[1])
        self.assertEqual(parts['decorator'], channel_string.split(':')[0])

if __name__ == '__main__':
    unittest.main()
