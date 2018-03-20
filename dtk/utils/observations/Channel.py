import re


class Channel:
    """
    Represents a string channel name that can be prefix-decorated. Decorations can be alphanumeric (\w)  e.g.
    Gender (undecorated)
    s:Gender (decorated, denoted as stratifier)
    X:Gender (decorated, no current interpretation of decorator)
    4:Gender (decorated, no current interpretation of decorator)
    """
    def __init__(self, string):
        parts = self.deconstruct_channel_string(string)
        self.name = parts['name']
        self.decorator = parts['decorator']

    @property
    def is_stratifier(self):
        return self.decorator == 's'

    def __str__(self):
        return self.construct_channel_string(self.name, self.decorator)

    @classmethod
    def construct_channel_string(cls, name, decorator=None):
        return name if decorator is None else '%s:%s' % (decorator, name)

    @classmethod
    def deconstruct_channel_string(cls, channel):
        regex = re.compile('^((?P<decorator>\w):)?(?P<name>.+)')
        match = regex.match(channel)
        decorator = match['decorator'] if match['decorator'] else None
        return {'name': match['name'], 'decorator': decorator}
