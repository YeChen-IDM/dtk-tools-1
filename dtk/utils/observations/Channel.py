class Channel:
    ALLOWED_TYPES = ['fraction', 'count']
    def __init__(self, name, type):
        self.name = name
        type = type.lower()
        if not type in self.ALLOWED_TYPES:
            raise Exception('Channel type is %s, must be one of: %s' % (type, self.ALLOWED_TYPES))
        self.type = type

    @property
    def needs_pop_scaling(self):
        return self.type == 'count'
