import shelve

from dtk.utils.analyzers.BaseAnalyzer import BaseAnalyzer

class BaseShelfAnalyzer(BaseAnalyzer):

    def __init__(self):
        super(BaseShelfAnalyzer, self).__init__()
        self.shelf_filename = self.__class__.__name__ + ".shelf"  # ck4, does this need more pathing (e.g. inside download directory)
        self.shelf = None # set in initialize()

    def initialize(self):
        # create the shelf to use
        self.shelf = shelve.open(self.shelf_filename, writeback=True)

    def update_shelf(self, key, value):
        self.shelf[str(key)] = value
        self.shelf.sync()

    def from_shelf(self, key):
        key = str(key)
        try:
            value = self.shelf[key]
        except KeyError:
            value = None
        return value

    def is_in_shelf(self, key):
        value = True
        try:
            self.shelf[str(key)]
        except KeyError:
            value = False
        return value
