import shelve
from dtk.utils.analyzers.BaseAnalyzer import BaseAnalyzer

class BaseShelfAnalyzer(BaseAnalyzer):

    LOCK = None # must be defined by child classes

    def __init__(self, shelf_filename=None):
        super(BaseShelfAnalyzer, self).__init__()
        if not shelf_filename:
            shelf_filename = self.__class__.__name__ + ".shelf" # ck4, does this need more pathing (e.g. inside download directory)
        self.shelf_filename = shelf_filename
        self._shelf = None # must set in initialize()

    def initialize(self):
        # create the shelf to use
        self._shelf = shelve.open(self.shelf_filename, writeback=True)
        if self.LOCK:
            self.lock = ShelfLock(self.LOCK)
        else:
            raise Exception('Child classes of BaseShelfAnalyzer must define the class constant LOCK as a Lock object.')

    def update_shelf(self, key, value):
        with self.lock as lock:
            self._shelf[str(key)] = value
            self._shelf.sync()

    def from_shelf(self, key):
        key = str(key)
        try:
            value = self._shelf[key]
        except KeyError:
            value = None
        return value

    def is_in_shelf(self, key):
        value = True
        try:
            self._shelf[str(key)]
        except KeyError:
            value = False
        return value

class ShelfLock(object):
    """
    Allows use of a multiprocessing.Lock object using the 'with ... as ...' syntax
    """
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release()
