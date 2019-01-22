import os
import shutil
import tempfile

from diskcache import FanoutCache, Cache, Deque, DEFAULT_SETTINGS

MAX_CACHE_SIZE = int(2**33)  # 8GB
DEFAULT_SETTINGS["size_limit"] = MAX_CACHE_SIZE
DEFAULT_SETTINGS["sqlite_mmap_size"] = 2 ** 28
DEFAULT_SETTINGS["sqlite_cache_size"] = 2 ** 15

class CacheEnabled:

    def __init__(self):
        self.cache_directory = None
        self.cache = None
        self.queue = False

    def initialize_cache(self, shards=None, timeout=1, queue=False):
        # Create a temporary directory for the cache
        self.cache_directory = tempfile.mkdtemp()

        # Create a queue?
        if queue:
            self.cache = Deque(directory=self.cache_directory)
            self.queue = True
        elif shards:
            self.cache = FanoutCache(self.cache_directory, shards=shards, timeout=timeout)
            self.queue = False
        else:
            self.cache = Cache(self.cache_directory, timeout=timeout)
            self.queue = False

        return self.cache

    def destroy_cache(self):
        self.cache.clear()
        if self.queue:
            # For the particular queue, we manually call the close on the internal cache
            self.cache._cache.close()
        else:
            # For all other cases, just call the normal close
            self.cache.close()

    def __del__(self):
        if self.cache:
            self.destroy_cache()

        if self.cache_directory and os.path.exists(self.cache_directory):
            shutil.rmtree(self.cache_directory)

