from simtools.Analysis.BaseAnalyzers.BaseAnalyzer import BaseAnalyzer
import shutil
import os


class BaseCacheAnalyzer(BaseAnalyzer):

    def __init__(self, cache_location=None, force=False, delete_cache_when_done=False, **kwargs):
        super().__init__(**kwargs)
        self.cache_location = cache_location
        self.cache = None
        self.force = force
        self.delete_cache_when_done = delete_cache_when_done

    def initialize(self):
        from diskcache import Cache
        self.cache = Cache(self.cache_location or self.uid + "_cache")

    def filter(self, simulation):
        return self.force or not self.is_in_cache(simulation.id)

    def to_cache(self, key, data):
        self.cache.set(key, data)

    def from_cache(self, key):
        return self.cache.get(key)

    def is_in_cache(self, key):
        return key in self.cache

    def destroy(self):
        if self.cache:
            self.cache.close()

        if self.cache and self.delete_cache_when_done and os.path.exists(self.cache.directory):
            cache_directory = self.cache.directory
            del self.cache
            shutil.rmtree(cache_directory)

    @property
    def keys(self):
        return list(self.cache.iterkeys()) if self.cache else None
