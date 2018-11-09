import importlib
import numpy as np

from abc import ABCMeta, abstractmethod


class BaseDistribution(object, metaclass=ABCMeta):

    class UnknownDistributionException(Exception): pass

    LOG_FLOAT_TINY = np.log(np.finfo(float).tiny)

    def __init__(self):
        self.additional_channels = []

    @abstractmethod
    def prepare(self, dfw, channel, provinciality, age_bins):
        pass

    @abstractmethod
    def compare(self, df, reference_channel, data_channel):
        pass

    @classmethod
    def from_string(cls, distribution_name):
        distribution_class = distribution_name.lower().capitalize() + 'Distribution'

        try:
            distribution_class = getattr(importlib.import_module('dtk.utils.observations.%s' % distribution_class),
                                         distribution_class)
        except ModuleNotFoundError:
            raise cls.UnknownDistributionException('No distribution class exists for: %s' % distribution_name)

        return distribution_class()
