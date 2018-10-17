import importlib
import numpy as np

from abc import ABCMeta, abstractmethod


class BaseDistribution(object, metaclass=ABCMeta):

    LOG_FLOAT_TINY = np.log(np.finfo(float).tiny)

    def __init__(self):
        self.additional_channels = []

    # dummy stub that can be overridden if needed
    def prepare(self, dfw, channel, provinciality):
        return dfw

    @abstractmethod
    def compare(self, df, reference_channel, data_channel):
        pass

    @classmethod
    def from_string(cls, distribution_name):
        distribution_class = distribution_name.lower().capitalize() + 'Distribution'
        distribution_class = getattr(importlib.import_module('dtk.utils.observations.%s' % distribution_class),
                                     distribution_class)
        return distribution_class()
