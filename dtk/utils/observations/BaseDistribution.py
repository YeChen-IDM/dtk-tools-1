import importlib
import numpy as np

from abc import ABCMeta, abstractmethod


class BaseDistribution(object, metaclass=ABCMeta):

    class UnknownDistributionException(Exception): pass

    LOG_FLOAT_TINY = np.log(np.finfo(float).tiny)

    def __init__(self):
        self.additional_channels = []

    @abstractmethod
    def prepare(self, dfw, channel, weight_channel):
        pass

    @abstractmethod
    def compare(self, df, reference_channel, data_channel):
        pass

    @abstractmethod
    def add_percentile_values(self, dfw, channel, p):
        pass

    @classmethod
    def from_string(cls, distribution_name):
        distribution_class_name = cls._construct_distribution_class_name(distribution_name=distribution_name)

        try:
            distribution_class = getattr(importlib.import_module('dtk.utils.observations.%s' % distribution_class_name),
                                         distribution_class_name)
        except ModuleNotFoundError:
            raise cls.UnknownDistributionException('No distribution class exists for: %s' % distribution_name)

        return distribution_class()

    @classmethod
    def from_uncertainty_channel(cls, uncertainty_channel):
        import os
        this_file = os.path.basename(__file__)
        distribution_class_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.dirname(__file__))
                                    if f.endswith('.py') and f != this_file]
        distribution = None
        # import the distribution classes one at a time and check if they are the right one
        for distribution_class_name in distribution_class_names:
            try:
                distribution_class = getattr(importlib.import_module('dtk.utils.observations.%s' % distribution_class_name),
                                             distribution_class_name)
                if distribution_class.UNCERTAINTY_CHANNEL == uncertainty_channel:
                    distribution = distribution_class()
                    break
            except AttributeError:
                pass # not a valid class, so keep going
        if distribution is None:
            raise Exception('Unable to determine distribution that uses uncertainty channel: %s' % uncertainty_channel)
        return distribution

    @staticmethod
    def _construct_distribution_class_name(distribution_name):
        return distribution_name.lower().capitalize() + 'Distribution'
