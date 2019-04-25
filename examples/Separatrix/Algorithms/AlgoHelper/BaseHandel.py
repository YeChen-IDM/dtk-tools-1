from abc import ABCMeta, abstractmethod


class BaseHandel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass
