#!/usr/bin/python

from lz4 import block


# noinspection PyCamelCase
class Uncompressed(object):

    @classmethod
    def compress(cls, data):
        return data

    @classmethod
    def uncompress(cls, data):
        return data


class EllZeeFour(object):

    @classmethod
    def compress(cls, data):
        return block.compress(data.encode())

    @classmethod
    def uncompress(cls, data):
        return block.decompress(data)


class SerialObject(dict):
    # noinspection PyDefaultArgument
    def __init__(self, dictionary={}):
        super(SerialObject, self).__init__(dictionary)
        self.__dict__ = self
        return