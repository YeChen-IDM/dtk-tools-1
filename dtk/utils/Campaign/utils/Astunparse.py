# coding: utf-8
from __future__ import absolute_import
from six.moves import cStringIO
from .Unparser import Unparser

__version__ = '1.5.0'


def unparse(tree):
    v = cStringIO()
    Unparser(tree, file=v)
    return v.getvalue()
