"""Modules and subpackages for jumpmetrics"""
from .core import *
from .events import *
from .metrics import *
from .signal_processing import *

from importlib.metadata import version
__version__ = version("jumpmetrics")
