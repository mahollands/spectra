"""
Module for working with astropyhsical spectra
"""
__author__ = "Mark Hollands"
__email__ = "M.Hollands.1@warwick.ac.uk"

import numpy as np
import matplotlib.pyplot as plt
from .spec_class import * 
from .spec_io import *
from .misc import air_to_vac, vac_to_air, voigt, jangstrom
