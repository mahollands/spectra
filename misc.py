"""
Contains additional functions for working with spectra, some of which are used
elsewhere in this module.
"""

from functools import reduce
import operator
from enum import Enum
import numpy as np
from scipy.special import wofz

__all__ = [
    "jangstrom",
    "voigt",
    "vac_to_air",
    "air_to_vac",
    "logarange",
    "keep_points",
]

jangstrom = \
    r"$\mathrm{erg}\;\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$"

rt2pi = np.sqrt(2*np.pi)
rt2 = np.sqrt(2)
fwhm2sigma = 1/(2*np.sqrt(2*np.log(2)))


class Wave(Enum):
    """
    Wavelengths can only be air or vacuum.
    """
    AIR, VAC = "air", "vac"


def gaussian(x, x0, sigma=None, fwhm=None, norm=False):
    """
    Gaussian line profile

    Parameters
    ----------
    x : np.ndarray
        A 1-D array of wavelengths for the line profile.
    x0 : float
        Central wavelength of the profile.
    sigma: float
        The standard deviation of the Gaussian line profile.
    fwhm: float
        The full width half maximum can be given as an alternative to sigma.
    norm: bool
        By default the line profile has a maximum height of 1. If norm is
        instead True, then the line profile will have unit area.
    """
    if (sigma is None) ^ (fwhm is not None):
        raise ValueError("One and only one of sigma and fwhm should be given")
    if sigma is None:
        sigma = fwhm * fwhm2sigma
    xx = (x-x0)/sigma
    y = np.exp(-0.5*xx*xx)
    if norm:
        y /= rt2pi*sigma
    return y


def lorentzian(x, x0, fwhm=None, norm=False):
    """
    Lorentzian line profile

    Parameters
    ----------
    x : np.ndarray
        A 1-D array of wavelengths for the line profile.
    x0 : float
        Central wavelength of the profile.
    fwhm: float
        The full width half maximum can be given as an alternative to sigma.
    norm: bool
        By default the line profile has a maximum height of 1. If norm is
        instead True, then the line profile will have unit area.
    """
    gamma = fwhm/2
    xx = (x-x0)/gamma
    y = 1/(1+xx*xx)
    if norm:
        y /= np.pi * gamma
    return y


def voigt(x, x0, fwhm_g, fwhm_l):
    """
    Normalised voigt profile.

    Parameters
    ----------
    x : np.ndarray
        A 1-D array of wavelengths for the line profile.
    x0 : float
        Central wavelength of the profile.
    fwhm_g: float
        The Full-Width at Half-Maximum of the Gaussian part of the voigt
        profile.
    fwhm_l: float
        The Full-Width at Half-Maximum of the Lorentzian part of the voigt
        profile.
    """
    sigma = fwhm2sigma*fwhm_g
    z = ((x-x0) + 0.5j*fwhm_l)/(sigma*rt2)
    return wofz(z).real/(rt2pi*sigma)


def vac_to_air(Wvac):
    """
    converts vacuum wavelengths to air wavelengths, as per VALD3 documentation
    (in Angstroms)
    """
    s = 1e4/Wvac
    n = 1.0000834254 + 0.02406147/(130.-s*s) + 0.00015998/(38.9-s*s)
    return Wvac/n


def air_to_vac(Wair):
    """
    converts air wavelengths to vacuum wavelengths, as per VALD3 documentation
    (in Angstroms)
    """
    s = 1e4/Wair
    n = 1.00008336624212083 + 0.02408926869968 / (130.1065924522-s*s) \
        + 0.0001599740894897/(38.92568793293-s*s)
    return Wair*n


def _between(x, x1, x2):
    """
    Helper function to determine if a wavelength is between two others.
    """
    return (x > float(x1)) & (x < float(x2))


def keep_points(x, fname):
    """
    creates a mask for a spectrum that regions between pairs from a file
    """
    with open(fname, 'r') as F:
        segments = (_between(x, *line.split()) for line in F)
    return reduce(operator.or_, segments)


def logarange(x0, x1, R):
    """
    Like np.arange but with log-spaced points. The spacing parameter, R, is
    such that R = x/dx.
    """
    lx0, lx1 = np.log(x0), np.log(x1)
    logx = np.arange(lx0, lx1, 1/R)
    return np.exp(logx)
