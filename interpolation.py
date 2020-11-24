"""
Routines for interpolating, and rebinning data
"""
import numpy as np
from scipy.interpolate import interp1d, Akima1DInterpolator

__all__ = [
    "interp",
    "interp_nan",
    "interp_inf",
]

def interp(x, y, e, xi, kind, model, **kwargs):
    """
    Interpolates data the wavelength axis X, if X is a numpy array,
    or X.x if X is Spectrum type. This returns a new spectrum rather than
    updating a spectrum in place, however this can be acheived by

    >>> S1 = S1.interp(X)

    Wavelengths outside the range of the original spectrum are filled with
    zeroes.
    """
    if kind == "Akima":
        yi = Akima1DInterpolator(x, y)(xi)
        ei = Akima1DInterpolator(x, e)(xi)
        nan = np.isnan(yi) | np.isnan(ei)
        yi[nan] = 0.
        if not model:
            ei[nan] = 0.
    elif kind == "sinc":
        yi = lanczos(x, y, xi)
        extrap = (xi < x.min()) | (xi > x.max())
        yi[extrap] = 0.
        if not model:
            ei = lanczos(x, np.log(e+1E-300), xi)
            ei[extrap] = np.inf
    else:
        yi = interp1d(x, y, kind=kind, \
            bounds_error=False, fill_value=0., **kwargs)(xi)
        if not model:
            #If any errors were inf (zero weight) we need to make
            #sure the interpolated range stays inf
            inf = np.isinf(e)
            if np.any(inf):
                ei = interp1d(x[~inf], e[~inf], kind=kind, \
                    bounds_error=False, fill_value=np.inf, **kwargs)(xi)
                inf2 = interp1d(x, inf, kind='nearest', \
                    bounds_error=False, fill_value=0, **kwargs)(xi).astype(bool)
                ei[inf2] = np.inf
            else:
                ei = interp1d(x, e, kind=kind, \
                    bounds_error=False, fill_value=np.inf, **kwargs)(xi)

    ei = 0 if model else np.where(ei < 0, 0, ei)
    return yi, ei

def interp_nan(x, y, e, interp_e):
    """
    Linearly interpolate over values with NaNs. If interp_e is set, these
    are also interpolated over, otherwise they are set to np.inf.
    """
    bad = np.isnan(y) | np.isnan(e)
    kwargs = {'kind':'linear', 'bounds_error':False}
    y[bad] = interp1d(x[~bad], y[~bad], fill_value=0, **kwargs)(x[bad])
    e[bad] = interp1d(x[~bad], e[~bad], fill_value=np.inf, **kwargs)(x[bad]) \
        if interp_e else np.inf
    return y, e

def interp_inf(x, y, e):
    """
    Linearly interpolate over values with infs
    """
    bad = np.isinf(y) | np.isinf(e)
    y[bad] = interp1d(x[~bad], y[~bad], bounds_error=False, fill_value=0)(x[bad])
    e[bad] = interp1d(x[~bad], e[~bad], bounds_error=False, fill_value=0)(x[bad])
    return y, e

def lanczos(x, y, xnew):
    """
    Helper function used for Lanczos interpolation.
    """
    n = np.arange(len(x))
    Ni = interp1d(x, n, kind='linear', fill_value='extrapolate')(xnew)
    ynew = [np.sum(y*np.sinc(ni-n)) for ni in Ni]
    return np.array(ynew)
