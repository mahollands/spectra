"""
Routines for calculating limb darkening coefficients and central intensities
"""
import numpy as np
from scipy.optimize import leastsq

__all__ = [
    "calc_limb_darkening_coeffs",
]

# See Claret et al., A&A 634, 93 (2020)
limb_functions = {
    'linear': (1, lambda P, mu: 1-P[0]*(1-mu)),
    'quad': (2, lambda P, mu: 1-sum(a*(1-mu)**i for i, a in enumerate(P, 1))),
    'sqrt': (2, lambda P, mu: 1-P[0]*(1-mu)-P[1]*(1-np.sqrt(mu))),
    'log': (2, lambda P, mu: 1-P[0]*(1-mu)-P[1]*mu*np.log(mu)),
    'power': (2, lambda P, mu: 1-P[0]*(1-mu**P[1])),
    'claret': (4, lambda P, mu: 1-sum(a_k*(1-mu**(k/2)) for k, a_k in enumerate(P, 1))),
}

def calc_limb_darkening_coeffs(MM, band, limb_model='claret', return_fluxes=False):
    """
    Calculate limb darkening coefficients for DK models for a given pandpass.
    MM is a list of models containing angular dependent spectra as a function
    of mu. Normalised fluxes and mu points can be returned if 'return_fluxes'
    is true.
    """
    n_param, f = limb_functions[limb_model]
    guess = np.zeros(n_param)

    mu = np.array([M.head['mu'] for M in MM])
    fluxes = np.array([M.flux_calc_AB(band) for M in MM])
    fluxes /= np.max(fluxes)

    vec, *_ = leastsq(lambda P : fluxes - f(P, mu), guess)

    return (vec, mu, fluxes) if return_fluxes else vec
