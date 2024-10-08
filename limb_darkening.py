"""
Routines for calculating limb darkening coefficients and central intensities
"""
import numpy as np

__all__ = [
    "calc_limb_darkening_coefs",
]

# See Claret et al., A&A 634, 93 (2020)
limb_functions = {
    'linear': lambda P, mu: 1-P[0]*(1-mu),
    'quad': lambda P, mu: 1-sum(a*(1-mu)**i for i, a in enumerate(P, 1)),
    'sqrt': lambda P, mu: 1-P[0]*(1-mu)-P[1]*(1-np.sqrt(mu)),
    'log': lambda P, mu: 1-P[0]*(1-mu)-P[1]*mu*np.log(mu),
    'claret': lambda P, mu: 1-sum(a_k*(1-mu**(k/2)) for k, a_k in enumerate(P, 1)),
}

limb_basis = {
    'linear': lambda mu: [(1-mu)],
    'quad': lambda mu: [(1-mu)**n for n in (1, 2)],
    'sqrt': lambda mu: [(1-mu), (1-np.sqrt(mu))],
    'log': lambda mu: [(1-mu), (mu*np.log(mu))],
    'claret': lambda mu: [(1-mu**(k/2)) for k in range(1, 5)],
}


def calc_limb_darkening_coefs(MM, band, limb_model='claret', return_fluxes=False):
    """
    Calculate limb darkening coefficients for DK models for a given pandpass.
    MM is a list of models containing angular dependent spectra as a function
    of mu (mu values are header items). Fluxes (in Jy/sr) and mu points can be
    returned if 'return_fluxes' is true.
    """
    mu_i = np.array([M.head['mu'] for M in MM])
    fluxes = np.array([M.flux_calc_AB(band) for M in MM])
    y_i = 1 - fluxes/np.max(fluxes)
    basis = np.array(limb_basis[limb_model](mu_i))
    vec = np.linalg.solve(basis @ basis.T, basis @ y_i)

    return (vec, mu_i, fluxes) if return_fluxes else vec
