"""
Sub module for implementing various forms of broadening. Specifically this supports
Gaussian broadening for instrumental broadening, and various forms of rotational
broadening.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import gamma
from astropy.convolution import convolve
from .misc import gaussian

__all__ = [
    "convolve_gaussian",
    "rotational_broadening",
]

rt_pi = np.sqrt(np.pi)
gamma_ratios = [gamma((k+4)/4)/gamma((k+6)/4) for k in range(1, 5)]

def _next_pow_2(N_in):
    """
    Helper function for finding the first power of two greater than N_in
    """
    N_out = 1
    while N_out < N_in:
        N_out *= 2
    return N_out

def convolve_gaussian(x, y, FWHM):
    """
    Convolve spectrum with a Gaussian with FWHM by oversampling and using an
    FFT approach. Wavelengths are assumed to be sorted, but uniform spacing is
    not required. Will cause wrap-around at the end of the spectrum.
    """

    #oversample data by at least factor 10 (up to 20).
    xi = np.linspace(x[0], x[-1], _next_pow_2(10*len(x)))
    yi = interp1d(x, y)(xi)

    yg = gaussian(xi, x[0], fwhm=FWHM) #half gaussian
    yg += yg[::-1]
    yg /= np.sum(yg) #Norm kernel

    yiF, ygF = np.fft.fft(yi), np.fft.fft(yg)
    yic = np.fft.ifft(yiF * ygF).real

    return interp1d(xi, yic)(x)

def rotational_kernel(y, n, method, coefs):
    kx = np.linspace(-1+1/n, 1-1/n, n)
    ybar2 = 1-kx**2
    if method == 'flat':
        #Uniform brightness on stellar disc
        ky = np.sqrt(ybar2)
    elif method == 'rect':
        #rectangular kernel (box smoothing)
        ky = np.ones(n) 
    elif method == 'linear':
        #Linear limb darkening law
        eps = coefs[1]
        ky = 4*(1-eps) * np.sqrt(ybar2) + np.pi * eps * ybar2
    elif method == 'claret':
        #Claret 4-term limb darkening law
        a_k = coefs[1]
        ky = np.sqrt(ybar2) * (
            2*(1-a_k.sum()) + rt_pi*sum(a_k*gr * ybar2**(k/4) 
            for k, gr in enumerate(gamma_ratios))
        )
    else:
        raise ValueError("Invalid kernel type")
    return kx, ky/ky.sum()

def rotational_broadening(y, n, method, coefs):
    """
    Implements rotational broadening of a spectrum. The spectrum should
    be in flux units of lam*F(lam) or equivalently nu*F(nu) to ensure flux
    convservation. Similarly the fluxes should have been rebinned to an axis
    linear in log-wavelength, with spacing such that dloglam = 2*vsini/n.
    Method is the 
    """
    kx, ky = rotational_kernel(y, n, method, coefs)
    return convolve(y, ky)
