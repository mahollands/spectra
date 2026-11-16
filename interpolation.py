"""
Routines for interpolating, and rebinning data
"""
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d, Akima1DInterpolator

__all__ = [
    "interp",
    "interp_nan",
    "interp_inf",
    "wbin",
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
        yi = interp1d(x, y,
            kind=kind,
            bounds_error=False,
            fill_value=0.,
            **kwargs,
        )(xi)
        if not model:
            #If any errors were inf (zero weight) we need to make
            #sure the interpolated range stays inf
            inf = np.isinf(e)
            if np.any(inf):
                ei = interp1d(x[~inf], e[~inf],
                    kind=kind,
                    bounds_error=False,
                    fill_value=np.inf,
                    **kwargs,
                )(xi)
                inf2 = interp1d(x, inf,
                    kind='nearest',
                    bounds_error=False,
                    fill_value=0,
                    **kwargs,
                )(xi).astype(bool)
                ei[inf2] = np.inf
            else:
                ei = interp1d(x, e,
                    kind=kind,
                    bounds_error=False,
                    fill_value=np.inf,
                    **kwargs
                )(xi)

    ei = 0 if model else np.where(ei < 0, 0, ei)
    return yi, ei


def interp_nan(x, y, e, interp_e):
    """
    Linearly interpolate over values with NaNs. If interp_e is set, these
    are also interpolated over, otherwise they are set to np.inf.
    """
    bad = np.isnan(y) | np.isnan(e)
    kwargs = {'kind': 'linear', 'bounds_error': False}
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


def wbin(xin, yin, xout, kind):
    """
    Rebin fluxes onto the xout axis, conserving flux. Based on the Molly
    REBIN routine. The input spectrum is treated as a piecewise function
    (constant for kind='linear', quadratic for kind='quadratic') whose
    integral over each input pixel equals that pixel's value. This function
    is integrated between the output pixel edges, then divided by the
    width of each output pixel (in input pixels).
    """
    kind = kind.lower()
    if kind in {'l', 'lin', 'linear'}:
        quadratic = False
    elif kind in {'q', 'quad', 'quadratic'}:
        quadratic = True
    else:
        raise ValueError("kind must be a form of linear/quadratic")

    #Smooth pixel -> wavelength mappings for both axes
    arc_in, npix = get_spec_arc(xin)
    arc_out, nout = get_spec_arc(xout)

    #Output pixel edges, converted to (0-indexed) input pixel coordinates
    edges_wave = arc_out((np.arange(nout+1) + 0.5) / nout)
    edges_px = invert_arc(arc_in, npix, edges_wave) - 1

    cumflux = cumulative_flux(yin, edges_px, quadratic)
    return np.diff(cumflux) / np.diff(edges_px)


def invert_arc(arc, npix, wave, acc=1e-4):
    """
    Find the (1-indexed) pixel positions corresponding to wavelengths wave,
    for an arc fitted by get_spec_arc. A linear interpolation of the arc
    provides the starting guess, which is refined with Newton's method until
    converged to within acc pixels.
    """
    px = np.arange(1, npix+1)
    arc_px = arc(px/npix)
    order = np.argsort(arc_px)
    p = np.interp(wave, arc_px[order], px[order])

    darc = arc.deriv()
    for _ in range(100):
        dp = npix * (wave - arc(p/npix)) / darc(p/npix)
        p += dp
        if np.all(np.abs(dp) < acc):
            break
    return p


def cumulative_flux(y, p, quadratic):
    """
    Integral of the piecewise interpolant of y from an arbitrary origin up
    to (0-indexed) pixel positions p. Pixel j spans j-0.5 to j+0.5. Beyond
    the ends of y, the edge values are extended.
    """
    j = np.rint(p).astype(int)
    t = p - j #position within pixel j, in [-0.5, 0.5]

    #Pad with edge values so every pixel (and its neighbours) exists
    lo = max(0, -j.min()) + 1
    hi = max(0, j.max() - (len(y)-1)) + 1
    ypad = np.pad(y, (lo, hi), mode='edge')
    j += lo

    #Total flux in all pixels before pixel j
    flux_before = np.concatenate(([0.], np.cumsum(ypad)))[j]

    Y = ypad[j]
    if quadratic:
        #Parabola f(t) = 3At^2 + 2Bt + C, constrained so its integral over
        #each of pixels j-1, j, j+1 matches the respective pixel values
        Yl, Yr = ypad[j-1], ypad[j+1]
        mean = 0.5*(Yl + Yr)
        A = (mean - Y) / 3
        B = (Yr - Yl) / 4
        C = (13*Y - mean) / 12
        partial = ((A*t + B)*t + C)*t + A/8 - B/4 + C/2
    else:
        partial = Y * (t + 0.5)
    return flux_before + partial


def get_spec_arc(x):
    """
    Fits an arc to a set of wavelengths including increasingly high order terms
    until satisfactory convergence. If an 8th order polynomial still fails, a
    ValueError is raised.
    """
    #Check px scale is smooth
    npx = len(x)
    px = 1+np.arange(npx)
    for deg in range(1, 9):
        arc = Polynomial.fit(px/npx, x, deg=deg)
        c = arc(px/npx)
        diff = (c-x)[1:]/np.diff(x)
        maxdiff = np.max(abs(diff))
        #poly fitted pixels within 1% of pixel width from data
        if maxdiff < 0.01:
            break
    else:
        raise ValueError("Could not adequately fit wavelengths(pixels)")
    return arc, npx
