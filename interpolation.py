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

def wbin(xin, yin, xout, kind):
    """
    Rebin onto fluxes onto xout axis. Based on Molly rebin routine
    """
    #Check x-in px scale is smooth, and get arc coefs
    arc1, npix = get_spec_arc(xin)
    arc2, nout = get_spec_arc(xout)

    acc = 1e-4/npix
    rnout = 1.0/nout
    yout = np.zeros(nout)

    rx2 = 0.5*rnout
    darc1 = arc1.copy()
    darc1.cutdeg(len(darc1)-2)
    darc1.coef *= np.arange(len(darc1))[::-1]
    rslope = 1/darc1(0.2)
    x1 = 0.2
    rx2, x1 = wbin_tform(rx2, x1, arc1, darc1, arc2, rslope, acc)
    x1 *= npix
    dx = 0
    nsgn = 1 if (arc2(1.0)-arc2.coef[0])*rslope >= 0 else -1
    nstop = nout

    j1 = round(float(x1))
    for k in range(nstop):
        rx2 += rnout
        x2 = (x1+dx)/npix
        rx2, x2 = wbin_tform(rx2, x2, arc1, darc1, arc2, rslope, acc)
        x2 *= npix
        dx = x2-x1
        j2 = round(float(x2))
        d = 0
        if kind.capitalize() in {'L','Lin','Linear'}:
            if k == 0:
                M1 = max(min(j1, npix), 1)-1
                dd = (nsgn*(j1-x1)-0.5)*yin[M1]
            M2 = max(min(j2, npix), 1)-1
            d += dd
            ddd = yin[M2]
            dd = ddd*(nsgn*(j2-x2)-0.5)
            d -= dd+ddd
        elif kind.capitalize() in {'Q','Quad','Quadratic'}:
            if k == 0:
                M1, M2, M3 = [max(min(j1+i, npix), 1)-1 for i in (-1, 0, 1)]
                A = 0.5*(yin[M1]+yin[M3])
                B = (A-yin[M1])*0.5
                C = (13.0*yin[M2]-A)/12.0
                A = (A-yin[M2])/3.0
                Y = x1-j1
                dd = nsgn*(((A*Y+B)*Y+C)*Y-B*0.25) + A*0.125 + C*0.5

            M1, M2, M3 = [max(min(j2+i, npix), 1)-1 for i in (-1, 0, 1)]
            A = 0.5*(yin[M1]+yin[M3])
            B = (A-yin[M1])*0.5
            C = (13.0*yin[M2]-A)/12.0
            A = (A-yin[M2])/3.0
            y = x2-j2
            d -= dd
            dd = nsgn*(((A*y+B)*y+C)-B*0.25)
            ddd = A*0.125 + C*0.5
            d += dd-ddd
            dd += ddd
        else:
            raise ValueError("kind must be a form of linear/quadratic")
        idx = np.array([max(min(kk, npix-1), 0) for kk in range(j1, j2, nsgn)])
        d += np.sum(yin[idx])
        yout[k] = d/abs(dx)
        x1 = x2
        j1 = j2
    return yout

def wbin_tform(rx, x, arc1, darc1, arc2, rslope, acc):
    """
    Routine based on WBIN_TFORM from Molly.
    """
    rl = arc2(rx)
    for n in range(101):
        l = arc1(x)
        dx = (rl-l)*rslope
        x += dx
        if abs(dx) < acc: break
        rslope=1/darc1(x)
    return rx, x

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
