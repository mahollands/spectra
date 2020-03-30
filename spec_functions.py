"""
Contains functions for generating spectra or operating on spectra
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import h, c, k_B
from scipy.optimize import leastsq
from scipy.integrate import quad
from .spec_class import Spectrum

__all__ = [
  "Black_body",
  "JuraDisc",
  "join_spectra",
  "spectra_mean",
]

def Black_body(x, T, wave='air', x_unit="AA", y_unit="erg/(s cm2 AA)", norm=False):
  """
  Returns a Black body curve like black_body(), but the return value
  is a Spectrum class.
  """
  BB = Spectrum(x, 0., 0., f'{T}K BlackBody', wave, x_unit, "W/(m2 Hz)")
  BB.x_unit_to("Hz")
  nu = BB.x * u.Hz
  T *= u.K
  bb = 2*h*nu**3/(c**2*np.expm1(h*nu/(k_B*T)))
  BB += bb.to("W/(m2 Hz)")
  BB.x_unit_to(x_unit)
  BB.y_unit_to(y_unit)
  if norm:
    BB /= BB.y.max()
  return BB

#

def JuraDisc(x, Tstar, Rstar, Tin, Tout, D, inc):
  """
  Generates the irradiated disc model of Jura (2003).
  Inputs are:
  x: wavlength [AA]
  Tstar: stellar effective temperature [K]
  Rstar: stellar radius [Rsun]
  Tin: temperature of disc inner edge [K]
  Tout: temperature of disc outer edge [K]
  D: Distance to star from the Sun [pc]
  inc: inclination angle of disc [radians]
  """
  
  nu = (x * u.AA).to(u.Hz, equivalencies=u.spectral())
  Tstar <<= u.K
  Rstar <<= u.Rsun
  Tin <<= u.K
  Tout <<= u.K
  D <<= u.pc
  
  I = np.array([_JuraIntegral(nu_i, Tin, Tout) for nu_i in nu]).T[0]

  t1 = 12*np.pi**(1/3)
  t2 = np.cos(inc) * (Rstar/D)**2
  t3 = ((2*k_B*Tstar)/(3*h*nu))**(8/3)
  t4 = h*nu**3/c**2
  Fring = t1*t2*t3*t4*I
  S = Spectrum(x, Fring.value, 0., name="Disc", wave='vac', x_unit="AA", y_unit=Fring.unit)
  S.y_unit_to("erg/(s cm2 AA)")
  return S

def _JuraIntegral(nu, Tin, Tout):
  Xin, Xout = [(h*nu/(k_B*T)).si.value for T in (Tin, Tout)]
  return quad(lambda X: X**(5/3)/np.expm1(X), Xin, Xout)
    

#..............................................................................

def join_spectra(SS, sort=False, name=None):
  """
  Joins a collection of spectra into a single spectrum. The name of the first
  spectrum is used as the new name. Can optionally sort the new spectrum by
  wavelengths.
  """
  S0 = SS[0]
  
  for S in SS:
    if not isinstance(S, Spectrum):
      raise TypeError('item is not Spectrum')
    if S.wave != S0.wave:
      raise ValueError("Spectra must have same wavelengths")
    S._compare_units(S0, xy='xy')

  x = np.hstack([S.x for S in SS])
  y = np.hstack([S.y for S in SS])
  e = np.hstack([S.e for S in SS])
  S = Spectrum(x, y, e, **S0.info)
  
  if name is not None:
    S.name = name
  if sort:
    S = S[np.argsort(x)]

  return S

def spectra_mean(SS):
  """
  Calculate the weighted mean spectrum of a list/tuple of spectra.
  All spectra should have identical wavelengths.
  """
  S0 = SS[0]
  for S in SS:
    if not isinstance(S, Spectrum):
      raise TypeError('item is not Spectrum')
    S._compare_units(S0, xy='xy')
    S._compare_x(S0)

  Y  = np.array([S.y    for S in SS])
  IV = np.array([S.ivar for S in SS])

  IVbar = np.sum(IV, axis=0)
  Ybar  = np.sum(Y*IV, axis=0) / IVbar
  Ebar  = 1.0 / np.sqrt(IVbar)

  return Spectrum(S0.x, Ybar, Ebar, **S0.info)

def sky_line_fwhm(S, x0, dx=5., return_model=False):
  """
  Given a sky spectrum, this fits a Gaussian to a
  sky line and returns the FWHM. The window width
  is 2*dx wide, centred on x0. The 
  """
  def sl_model(params, S):
    x0, fwhm, A, M, C = params
    xw = fwhm /2.355
    xx = S.x-x0
    ymod = A*np.exp(-0.5*(xx/xw)**2) + M*xx + C
    info = S.info
    info.pop('name')
    info.pop('head')
    return Spectrum(S.x, ymod, 0, **info)
    
  #intial pass to refine center
  Sc = S.clip(x0-dx, x0+dx)
  x0 = Sc.x[np.argmax(Sc.y)]
  Sc = S.clip(x0-dx, x0+dx)

  guess = (
    x0,
    2*Sc.dx.mean(), #fwhm ~2pixels
    Sc.y.max()-Sc.y.min(), #A
    (S.y[-1]-S.y[0])/(S.x[-1]-S.x[0]), #M
    0.5*(S.y[-1]+S.y[0]) #C
  )
  res = leastsq(lambda p, S: (S - sl_model(p, S)).y_e, guess, args=(Sc,), full_output=True)
  try:
    vec, err = res[0], np.sqrt(np.diag(res[1]))
  except ValueError:
    print(f"Fit did not converge for line at wavelength {x0}")
    return (None, None) if return_model else None

  Pnames = "x0 fwhm A M C".split()
  res = {p : (v, e) for p, v, e in zip(Pnames, vec, err)}

  return (res, sl_model(vec, Sc)) if return_model else res
#
