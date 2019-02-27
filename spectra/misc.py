import numpy as _np
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.special import wofz
from functools import reduce
import operator

__all__ = [
  "jangstrom",
  "voigt",
  "vac_to_air",
  "air_to_vac",
  "convolve_gaussian",
  "convolve_gaussian_R",
  "black_body",
  "sky_line_fwhm",
  "keep_points",
  "Lanczos",
]


jangstrom = \
  "$\mathrm{erg}\;\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$"

def voigt(x, x0, fwhm_g, fwhm_l):
  sigma = voigt.Va*fwhm_g
  z = ((x-x0) + 0.5j*fwhm_l)/(sigma*voigt.Vb)
  return wofz(z).real/(sigma*voigt.Vc)
voigt.Va = 1/(2*_np.sqrt(2*_np.log(2)))
voigt.Vb = _np.sqrt(2)
voigt.Vc = _np.sqrt(2*_np.pi)

def vac_to_air(Wvac):
  """
  converts vacuum wavelengths to air wavelengths,
  as per VALD3 documentation (in Angstroms)
  """
  s = 1e4/Wvac
  n = 1.0000834254 \
    + 0.02406147/(130.-s*s) \
    + 0.00015998/(38.9-s*s)
  return Wvac/n
#

def air_to_vac(Wair):
  """
  converts air wavelengths to vacuum wavelengths,
  as per VALD3 documentation (in Angstroms)
  """
  s = 1e4/Wair
  n = 1.00008336624212083 \
    + 0.02408926869968 / (130.1065924522-s*s) \
    + 0.0001599740894897/(38.92568793293-s*s)
  return Wair*n
#

def convolve_gaussian(x, y, FWHM):
  """
  Convolve spectrum with a Gaussian with FWHM by oversampling and
  using an FFT approach. Wavelengths are assumed to be sorted,
  but uniform spacing is not required. Will cause wrap-around at
  the end of the spectrum.
  """
  sigma = FWHM/2.355

  def next_pow_2(N_in):
    N_out = 1
    while N_out < N_in:
      N_out *= 2
    return N_out

  #oversample data by at least factor 10 (up to 20).
  xi = _np.linspace(x[0], x[-1], next_pow_2(10*len(x)))
  yi = interp1d(x, y)(xi)

  yg = _np.exp(-0.5*((xi-x[0])/sigma)**2) #half gaussian
  yg += yg[::-1]
  yg /= _np.sum(yg) #Norm kernel

  yiF = _np.fft.fft(yi)
  ygF = _np.fft.fft(yg)
  yic = _np.fft.ifft(yiF * ygF).real

  return interp1d(xi, yic)(x)
#

def convolve_gaussian_R(x, y, R):
  """
  Similar to convolve_gaussian, but convolves to a specified resolution
  rather than a specfied FWHM. Essentially this amounts to convolving
  along a log-uniform x-axis instead.
  """
  return convolve_gaussian(_np.log(x), y, 1./R)
#

def black_body(x, T, norm=True):
  """
  x in angstroms
  T in Kelvin
  returns un-normed spectrum
  """
  logf = _np.empty_like(x,dtype='float')
  Q = 143877516. /(x*T) # const. = ( h * c )/( 1e-10 * kB )
  lo = Q < 10.
  hi = ~lo
  #log form needed to stop overflow in x**-5
  #for Q>7. exp(Q)==expm1(Q) to better than 0.1%.
  logf[lo] = -5. * _np.log( x[lo] ) - _np.log( _np.expm1(Q[lo]) )
  logf[hi] = -5. * _np.log( x[hi] ) - Q[hi]
  if norm:
    logf -= logf.max() #normalise to peak at 1.
  return _np.exp( logf )
#

def sky_line_fwhm(S, x0, dx=5.):
  """
  Given a sky spectrum, this fits a Gaussian to a
  sky line and returns the FWHM.
  """
  def sky_residual(params, S):
    A, x0, fwhm, C = params
    xw = fwhm /2.355
    y_fit = A*_np.exp(-0.5*((S.x-x0)/xw)**2) + C
    return (S.y - y_fit)/S.e

  Sc = S.clip(x0-dx, x0+dx)
  guess = Sc.y.max(), x0, 2*_np.diff(Sc.x).mean(), Sc.y.min()
  res = leastsq(sky_residual, guess, args=(Sc,), full_output=True)
  vec, err = res[0], _np.sqrt(_np.diag(res[1]))

  return vec[2], err[2]
#

def keep_points(x, fname):
  """
  creates a mask for a spectrum that regions between pairs from a file
  """
  try:
    lines = open(fname,'r').readlines()
  except IOError:
    print("file %s does not exist" %fname)
    exit()
  between = lambda x, x1, x2: (x>float(x1))&(x<float(x2))
  segments = (between(x, *line.split()) for line in lines)
  return reduce(operator.or_, segments)

def Lanczos(x, y, xnew):
  n = _np.arange(len(x))
  Ni = interp1d(x, n, kind='linear', fill_value='extrapolate')(xnew)
  ynew = [_np.sum(y*_np.sinc(ni-n)) for ni in Ni]
  return _np.array(ynew)
