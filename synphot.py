import numpy as np
from scipy.integrate import trapz as Itrapz, simps as Isimps
import os.path

__all__ = [
  "load_transmission_curve",
  "mag_calc_AB",
  "filter_names",
]

filters_dir = "{}/filt_profiles".format(os.path.dirname(__file__))

GaiaDict = {'G':'G', 'Bp':'Gbp', 'Rp':'Grp'}
JPLUS = "gSDSS iSDSS J0378 J0395 J0410 J0430 J0515 J0660 J0861 rSDSS uJAVA zSDSS".split()
filter_paths = {
  **{f"2m{b}"    : f"2MASS/2MASS_2MASS.{b}.npy" for b in "JHK"}, #2Mass
  **{f"Denis{b}" : f"DENIS/DENIS_DENIS.{b}.npy" for b in "I"}, #DENIS
  **{f"Gaia{b}"  : f"GAIA/GAIA_GAIA2r.{GaiaDict[b]}.npy" for b in ("G","Bp","Rp")}, #Gaia
  **{f"Galex{b}" : f"GALEX/GALEX_GALEX.{b}.npy" for b in ("NUV", "FUV")}, #GALEX
  **{b           : f"GENERIC/Generic_Johnson.{b}.npy" for b in "UBVRI"}, #Generic Johnson
  **{f"JPLUS-{b}": f"JPLUS/OAJ_JPLUS.{b}.npy" for b in JPLUS}, #JPLUS
  **{f"ps{b}"    : f"PANSTARRS/PAN-STARRS_PS1.{b}.npy" for b in "grizy"}, #PanStarrs
  **{b           : f"SDSS/SLOAN_SDSS.{b}.npy" for b in "ugriz"}, #SDSS
  **{f"sm{b}"    : f"SKYMAPPER/SkyMapper_SkyMapper.{b}.npy" for b in "uvgriz"}, #SkyMapper
  **{f"S{b}"     : f"SPITZER/Spitzer_IRAC.I{b}.npy" for b in "12"}, #Spitzer
  **{f"sw{b}"    : f"SWIFT/Swift_UVOT.{b}.npy" for b in ("U","UVW1","UVW2","UVM2")}, #Swift
  **{f"UK{b}"    : f"UKIRT/UKIRT_UKIDSS.{b}.npy" for b in "ZYJHK"}, #UKIRT
  **{f"W{b}"     : f"WISE/WISE_WISE.W{b}.npy" for b in "12"}, #Wise
}
filter_names = list(filter_paths)

loaded_filters = {}

def load_transmission_curve(filt):
  """
  Loads the filter curves obtained from VOSA (SVO).
  """
  from .spec_io import spec_from_npy

  try:
    full_path = "{}/{}".format(filters_dir, filter_paths[filt])
  except KeyError:
    print('Invalid filter name: {}'.format(filt))
    exit()

  return spec_from_npy(full_path, wave="vac", x_unit="AA", y_unit="")
#

def load_Vega():
  """
  Loads the filter curves obtained from VOSA (SVO).
  """
  from .spec_io import spec_from_npy

  full_path = "{}/alpha_lyr_mod_003.npy".format(filters_dir)

  return spec_from_npy(full_path, wave='vac', x_unit="AA", y_unit="")
#

def m_AB_int(X, Y, R, Ifun):
  y_nu = Ifun(Y*R/X, X)/Ifun(R/X, X) 
  m = -2.5 * np.log10(y_nu) + 8.90
  return m

def mag_calc_AB(S, filt, NMONTE=1000, Ifun=Itrapz):
  """
  Calculates the synthetic AB magnitude of a spectrum for a given filter.
  If NMONTE is > 0, monte-carlo error propagation is performed outputting
  both a synthetic-mag and error. For model-spectra, i.e. no errors,
  use e=np.ones_like(f) and NMONTE=0. List of currently supported filters:

  2Mass:     ['2mJ','2mH','2mK']

  Denis:     ['DenisI']

  Gaia:      ['Gaia(G,Bp,Rp)']

  Galex:     ['GalexFUV' 'GalexNUV']

  Johnson:   ['U','B','V','R','I']

  PanSTARRS: ['ps(grizy)']

  SDSS:      ['u','g','r','i','z']

  Skymapper: ['sm(uvgriz)']

  Spitzer:   ['S1','S2']

  Swift:     ['sw(U,UVW1,UVW2,UVM1)']

  WISE:      ['W1','W2']
  """

  #load filter
  if filt not in loaded_filters:
    loaded_filters[filt] = load_transmission_curve(filt)
  R  = loaded_filters[filt]
  R.wave = S.wave

  #Convert Spectra/filter-curve to Hz/Jy for integrals
  R.x_unit_to("Hz")
  S.x_unit_to("Hz")
  S.y_unit_to("Jy")

  #clip data to filter range and interpolate filter to data axis
  S = S.clip(np.min(R.x), np.max(R.x))
  R = R.interp(S, kind='linear')

  #Calculate AB magnitudes, potentially including flux errors
  if NMONTE == 0:
    return m_AB_int(S.x, S.y, R.y, Ifun)
  else:
    y_mc = lambda S: np.random.normal(S.y, S.e)
    m = np.array([m_AB_int(S.x, y_mc(S), R.y, Ifun) for i in range(NMONTE)])
    return np.mean(m), np.std(m)
#

def lambda_mean(filt, Ifun=Itrapz):
  """
  Calculates lambda_mean for one of the filters
  """
  R = load_transmission_curve(filt)
  return Ifun(R.y*R.x, R.x) / Ifun(R.y, R.x)
#

def lambda_eff(filt, Ifun=Itrapz):
  """
  Calculates lambda_eff for one of the filters, integrated
  over the spectrum of Vega.
  """
  R = load_transmission_curve(filt)
  V = load_Vega().interp(R)
  return Ifun(R.y*V.y*R.x, R.x) / Ifun(R.y*V.y, R.x)
#
