"""
Sub-module for synthetic photometry of spectra
"""
import os.path
import sys
import numpy as np
from scipy.integrate import trapz as Itrapz

__all__ = [
    "load_transmission_curve",
    "calc_AB_flux",
    "filter_names",
]

filters_dir = "{}/passbands".format(os.path.dirname(__file__))

GaiaDict = {'G':'G', 'Bp':'Gbp', 'Rp':'Grp'}
JPLUS = "gSDSS iSDSS J0378 J0395 J0410 J0430 J0515 J0660 J0861 rSDSS uJAVA zSDSS".split()
filter_paths = {
    **{f"2m{b}"    : f"2MASS/2MASS_2MASS.{b}.npy" for b in "JHK"}, #2Mass
    **{f"Denis{b}" : f"DENIS/DENIS_DENIS.{b}.npy" for b in "I"}, #DENIS
    **{f"Gaia{b}3"  : f"GAIA/GAIA_GAIA3.{GaiaDict[b]}.npy" for b in ("G", "Bp", "Rp")}, #Gaia
    **{f"Gaia{b}2r"  : f"GAIA/GAIA_GAIA2r.{GaiaDict[b]}.npy" for b in ("G", "Bp", "Rp")}, #Gaia
    **{f"Galex{b}" : f"GALEX/GALEX_GALEX.{b}.npy" for b in ("NUV", "FUV")}, #GALEX
    **{b           : f"GENERIC/Generic_Johnson.{b}.npy" for b in "UBVRI"}, #Generic Johnson
    **{f"JPLUS-{b}": f"JPLUS/OAJ_JPLUS.{b}.npy" for b in JPLUS}, #JPLUS
    **{f"ps{b}"    : f"PANSTARRS/PAN-STARRS_PS1.{b}.npy" for b in "grizy"}, #PanStarrs
    **{b           : f"SDSS/SLOAN_SDSS.{b}.npy" for b in "ugriz"}, #SDSS
    **{f"sm{b}"    : f"SKYMAPPER/SkyMapper_SkyMapper.{b}.npy" for b in "uvgriz"}, #SkyMapper
    **{f"S{b}"     : f"SPITZER/Spitzer_IRAC.I{b}.npy" for b in "12"}, #Spitzer
    **{f"sw{b}"    : f"SWIFT/Swift_UVOT.{b}.npy" for b in ("U", "UVW1", "UVW2", "UVM2")}, #Swift
    **{f"UK{b}"    : f"UKIRT/UKIRT_UKIDSS.{b}.npy" for b in "ZYJHK"}, #UKIRT
    **{f"W{b}"     : f"WISE/WISE_WISE.W{b}.npy" for b in "12"}, #Wise
}
filter_names = list(filter_paths)

loaded_filters = {}

def load_transmission_curve(band):
    """
    Loads the filter curves obtained from VOSA (SVO).
    """
    from .spec_io import spec_from_npy

    try:
        full_path = "{}/{}".format(filters_dir, filter_paths[band])
    except KeyError:
        print('Invalid filter name: {}'.format(band))
        sys.exit()

    return spec_from_npy(full_path, wave="vac", x_unit="AA", y_unit="")
#

def load_Vega():
    """
    Loads the Vega model
    """
    from .spec_io import spec_from_npy

    full_path = "{}/alpha_lyr_mod_003.npy".format(filters_dir)

    return spec_from_npy(full_path, wave='vac', x_unit="AA", y_unit="")
#

def calc_AB_flux(S, band, Nmc=1000, Ifun=Itrapz):
    """
    Calculates the synthetic AB flux (Jy) of a spectrum for a given bandpass.
    If Nmc is > 0, monte-carlo error propagation is performed outputting
    both a synthetic-mag and error. For model-spectra, i.e. no errors,
    use e=np.ones_like(f) and Nmc=0. List of currently supported filters:

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

    #load filters (only once per runtime)
    if band not in loaded_filters:
        loaded_filters[band] = load_transmission_curve(band)
    R = loaded_filters[band]
    R.wave = S.wave
    Inorm = Ifun(R.y/R.x, R.x)

    #Need specific units for integrals
    R.x_unit_to("Hz")
    S.x_unit_to("Hz")
    S.y_unit_to("Jy")

    #clip data to filter range and interpolate filter to data axis
    S = S.clip(np.min(R.x), np.max(R.x))

    #Calculate AB fluxes, or MC sampled fluxes
    R = R.interp(S, kind='linear')
    if Nmc == 0:
        return Ifun(S.y*R.y/S.x, S.x)/Inorm

    return np.array([Ifun(y_mc*R.y/S.x, S.x) for y_mc in S.y_mc(Nmc)])/Inorm
#

def lambda_mean(band, Ifun=Itrapz):
    """
    Calculates lambda_mean for one of the filters
    """
    R = load_transmission_curve(band)
    return Ifun(R.y*R.x, R.x) / Ifun(R.y, R.x)
#

def lambda_eff(band, Ifun=Itrapz):
    """
    Calculates lambda_eff for one of the filters, integrated
    over the spectrum of Vega.
    """
    R = load_transmission_curve(band)
    V = load_Vega().interp(R)
    return Ifun(R.y*V.y*R.x, R.x) / Ifun(R.y*V.y, R.x)
#
