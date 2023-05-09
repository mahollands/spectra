"""
Sub-module for synthetic photometry of spectra
"""
import os.path
import sys
import functools
import numpy as np
from scipy.integrate import trapz as Itrapz
from scipy.interpolate import interp1d

__all__ = [
    "get_transmission_curve",
    "calc_AB_flux",
    "filter_names",
]

filters_dir = "{}/passbands".format(os.path.dirname(__file__))

GAIA = {'G':'G', 'Bp':'Gbp', 'Rp':'Grp'}
JPLUS = "gSDSS iSDSS J0378 J0395 J0410 J0430 J0515 J0660 J0861 rSDSS uJAVA zSDSS".split()
HCAM = list("ugriz") + [b+"_s" for b in "ugriz"]
UCAM = list("ugriz") + [b+"_s" for b in "ugriz"] + "iz Ha_broad Ha_narrow".split()
USPEC = list("ugriz") + "iz Ha_broad Ha_narrow bowen KG5 N86 NaI".split()
SWIFT = ("U", "UVW1", "UVW2", "UVM2")

filter_paths = {
    **{f"2m{b}": f"2MASS/2MASS_2MASS.{b}.npy" for b in "JHK"}, #2Mass
    **{f"Gaia{b}3": f"GAIA/GAIA_GAIA3.{GAIA[b]}.npy" for b in GAIA}, #Gaia
    **{f"Gaia{b}2r": f"GAIA/GAIA_GAIA2r.{GAIA[b]}.npy" for b in GAIA}, #Gaia
    **{f"Galex{b}": f"GALEX/GALEX_GALEX.{b}.npy" for b in ("NUV", "FUV")}, #GALEX
    **{b: f"GENERIC/Generic_Johnson.{b}.npy" for b in "UBVRI"}, #Generic Johnson
    **{f"JPLUS-{b}": f"JPLUS/OAJ_JPLUS.{b}.npy" for b in JPLUS}, #JPLUS
    **{f"ps{b}": f"PANSTARRS/PAN-STARRS_PS1.{b}.npy" for b in "grizy"}, #PanStarrs
    **{f"SDSS{b}": f"SDSS/SLOAN_SDSS.{b}.npy" for b in "ugriz"}, #SDSS
    **{f"sm{b}": f"SKYMAPPER/SkyMapper_SkyMapper.{b}.npy" for b in "uvgriz"}, #SkyMapper
    **{f"S{b}": f"SPITZER/Spitzer_IRAC.I{b}.npy" for b in "12"}, #Spitzer
    **{f"sw{b}": f"SWIFT/Swift_UVOT.{b}.npy" for b in SWIFT}, #Swift
    **{f"UK{b}": f"UKIRT/UKIRT_UKIDSS.{b}.npy" for b in "ZYJHK"}, #UKIRT
    **{f"W{b}": f"WISE/WISE_WISE.W{b}.npy" for b in "12"}, #Wise
    **{f"UCAM_{b}": f"ULTRACAM/WHT_ULTRACAM.{b}.npy" for b in UCAM}, #ULTRACAM
    **{f"USPEC_{b}": f"ULTRASPEC/TNT_ULTRASPEC.{b}.npy" for b in USPEC}, #ULTRASPEC
    **{f"HCAM_{b}": f"HIPERCAM/GTC_HIPERCAM.{b}.npy" for b in HCAM}, #HIPERCAM
}
filter_names = list(filter_paths)

@functools.cache
def get_transmission_curve(band):
    """
    Loads filter curves obtained from VOSA (SVO).
    """
    try:
        full_path = f"{filters_dir}/{filter_paths[band]}"
    except KeyError:
        print(f'Invalid filter name: {band}')
        sys.exit()
    x, y = np.load(full_path)
    R = interp1d(x, y, kind='linear', assume_sorted=True)
    Inorm = Itrapz(R.y/R.x, R.x)
    return R, Inorm

def load_Vega(mod="002"):
    """
    Loads a Kurucz Vega model. The newer 003 is provided, but Gaia
    uses 002, so this is used by default.
    """
    from .spec_io import spec_from_npy

    if mod not in {"002", "003"}:
        raise ValueError("Vega model must be 002 or 003")

    full_path = f"{filters_dir}/alpha_lyr_mod_{mod}.npy"

    Vega = spec_from_npy(full_path, wave='vac')

    #CALSPEC correction at 550nm (V=0.023)
    px = Vega.closest_x(5500)
    Vega *= 3.62286e-09 / Vega.y[px]

    return Vega

def calc_AB_flux(S, band, Nmc=1000, Ifun=Itrapz):
    """
    Calculates the synthetic AB flux (Jy) of a spectrum for a given bandpass.
    If Nmc is > 0, monte-carlo error propagation is performed outputting
    both a synthetic-mag and error. For model-spectra, i.e. no errors,
    use e=np.ones_like(f) and Nmc=0. List of currently supported filters:

    2Mass:     ['2m(JHK)']

    Denis:     ['DenisI']

    Gaia:      ['Gaia(G,Bp,Rp)']

    Galex:     ['Galex(FUV,NUV)']

    Johnson:   ['U','B','V','R','I']

    PanSTARRS: ['ps(grizy)']

    SDSS:      ['SDSS(ugriz)']

    Skymapper: ['sm(uvgriz)']

    Spitzer:   ['S(12)']

    Swift:     ['sw(U,UVW1,UVW2,UVM1)']

    WISE:      ['W(12)']
    """

    #Need specific units for integrals
    if S.x_unit != "AA":
        S.x_unit_to("AA")
    if S.y_unit != "Jy":
        S.y_unit_to("Jy")

    #clip data to filter range and interpolate filter to data axis
    R, Inorm = get_transmission_curve(band)
    S = S.clip(R.x[0], R.x[-1])
    Ri = R(S.x)

    #Calculate AB fluxes, or MC sampled fluxes
    if Nmc == 0:
        return Ifun(S.y*Ri/S.x, S.x)/Inorm

    return np.array([Ifun(y_mc*Ri/S.x, S.x) for y_mc in S.y_mc(Nmc)])/Inorm

def lambda_mean(band, Ifun=Itrapz):
    """
    Calculates lambda_mean for one of the filters
    """
    R, _ = get_transmission_curve(band)
    return Ifun(R.y*R.x, R.x) / Ifun(R.y, R.x)

def lambda_eff(band, Ifun=Itrapz):
    """
    Calculates lambda_eff for one of the filters, integrated over the spectrum
    of Vega.
    """
    R, _ = get_transmission_curve(band)
    V = load_Vega().interp(R.x)
    return Ifun(R.y*V.y*R.x, R.x) / Ifun(R.y*V.y, R.x)
