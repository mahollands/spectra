"""
Sub-module for synthetic photometry of spectra. Currently supported telescopes/intruments:

    2Mass
    Gaia DR2r/3
    Galex
    HiperCam
    JPLUS
    Johnson (GENERIC)
    PanSTARRS
    SDSS
    Skymapper
    Spitzer
    Swift
    TESS
    UltraCAM
    UKIRT/UKIDSS
    UltraSpec
    WISE
    XMM

    See 'filter_names' list for details.
"""
import os.path
import functools
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

__all__ = [
    "load_bandpass",
    "calc_AB_flux",
    "filter_names",
    "Vega_AB_mag_offset",
]

filters_dir = "{}/passbands".format(os.path.dirname(__file__))

GAIA = {'G':'G', 'Bp':'Gbp', 'Rp':'Grp'}
JPLUS = "gSDSS iSDSS J0378 J0395 J0410 J0430 J0515 J0660 J0861 rSDSS uJAVA zSDSS".split()
HCAM = list("ugriz") + [b+"_s" for b in "ugriz"]
UCAM = list("ugriz") + [b+"_s" for b in "ugriz"] + "Ha_broad-1 Ha_narrow-1".split()
USPEC = list("ugriz") + "iz Ha_broad Ha_narrow bowen KG5 N86 NaI".split()
SWIFT = ("U", "UVW1", "UVW2", "UVM2")
XMM = ("U", "B", "V", "UVM2", "UVM1", "UVW1")

filter_paths = {
    **{f"2m{b}": f"2MASS/2MASS_2MASS.{b}.npy" for b in "JHK"}, #2Mass
    **{f"Gaia{b}3": f"GAIA/GAIA_GAIA3.{GAIA[b]}.npy" for b in GAIA}, #Gaia DR3
    **{f"Gaia{b}2r": f"GAIA/GAIA_GAIA2r.{GAIA[b]}.npy" for b in GAIA}, #Gaia DR2r
    **{f"Galex{b}": f"GALEX/GALEX_GALEX.{b}.npy" for b in ("NUV", "FUV")}, #GALEX
    **{b: f"GENERIC/Generic_Johnson.{b}.npy" for b in "UBVRI"}, #Generic Johnson
    **{f"HCAM_{b}": f"HIPERCAM/GTC_HIPERCAM.{b}.npy" for b in HCAM}, #HIPERCAM
    **{f"JPLUS-{b}": f"JPLUS/OAJ_JPLUS.{b}.npy" for b in JPLUS}, #JPLUS
    **{f"ps{b}": f"PANSTARRS/PAN-STARRS_PS1.{b}.npy" for b in "grizy"}, #PanStarrs
    **{f"SDSS{b}": f"SDSS/SLOAN_SDSS.{b}.npy" for b in "ugriz"}, #SDSS
    **{f"sm{b}": f"SKYMAPPER/SkyMapper_SkyMapper.{b}.npy" for b in "uvgriz"}, #SkyMapper
    **{f"sp{b}": f"SPITZER/Spitzer_IRAC.I{b}.npy" for b in "12"}, #Spitzer
    **{f"sw{b}": f"SWIFT/Swift_UVOT.{b}.npy" for b in SWIFT}, #Swift
    "TESS": "TESS/TESS_TESS.Red.npy", #TESS
    **{f"UCAM_{b}": f"ULTRACAM/TNO_ULTRACAM.{b}.npy" for b in UCAM}, #ULTRACAM
    **{f"UK{b}": f"UKIRT/UKIRT_UKIDSS.{b}.npy" for b in "ZYJHK"}, #UKIRT
    **{f"USPEC_{b}": f"ULTRASPEC/TNT_ULTRASPEC.{b}.npy" for b in USPEC}, #ULTRASPEC
    **{f"W{b}": f"WISE/WISE_WISE.W{b}.npy" for b in "12"}, #Wise
    **{f"XMM_{b}": f"XMM/XMM_OM.{b}.npy" for b in XMM}, #XMM
}
filter_names = list(filter_paths)

@functools.cache
def load_bandpass(band):
    """
    Loads filter curves obtained from VOSA (SVO).
    """
    if band not in filter_paths:
        raise ValueError(f'Invalid filter name: {band}')
    x, y = np.load(f"{filters_dir}/{filter_paths[band]}")
    R = interp1d(x, y, kind='linear', assume_sorted=True)
    Inorm = trapezoid(R.y/R.x, R.x)
    return R, Inorm

@functools.cache
def load_Vega(mod="002"):
    """
    Loads a Kurucz Vega model. The newer 003 is available, but Gaia uses 002,
    so this is used by default.
    """
    from .spec_class import Spectrum

    if mod not in {"002", "003"}:
        raise ValueError("Vega model must be 002 or 003")

    full_path = f"{filters_dir}/alpha_lyr_mod_{mod}.npy"
    Vega = Spectrum.from_npy(full_path, wave='vac')

    #CALSPEC correction at 550nm (V=0.023)
    px = Vega.closest_x(5500)
    Vega *= 3.62286e-09 / Vega.y[px]

    return Vega

def calc_AB_flux(S, band, Nmc=1000, Ifun=trapezoid):
    """
    Calculates the synthetic AB flux (Jy) of a spectrum for a given bandpass.
    If Nmc is > 0, monte-carlo error propagation is performed outputting
    both a synthetic-mag and error. For model-spectra, i.e. no errors,
    use e=np.ones_like(f) and Nmc=0.
    """

    #Need specific units for integrals
    if S.x_unit != "AA":
        S.x_unit_to("AA")
    if S.y_unit != "Jy":
        S.y_unit_to("Jy")

    #clip data to filter range and interpolate filter to data axis
    R, Inorm = load_bandpass(band)
    S = S.clip(R.x[0], R.x[-1])
    Ri = R(S.x)

    #Calculate AB fluxes, or MC sampled fluxes
    if Nmc == 0:
        return Ifun(S.y*Ri/S.x, S.x)/Inorm

    return np.array([Ifun(y_mc*Ri/S.x, S.x) for y_mc in S.y_mc(Nmc)])/Inorm

@functools.cache
def lambda_mean(band, Ifun=trapezoid):
    """
    Calculates lambda_mean for one of the filters
    """
    R, _ = load_bandpass(band)
    return Ifun(R.y*R.x, R.x) / Ifun(R.y, R.x)

@functools.cache
def lambda_eff(band, mod="002", Ifun=trapezoid):
    """
    Calculates lambda_eff for one of the filters, integrated over the spectrum
    of Vega.
    """
    R, _ = load_bandpass(band)
    V = load_Vega(mod).clip(R.x[0], R.x[-1])
    Ri = R(V.x)
    return Ifun(Ri*V.y*V.x, V.x) / Ifun(Ri*V.y, V.x)

@functools.cache
def Vega_AB_mag_offset(band, mod="002"):
    """
    Calculates the magnitude difference between Vega and the AB scale. This
    is essentially just the AB magnitude of Vega in a specific band.
    """
    V = load_Vega(mod=mod)
    return V.mag_calc_AB(band)
