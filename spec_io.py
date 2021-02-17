"""
Utilities for reading in spectra from different filetypes and returning a
Spectrum class.
"""
import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from trm import molly
from .spec_class import Spectrum

__all__ = [
    "spec_from_txt",
    "spec_from_npy",
    "model_from_txt",
    "model_from_dk",
    "head_from_dk",
    "spec_from_sdss_fits",
    "subspectra_from_sdss_fits",
    "spec_from_fits_generic",
    "spec_list_from_molly",
]

#element dict for dk headers
el_dict = {
     1:'H' ,  2:'He',  3:'Li',  6:'C' ,  7:'N' ,  8:'O' ,  9:'F' ,
    10:'Ne', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P' , 16:'S' ,
    17:'Cl', 18:'Ar', 19:'K' , 20:'Ca', 21:'Sc', 22:'Ti', 23:'V' ,
    24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn',
    31:'Ga', 32:'Ge', 38:'Sr', 56:'Ba',
}

def spec_from_txt(fname, wave='air', x_unit='AA', y_unit='erg/(s cm2 AA)', \
    delimiter=r'\s+', **kwargs):
    """
    Loads a text file with the first 3 columns as wavelengths, fluxes, errors.
    kwargs are passed to pd.read_csv.
    """
    x, y, e = pd.read_csv(fname, delimiter=delimiter, usecols=range(3), \
        na_values="NAN", **kwargs).values.T
    name, _ = os.path.splitext(os.path.basename(fname))
    return Spectrum(x, y, e, name, wave, x_unit, y_unit)

def model_from_txt(fname, wave='vac', x_unit='AA', y_unit='erg/(s cm2 AA)', \
    delimiter=r'\s+', **kwargs):
    """
    Loads a text file with the first 2 columns as wavelengths and fluxes.
    This produces a spectrum object where the errors are just set to zero.
    This is therefore good to use for models. kwargs are passed to pd.read_csv.
    """
    x, y = pd.read_csv(fname, delimiter=delimiter, usecols=range(2), **kwargs).values.T
    name, _ = os.path.splitext(os.path.basename(fname))
    return Spectrum(x, y, 0, name, wave, x_unit, y_unit)

def multi_model_from_txt(fname, nfluxcols, wave='vac', \
    x_unit='AA', y_unit='erg/(s cm2 AA)', delimiter=r'\s+', **kwargs):
    """
    Similar to model_from_txt, but for files with multiple flux columns. This
    can be useful for loading model spectra with a single wavelength axis and
    different model fluxes for each column.
    """
    x, *yy = pd.read_csv(fname, delimiter=delimiter, usecols=range(nfluxcols+1), \
        **kwargs).values.T
    name, _ = os.path.splitext(os.path.basename(fname))
    return [Spectrum(x, y, 0, name, wave, x_unit, y_unit) for y in yy]

def head_from_dk(fname, return_skip=False):
    """
    Return the header from a DK file, optionally the number of rows to skip for
    reading the data after.
    """
    hdr = {'el':{}}
    with open(fname, 'r') as Fdk:
        for skip, line in enumerate(Fdk, 1):
            if line.startswith("TEFF"):
                hdr['Teff'] = float(line.split()[2])
            elif line.startswith("LOG_G"):
                hdr['logg'] = float(line.split()[2])
            elif line.startswith("COMMENT   el"):
                *_, Z, logZ = line.split()
                Z = int(Z)
                if Z >= 100: #compatability with older dk files
                    Z //= 100
                logZ = float(logZ)
                hdr['el'][el_dict[Z]] = logZ
            elif line.startswith("NMU"):
                #angular dependent fluxes
                hdr['mu'] = []
                hdr['wmu'] = []
            elif line.startswith("MU"):
                hdr['mu'] += [float(x) for x in line.split()[2:]]
            elif line.startswith("WMU"):
                hdr['wmu'] += [float(x) for x in line.split()[2:]]
            elif line.startswith("END"):
                break
            else:
                continue
    return (hdr, skip) if return_skip else hdr

def model_from_dk(fname, x_unit='AA', y_unit='erg/(s cm2 AA)'):
    """
    Similar to model_from_txt, but for Detlev Koester model spectra. Units are
    converted to those specified, and DK header items placed in the Spectrum
    object header. For models with angular dependent fluxes, a list of models
    is returned.
    """
    hdr, skip = head_from_dk(fname, True)
    kwargs = {'wave':'vac', 'x_unit':'AA', 'y_unit':'erg/(s cm3)', 'skiprows':skip}
    if 'mu' in hdr:
        #angular dependent fluxes
        mus, wmus = hdr.pop('mu'), hdr.pop('wmu')
        MM = multi_model_from_txt(fname, len(mus), **kwargs)
        for M in MM:
            M.x_unit_to(x_unit)
            M.y_unit_to(y_unit)
            M.head.update(hdr)
        for M, mu, wmu in zip(MM[1:], mus, wmus):
            M.name += f"_mu={mu:f}"
            M.head['mu'] = mu
            M.head['wmu'] = wmu
        return MM
    else:
        M = model_from_txt(fname, **kwargs)
        M.x_unit_to(x_unit)
        M.y_unit_to(y_unit)
        M.head.update(hdr)
        return M

def spec_from_npy(fname, wave='air', x_unit='AA', y_unit='erg/(s cm2 AA)'):
    """
    Loads a npy file with 2 or 3 columns as wavelengths, fluxes(, errors).
    """
    data = np.load(fname)
    if data.ndim != 2:
        raise ValueError("Data must be 2D")

    if data.shape[0] == 2:
        x, y, e = *data, 0
    elif data.shape[0] == 3:
        x, y, e = data
    else:
        print("Data should have 2 or 3 columns")
        sys.exit()
    name, _ = os.path.splitext(os.path.basename(fname))
    return Spectrum(x, y, e, name, wave, x_unit, y_unit)

def spec_from_sdss_fits(fname, **kwargs):
    """
    loads a sdss fits file as spectrum (result in vac wavelengths)
    """
    hdulist = fits.open(fname, **kwargs)
    S = _get_spec_from_hdu(hdulist[1])
    name, _ = os.path.splitext(os.path.basename(fname))
    S.name = name
    S.head['PLATE'] = hdulist[0].header['PLATEID']
    S.head['MJD'] = hdulist[0].header['MJD']
    S.head['FIBER'] = hdulist[0].header['PLATEID']
    return S

def subspectra_from_sdss_fits(fname, **kwargs):
    """
    loads a sdss fits file as spectrum (result in vac wavelengths)
    """
    hdulist = fits.open(fname, **kwargs)
    return [_get_spec_from_hdu(hdu) for hdu in hdulist[4:]]

def _get_spec_from_hdu(hdu):
    loglam, flux, ivar, sky = [hdu.data[key] for key in 'loglam flux ivar sky'.split()]
    lam = 10**loglam
    ivar[ivar == 0.] = 0.001
    err = 1/np.sqrt(ivar)
    sky *= 1e17
    name = hdu.header['EXTNAME']
    head = {'sky':sky}
    return Spectrum(lam, flux, err, name, 'vac', head=head)*1e-17

def spec_from_fits_generic(fname, wave='air', x_unit="AA", y_unit="erg/(s cm2 AA)"):
    """
    Load a spectrum from a generic fits file
    """
    hdulist = fits.open(fname)
    hdr = dict(hdulist[0].header)
    data = hdulist[1].data
    x, y, e = [data[col] for col in ("Wavelength", "Flux", "Error")]
    e[e < 0] = np.inf
    name = hdr['OBJECT'] if 'OBJECT' in hdr else fname
    return Spectrum(x, y, e, name, wave, x_unit, y_unit, hdr)

def spec_list_from_molly(fname):
    """
    Returns a list of spectra read in from a TRM molly file.
    """
    return list(map(_convert_mol, molly.gmolly(fname)))

def _convert_mol(molsp):
    x, y, e = molsp.wave, molsp.f, molsp.fe
    name = molsp.head['Object']
    S = Spectrum(x, y, np.abs(e), name, y_unit="mJy")
    S.head = molsp.head
    return S
