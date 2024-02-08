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
    "model_from_dk",
    "head_from_dk",
    "spec_from_sdss_fits",
    "subspectra_from_sdss_fits",
    "spec_from_fits_generic",
    "spec_list_from_molly",
]

#element dict for dk headers
el_dict = {
     1:'H' ,  2:'He',  3:'Li',  4:'Be',  6:'C' ,  7:'N' ,  8:'O' ,
     9:'F' , 10:'Ne', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P' ,
    16:'S' , 17:'Cl', 18:'Ar', 19:'K' , 20:'Ca', 21:'Sc', 22:'Ti',
    23:'V' , 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu',
    30:'Zn', 31:'Ga', 32:'Ge', 38:'Sr', 56:'Ba',
}

def spec_from_txt(fname, wave=None, x_unit='AA', y_unit='erg/(s cm2 AA)', \
    delimiter=r'\s+', model=False, **kwargs):
    """
    Loads a text file as a Spectrum object. If model is set to True, only two
    columns are read (wavelengths and fluxes) with errors set to zero,
    otherwise errors are read from the third column. If wave is not explicitly
    set, models are assumed to be 'vac', or 'air' for observed spectra with
    errors. kwargs are passed to pd.read_csv.
    """
    ncols = 2 if model else 3
    data = pd.read_csv(fname, delimiter=delimiter, usecols=range(ncols), **kwargs)
    data = data.values.T
    x, y, e = (*data, 0) if model else data
    if wave is None:
        wave = 'vac' if model else 'air'
    name, _ = os.path.splitext(os.path.basename(fname))
    return Spectrum(x, y, e, name, wave, x_unit, y_unit)

def multi_model_from_txt(fname, nfluxcols, wave='vac', \
    x_unit='AA', y_unit='erg/(s cm2 AA)', delimiter=r'\s+', **kwargs):
    """
    Read files with multiple flux columns as model spectra (no uncertainties).
    This can be useful for loading model spectra with a single wavelength axis
    and different model fluxes for each column.
    """
    x, *yy = pd.read_csv(fname, delimiter=delimiter, usecols=range(nfluxcols+2), \
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
                hdr['el'][el_dict[Z]] = float(logZ)
            elif line.startswith("NMU"):
                #angular dependent fluxes
                hdr['mu'], hdr['wmu'] = [], []
            elif line.startswith("MU"):
                hdr['mu'] += [float(x) for x in line.split()[2:]]
            elif line.startswith("WMU"):
                hdr['wmu'] += [float(x) for x in line.split()[2:]]
            elif line.startswith("END"):
                break
            else:
                continue
    return (hdr, skip) if return_skip else hdr

def model_from_dk(fname, x_unit='AA', y_unit='erg/(s cm2 AA)', use_Imu=False):
    """
    Read Detlev Koester white dwarf models as spectra. Units are converted to
    those specified, and DK header items placed in the Spectrum object header.
    For models with angular dependent fluxes, if use_Imu is set to True a list
    of models is returned, otherwise just the disc average flux is used.
    """
    hdr, skip = head_from_dk(fname, True)
    kwargs = {'wave':'vac', 'x_unit':'AA', 'y_unit':'erg/(s cm3)', 'skiprows':skip}
    if 'mu' in hdr and use_Imu:
        #angular dependent fluxes
        mus, wmus = hdr.pop('mu'), hdr.pop('wmu')
        MM = multi_model_from_txt(fname, len(mus), **kwargs)
        for M in MM:
            M.x_unit_to(x_unit)
            M.y_unit_to(y_unit)
            M.head.update(hdr)
        for M, mu, wmu in zip(MM[1:], mus, wmus):
            M.name += f"_mu_{mu:f}"
            M.head['mu'], M.head['wmu'] = mu, wmu
        return MM
    M = spec_from_txt(fname, model=True, **kwargs)
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
    if data.shape[0] not in (2, 3):
        raise ValueError("Data should have 2 or 3 columns")

    x, y, e = (*data, 0) if data.shape[0] == 2 else data
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
    hdr = hdulist[0].header
    S.head['PLATE'] = hdr['PLATEID']
    S.head['MJD'] = hdr['MJD']
    #FIXME using wrong keyword
    S.head['FIBER'] = hdr['PLATEID']
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
    name = hdu.header['EXTNAME'] if 'EXTNAME' in hdu.header else ""
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
    return [_convert_mol(mol) for mol in molly.gmolly(fname)]

def _convert_mol(molsp):
    x, y, e = molsp.wave, molsp.f, molsp.fe
    name = molsp.head['Object']
    S = Spectrum(x, y, np.abs(e), name, y_unit="mJy")
    S.head = molsp.head
    return S
