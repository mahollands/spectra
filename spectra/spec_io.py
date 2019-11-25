"""
Utilities for reading in spectra from different filetypes and returning a
Spectrum class.
"""
import numpy as np
import os
from sys import exit
from trm import molly
from astropy.io import fits
from .spec_class import Spectrum

__all__ = [
  "spec_from_txt",
  "spec_from_npy",
  "model_from_txt",
  "model_from_dk",
  "spec_from_sdss_fits",
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

def spec_from_txt(fname, wave='air', x_unit='AA', y_unit='erg/(s cm2 AA)', **kwargs):
  """
  Loads a text file with the first 3 columns as wavelengths, fluxes, errors.
  kwargs are passed to np.loadtxt.
  """
  x, y, e = np.loadtxt(fname, unpack=True, usecols=(0,1,2), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, e, name, wave, x_unit, y_unit)
    
def model_from_txt(fname, wave='vac', x_unit='AA', y_unit='erg/(s cm2 AA)', **kwargs):
  """
  Loads a text file with the first 2 columns as wavelengths and fluxes.
  This produces a spectrum object where the errors are just set to zero.
  This is therefore good to use for models. kwargs are passed to np.loadtxt.
  """
  x, y = np.loadtxt(fname, unpack=True, usecols=(0,1), **kwargs)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, 0, name, wave, x_unit, y_unit)

def model_from_dk(fname, x_unit='AA', y_unit='erg/(s cm2 AA)'):
  """
  Similar to model_from_txt, but will autoskip past the DK header.
  Units are converted to those specified.
  """
  #extract header first
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
      elif line.startswith("END"):
        break
      else:
        continue
  M = model_from_txt(fname, 'vac', 'AA', 'erg/(s cm3)', skiprows=skip)
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
    exit()
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(x, y, e, name, wave, x_unit, y_unit)

def spec_from_sdss_fits(fname, **kwargs):
  """
  Loads a SDSS fits file as spectrum (result in vac wavelengths)
  """
  hdulist = fits.open(fname)
  loglam, flux, ivar = [hdulist[1].data[key] for key in ('loglam', 'flux', 'ivar')]
  lam = 10**loglam
  ivar[ivar==0.] = 0.001
  err = 1/np.sqrt(ivar)
  name = os.path.splitext(os.path.basename(fname))[0]
  return Spectrum(lam, flux, err, name, 'vac')*1e-17

def spec_from_fits_generic(fname, wave='air', x_unit="AA", y_unit="erg/(s cm2 AA)"):
  hdulist = fits.open(fname)
  hdr = dict(hdulist[0].header)
  data = hdulist[1].data
  x, y, e = [data[col] for col in ("Wavelength","Flux","Error")]
  e[e<0] = np.inf
  name = hdr['OBJECT'] if 'OBJECT' in hdr else fname
  return Spectrum(x, y, e, name, wave, x_unit, y_unit, hdr)


def spec_list_from_molly(fname):
  """
  Returns a list of spectra read in from a TRM molly file.
  """
  def convert_mol(molsp):
    x, y, e = molsp.wave, molsp.f, molsp.fe
    name = molsp.head['Object']
    S = Spectrum(x, y, np.abs(e), name, y_unit="mJy")
    S.head = molsp.head
    return S
  
  return [convert_mol(molsp) for molsp in molly.gmolly(fname)]

