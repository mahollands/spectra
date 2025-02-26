"""
Contains the Spectrum class for working with astrophysical spectra.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from dust_extinction import parameter_averages as dust_params
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import minimize
from . import spec_io
from .interpolation import interp, interp_nan, interp_inf, wbin
from .synphot import calc_AB_flux, Vega_AB_mag_offset
from .broadening import convolve_gaussian, rotational_broadening
from .misc import Wave, vac_to_air, air_to_vac, logarange

__all__ = [
    "Spectrum",
]


class Spectrum:
    """
    spectrum class contains wavelengths, fluxes, and flux errors.  Arithmetic
    operations with single values, array, or other spectra are defined, with
    standard error propagation supported. Spectra also support numpy style
    array slicing

    Example:
    >>> S1 = Spectrum(x1, y1, e1)
    >>> S2 = Spectrum(x2, y2, e2)
    >>> S3 = S1 - S2

    .............................................................................
    In this case S1, S2, and S3 are all 'Spectrum' objects but the errors on S3
    are calculated as S3.e = sqrt(S1.e**2 + S2.e**2)

    If one needs only one of the arrays, these can be accessed as attributes.

    Example:
    .............................................................................
    >>> S.plot('k-') #plots the spectrum with matplotlib
    >>> plt.show()

    .............................................................................
    Note that for operations of the form
    >>> S + a ;  S - a ;  S * a ;  S / a
    >>> a + S ;  a - S ;  a * S ;  a / S

    both rows work as expected if 'a' is an int/float. However if 'a' is an
    ndarray or Quantity object, the second row (__radd__ etc) is overridden
    by undefined behaviour of numpy/astropy implementations.
    """
    __slots__ = ['_x', '_y', '_e', '_name', '_wave', '_xu', '_yu', '_head']

    def __init__(self, x, y, e, name="", wave=Wave.AIR, x_unit="AA",
                 y_unit="erg/(s cm^2 AA)", head=None):
        """
        Initialise spectrum. Arbitrary header items can be added to self.head
        x must be an ndarray. y and e can either by int/floats or ndarrays of
        the same length.
        """
        self.x = x
        self.y = y
        self.e = e
        self.name = name
        self.wave = wave
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.head = head

    from_txt = classmethod(spec_io.spec_from_txt)
    from_npy = classmethod(spec_io.spec_from_npy)
    from_dk = classmethod(spec_io.model_from_dk)
    from_sdss = classmethod(spec_io.spec_from_sdss_fits)
    from_fits = classmethod(spec_io.spec_from_fits_generic)
    list_from_txt = classmethod(spec_io.multi_model_from_txt)
    list_from_sdss_subspectra = classmethod(spec_io.spec_from_sdss_fits)
    list_from_molly = classmethod(spec_io.spec_list_from_molly)
    write = spec_io.write

    @property
    def x(self):
        """
        x array, i.e. wavelengths or wavenumber
        """
        return self._x

    @x.setter
    def x(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be an ndarray")
        if x.ndim != 1:
            raise ValueError("x arrays must be 1D")
        self._x = x.astype(float)

    @property
    def dx(self):
        """
        derivative of x-values per pixel
        """
        return np.diff(self.x)

    @property
    def y(self):
        """
        flux density array
        """
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, (int, float)):
            self._y = y*np.ones_like(self.x)
        elif isinstance(y, np.ndarray):
            if y.shape != self.x.shape:
                raise ValueError("for ndarrays, y must be the same shape as x")
            self._y = y.astype(float)
        else:
            raise TypeError("y must be of type int/float/ndarray")
        # Add tiny flux to avoid div0 problems with models
        self.y[self.y == 0] += 1E-100

    @property
    def e(self):
        """
        flux density uncertainty array
        """
        return self._e

    @e.setter
    def e(self, e):
        if isinstance(e, (int, float)):
            if e < 0:
                raise ValueError("Uncertainties cannot be negative")
            self._e = e*np.ones_like(self.x)
        elif isinstance(e, np.ndarray):
            if e.shape != self.x.shape:
                raise ValueError("for ndarrays, e must be the same shape as x")
            if np.any(e < 0):
                raise ValueError("Uncertainties cannot be negative")
            self._e = e.astype(float)
        else:
            raise TypeError("y must be of type int/float/ndarray")

    @property
    def name(self):
        """
        spectrum name property
        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name

    @property
    def wave(self):
        """
        wavelength type attribute ("air"/"vac"). Ought to be "vac" when
        x-values are wavenumber rather than wavelengths.`
        """
        return self._wave

    @wave.setter
    def wave(self, wave):
        self._wave = Wave(wave)

    @property
    def x_unit(self):
        """
        x-unit attribute. Can be set with either a string or astropy unit type,
        but returns string for simplicity.
        """
        return self._xu

    @x_unit.setter
    def x_unit(self, unit):
        if not isinstance(unit, (str, u.UnitBase)):
            raise TypeError("x_unit must be str or Unit type")
        self._xu = u.Unit(unit)

    @property
    def y_unit(self):
        """
        y-unit attribute. Can be set with either a string or astropy unit type,
        but returns string for simplicity.
        """
        return self._yu

    @y_unit.setter
    def y_unit(self, unit):
        if not isinstance(unit, (str, u.UnitBase)):
            raise TypeError("y_unit must be str or Unit type")
        self._yu = u.Unit(unit)

    def x_unit_to(self, new_unit):
        """
        Changes units of the x-data. Supports conversion between wavelength
        and energy etc. Argument should be a string or Unit.
        """
        x = self.xq
        x = x.to(new_unit, u.spectral())
        self.x, self.x_unit = x.value, new_unit

    def y_unit_to(self, new_unit):
        """
        Changes units of the y-data. Supports conversion between Fnu
        and Flambda etc. Argument should be a string or Unit.
        """
        x, y, e = self.xq, self.yq, self.eq
        y = y.to(new_unit, u.spectral_density(x))
        e = e.to(new_unit, u.spectral_density(x))
        self.y, self.e, self.y_unit = y.value, e.value, new_unit

    @property
    def head(self):
        """
        spectrum header attribute. This is a dictionary which can be used to
        store arbtirary data associated with the spectrum.
        """
        return self._head

    @head.setter
    def head(self, head):
        if head is None:
            self._head = {}
            return
        if not isinstance(head, dict):
            raise ValueError("head must be a dictionary")
        self._head = head

    @property
    def var(self):
        """
        Variance attribute from flux errors
        """
        return self.e**2

    @var.setter
    def var(self, value):
        self.e = np.sqrt(value)

    @property
    def ivar(self):
        """
        Inverse variance attribute from flux errors
        """
        return 1.0/self.var

    @ivar.setter
    def ivar(self, value):
        self.var = 1.0/value

    @property
    def xq(self):
        """
        returns x as a quantity by attatching its unit
        """
        return self.x << self.x_unit

    @property
    def yq(self):
        """
        returns y as a quantity by attatching its unit
        """
        return self.y << self.y_unit

    @property
    def eq(self):
        """
        returns e as a quantity by attatching its unit
        """
        return self.e << self.y_unit

    @property
    def varq(self):
        """
        returns var as a quantity by attatching its unit
        """
        return self.var << self.y_unit**2

    @property
    def ivarq(self):
        """
        returns ivar as a quantity by attatching its unit
        """
        return self.ivar << 1/self.y_unit**2

    @property
    def y_e(self):
        """
        fluxes divided by errors. SN is just the modulus of this.
        This can be useful calculating residuals when a model is
        subtracted off the data, e.g.
        >>> (S-M).y_e == (S.y - M.y)/S.e

        where M.e are all zero.

        Can be used for making residual plots
        >>> (S-M).plot(kind='y_e')
        """
        return self.y/self.e

    @property
    def y_e2(self):
        """
        fluxes divided by errors squared
        """
        return self.y_e**2

    @property
    def chi2(self):
        """
        Calculates a chi squared value assuming y-values are scattered about 0.
        """
        return np.sum(self.y_e2)

    @property
    def SN(self):
        """
        Signal to noise ratio (modulus of y_e property)
        """
        return np.abs(self.y_e)

    @SN.setter
    def SN(self, value):
        self.e = np.abs(self.y/value)

    @property
    def magAB(self):
        """
        fluxes in terms of AB magnitudes
        """
        S = self.copy()
        S.y_unit_to("Jy")
        return -2.5*np.log10(S.y/3631)

    @property
    def magABe(self):
        """
        flux errors in terms of AB magnitudes
        """
        S = self.copy()
        S.y_unit_to("Jy")
        return -2.5*np.log10(S.e/3631)

    @property
    def x01(self):
        """
        Returns the lowest and highest x-values,
        e.g. for setting plot xlims
        """
        x0, x1 = self.x[0], self.x[-1]
        return (x0, x1) if x1 > x0 else (x1, x0)

    @property
    def data(self):
        """
        Returns all three arrays as a tuple. Useful for creating new spectra,
        e.g.
        >>> Spectrum(*S.data)
        """
        return self.x, self.y, self.e

    @property
    def info(self):
        """
        Returns non-array attributes as a dictionary. This can be used to
        create new spectra with the same information, e.g.
        >>> Spectrum(x, y, e, **S.info)
        """
        keys = {'name', 'wave', 'x_unit', 'y_unit', 'head'}
        return {k: getattr(self, k) for k in keys}

    @property
    def _model(self):
        """
        Spectra with zero uncertainties are assumed to be models.
        """
        return np.all(self.e == 0)

    def __len__(self):
        """
        Return number of pixels in spectrum
        """
        return len(self.x)

    def __repr__(self):
        """
        Return spectrum representation
        """
        parts = [
            f"*data[{len(self)}]",
            f"name='{self.name}'",
            f"wave='{self.wave}'",
            f"x_unit='{self.x_unit.to_string()}'",
            f"y_unit='{self.y_unit.to_string()}'",
            f"head={{{len(self.head)}}}",
        ]
        parts_str = ", ".join(parts)
        return f"Spectrum({parts_str})"

    def __getitem__(self, key):
        """
        Return self[key]
        """
        if not isinstance(key, (int, slice, np.ndarray)):
            raise TypeError("spectra must be indexed with int/slice/ndarray types")
        data_keyed = self.x[key], self.y[key], self.e[key]
        if isinstance(key, int):
            return data_keyed
        return Spectrum(*data_keyed, **self.info)

    def __setitem__(self, key, value):
        """
        Sets the indexed y-values to 'value', i.e.
        >>> self[key] = value
        """
        self.y[key] = value

    def __iter__(self):
        """
        Return iterator of spectrum
        """
        return zip(*self.data)

    def __contains__(self, value):
        """
        Return whether value is in the x-range of self
        """
        return self.x.min() < value < self.x.max()

    def _compare_units(self, other, xy):
        """
        Check units match another spectrum or kind of unit
        """
        if isinstance(other, (str, u.UnitBase)):
            # check specific unit
            if xy not in {'x', 'y'}:
                raise ValueError("xy not 'x' or 'y'")
            if xy == 'x' and self.x_unit != u.Unit(other):
                raise u.UnitsError("x_units differ")
            if xy == 'y' and self.y_unit != u.Unit(other):
                raise u.UnitsError("y_units differ")
        elif isinstance(other, u.Quantity):
            self._compare_units(other.unit, xy)
        elif isinstance(other, Spectrum):
            # compare two spectra
            if xy not in {'x', 'y', 'xy'}:
                raise ValueError("xy not 'x', 'y', or 'xy'")
            if xy in {'x', 'xy'} and self.x_unit != other.x_unit:
                raise u.UnitsError("x_units differ")
            if xy in {'y', 'xy'} and self.y_unit != other.y_unit:
                raise u.UnitsError("y_units differ")
        else:
            raise TypeError("other was not Spectrum or interpretable as a unit")

    def _compare_wave(self, other):
        if self.wave is not other.wave:
            raise ValueError("Spectra must have same wavelengths (air/vac)")

    def _compare_x(self, other):
        self._compare_wave(other)
        if not np.allclose(self.x, other.x):
            raise ValueError("Spectra must have same x values")

    def promote_to_spectrum(self, other, dimensionless_y=False):
        """
        Promote non-Spectrum objects (int/float/ndarray/quantity) to a Spectrum
        with similar properties to self. This is used internally to simplify
        arithmetic implementation, but also necessary for reverse arithmetic
        operations using ndarrays and quantities, e.g. 1 / Spectrum.
        """
        info = self.info
        if isinstance(other, u.Quantity):
            ynew = other.value
            info['y_unit'] = other.unit
        elif isinstance(other, (int, float, np.ndarray)):
            ynew = other
            if dimensionless_y:
                info['y_unit'] = u.dimensionless_unscaled
        else:
            raise NotImplementedError("Cannot cast object to Spectrum")
        return Spectrum(self.x, ynew, 0, **info)

    def _arithmetic_check_other(self, other, xy):
        if not isinstance(other, Spectrum):
            return self.promote_to_spectrum(other, xy == 'x')
        self._compare_units(other, xy)
        self._compare_x(other)
        return other

    def __add__(self, other):
        """
        Return self + other (with standard error propagation)
        """
        other = self._arithmetic_check_other(other, 'xy')
        ynew = self.y + other.y
        enew = np.hypot(self.e, other.e)
        return Spectrum(self.x, ynew, enew, **self.info)

    def __sub__(self, other):
        """
        Return self - other (with standard error propagation)
        """
        other = self._arithmetic_check_other(other, 'xy')
        ynew = self.y - other.y
        enew = np.hypot(self.e, other.e)
        return Spectrum(self.x, ynew, enew, **self.info)

    def __mul__(self, other):
        """
        Return self * other (with standard error propagation)
        """
        other = self._arithmetic_check_other(other, 'x')
        infonew = self.info
        infonew['y_unit'] = self.y_unit * other.y_unit
        ynew = self.y * other.y
        enew = np.hypot(self.e*other.y, other.e*self.y)
        return Spectrum(self.x, ynew, enew, **infonew)

    def __truediv__(self, other):
        """
        Return self / other (with standard error propagation)
        """
        other = self._arithmetic_check_other(other, 'x')
        infonew = self.info
        infonew['y_unit'] = self.y_unit / other.y_unit
        ynew = self.y / other.y
        enew = np.hypot(self.e, ynew*other.e)/abs(other.y)
        return Spectrum(self.x, ynew, enew, **infonew)

    def __radd__(self, other):
        """
        Return other + self (with standard error propagation)
        """
        return self + other

    def __rsub__(self, other):
        """
        Return other - self (with standard error propagation)
        """
        return -(self - other)

    def __rmul__(self, other):
        """
        Return other * self (with standard error propagation)
        """
        return self * other

    def __rtruediv__(self, other):
        """
        Return other / self (with standard error propagation)
        """
        other = self.promote_to_spectrum(other, True)
        return other / self

    def __pow__(self, other):
        """
        Return S**other (with standard error propagation)
        """
        if not isinstance(other, (int, float)):
            raise TypeError("other must be int/float")
        infonew = self.info
        infonew['y_unit'] = self.y_unit**other
        ynew = self.y**other
        enew = np.abs(other * ynew * self.e/self.y)
        return Spectrum(self.x, ynew, enew, **infonew)

    def __neg__(self):
        """
        Implements -self
        """
        return -1 * self

    def __pos__(self):
        """
        Implements +self
        """
        return self.copy()

    def __abs__(self):
        """
        Implements abs(self)
        """
        S = self.copy()
        S.y = np.abs(S.y)
        return S

    def apply_mask(self, mask):
        """
        Apply a mask to the spectral fluxes
        """
        self.x = np.ma.masked_array(self.x, mask)
        self.y = np.ma.masked_array(self.y, mask)
        self.e = np.ma.masked_array(self.e, mask)

    def remove_mask(self):
        """
        Remove mask from spectral fluxes
        """
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.e = np.array(self.e)

    def y_mc(self, N=1):
        """
        Iterator of Monte-Carlo sampled flux arrays distributed according to
        the uncertainties. If N is 1, a single array is returned.
        """
        if self._model:
            raise ValueError("Flux uncertainties are zero")
        if N < 1:
            raise ValueError("N must be >=1")
        if N == 1:
            return np.random.normal(self.y, self.e)
        return (np.random.normal(self.y, self.e) for i in range(N))

    def boot(self, N=1):
        """
        Iterator of bootstrapped sampled spectra. If N is 1, a single array is
        returned.
        """
        if N < 1:
            raise ValueError("N must be >=1")
        n = len(self)
        if N == 1:
            return self[np.random.randint(0, n, n)]
        return (self[np.random.randint(0, n, n)] for i in range(N))

    def flux_calc_AB(self, band, unit='Jy', attach_unit=False, Nmc=1000):
        """
        Calculates the AB flux (Jy) of a bandpass called 'band'. Errors
        are calculated in Monte-Carlo fashion, and assume all fluxes
        are statistically independent (not that realistic). See the
        definition of synphot.calc_AB_flux for valid filter names.
        """
        if self._model:
            Nmc = 0
        fnu = calc_AB_flux(self.copy(), band, Nmc) * u.Unit('Jy')
        fnu = fnu.to(unit)
        if not attach_unit:
            fnu = fnu.value
        return fnu if Nmc == 0 else (fnu.mean(), fnu.std())

    def mag_calc_AB(self, band, Nmc=1000):
        """
        Calculates the AB magnitude of a pandpass called 'band'. Errors
        are calculated in Monte-Carlo fashion, and assume all fluxes
        are statistically independent (not that realistic).
        """
        if self._model:
            Nmc = 0
        fnu = calc_AB_flux(self.copy(), band, Nmc)
        m = -2.5 * np.log10(fnu) + 8.90
        return m if Nmc == 0 else (m.mean(), m.std())

    def mag_calc_Vega(self, band, Nmc=1000, mod="002"):
        """
        Calculates the Vega magnitude of a pandpass called 'band'. Errors are
        calculated in Monte-Carlo fashion, and assume all fluxes are
        statistically independent (not that realistic). 'mod' refers to the
        specific Vega model to be used.
        """
        offset = Vega_AB_mag_offset(band, mod=mod)
        return self.mag_calc_AB(band, Nmc=Nmc) - offset

    def interp(self, xi, kind='cubic', **kwargs):
        """
        Interpolates a spectrum onto the wavelength axis xi, if xi is a numpy
        array, or xi.x if xi is Spectrum type. This returns a new spectrum
        rather than updating a spectrum in place, however this can be acheived
        by

        >>> S1 = S1.interp(X)

        Wavelengths outside the range of the original spectrum are filled with
        zeroes.
        """
        if not isinstance(xi, (np.ndarray, Spectrum)):
            raise TypeError("interpolant was not ndarray/Spectrum type")
        if isinstance(xi, Spectrum):
            self._compare_units(xi, 'x')
            self._compare_wave(xi)
            xi = xi.x

        yi, ei = interp(*self.data, xi, kind, self._model, **kwargs)
        return Spectrum(xi, yi, ei, **self.info)

    def interp_nan(self, interp_e=False):
        """
        Linearly interpolate over values with NaNs. If interp_e is set, these
        are also interpolated over, otherwise they are set to np.inf.
        """
        S = self.copy()
        S.y, S.e = interp_nan(*S.data, interp_e)
        return S

    def interp_inf(self):
        """
        Linearly interpolate over values with infs
        """
        S = self.copy()
        S.y, S.e = interp_inf(*S.data)
        return S

    def wbin(self, xbin, kind, logscale=False):
        """
        Wavelengths bins a spectrum onto xbin using linear or quadratic
        binning. Based on the REBIN routine in Molly.
        """
        if not isinstance(xbin, (np.ndarray, Spectrum)):
            raise TypeError("interpolant was not ndarray/Spectrum type")
        if isinstance(xbin, Spectrum):
            self._compare_units(xbin, 'x')
            self._compare_wave(xbin)
            xbin = xbin.x

        xin = np.log(self.x) if logscale else self.x
        xout = np.log(xbin) if logscale else xbin

        ybin = wbin(xin, self.y, xout, kind)
        ebin = wbin(xin, self.e, xout, kind)
        return Spectrum(xbin, ybin, ebin, **self.info)

    def down_sample(self, n_pixels):
        """
        Downsamples a spectrum by doing a simple mean
        of n pixel blocks. Extra pixels are trimmed from
        the left edge.
        """
        S = self[len(self) % n_pixels:]
        x, y, ivar = [arr.reshape((-1, n_pixels)) for arr in (S.x, S.y, S.ivar)]
        x_new = np.mean(x, axis=1)
        ivar_new = np.sum(ivar, axis=1)
        y_new = np.sum(y*ivar, axis=1) / ivar_new
        e_new = 1/np.sqrt(ivar_new)
        return Spectrum(x_new, y_new, e_new, **self.info)

    def copy(self):
        """
        Returns a copy of self
        """
        return Spectrum(*self.data, **self.info)

    def sect(self, x0, x1):
        """
        Returns a truth array for wavelengths between x0 and x1.
        """
        return (self.x > x0) & (self.x < x1)

    def sect2(self, x0, dx):
        """
        Returns a truth array for wavelengths between x0-dx and x0+dx.
        """
        return self.sect(x0-dx, x0+dx)

    def clip(self, x0, x1, pad=0, invert=False):
        """
        Returns Spectrum clipped between x0 and x1. If invert=True, returns the
        pixels outside that range. If pad > 0, the clip range is extended by
        this amount (useful when clipping a model to slightly wider than an
        observed spectrum.
        """
        if pad < 0: 
            raise ValueError
        return self[self.sect(x0-pad, x1+pad) ^ invert]

    def clip2(self, x0, dx, invert=False):
        """
        Returns Spectrum clipped between x0-dx and x0+dx. If invert=True,
        returns the pixels outside that range.
        """
        return self[self.sect2(x0, dx) ^ invert]

    def norm_percentile(self, pc):
        """
        Normalises a spectrum to a certain percentile of its fluxes. E.g.:
        >>> S.norm_percentile(99)
        """
        self /= np.percentile(self.y, pc)

    def air_to_vac(self):
        """
        Changes air wavelengths to vaccuum wavelengths in place
        """
        if self.wave is Wave.VAC:
            print(f"Wavelengths for {self.name} already vac")
            return
        unit0 = self.x_unit
        self.x_unit_to(u.AA)
        self.x = air_to_vac(self.x)
        self.wave = Wave.VAC
        self.x_unit_to(unit0)

    def vac_to_air(self):
        """
        Changes vaccuum wavelengths to air wavelengths in place
        """
        if self.wave is Wave.AIR:
            print(f"Wavelengths for {self.name} already air")
            return
        unit0 = self.x_unit
        self.x_unit_to(u.AA)
        self.x = vac_to_air(self.x)
        self.wave = Wave.AIR
        self.x_unit_to(unit0)

    def redden(self, E_BV, Rv=3.1, model='G23'):
        """
        Apply reddening curve to the spectrum given an E(B-V)
        and a value of Rv (default=3.1). The default model
        is currently Gordon et al. (2023).
        """
        S = self.copy()
        if S.wave is Wave.AIR:
            S.air_to_vac()

        extmod = getattr(dust_params, model)
        extmod_rv = extmod(Rv=Rv)
        extinction = extmod_rv.extinguish(S.xq, Ebv=E_BV)

        self.y *= extinction
        self.e *= extinction

    def apply_redshift(self, v, v_unit="km/s"):
        """
        Applies redshift of v km/s to spectrum for "air" or "vac" wavelengths
        """
        v *= u.Unit(v_unit)
        beta = (v/const.c).decompose()
        factor = math.sqrt((1+beta)/(1-beta))
        if self.wave is Wave.AIR:
            self.x = air_to_vac(self.x)
            self.x *= factor
            self.x = vac_to_air(self.x)
        else:
            self.x *= factor

    def scale_model(self, other, return_scaling_factor=False, assume_same_x=False):
        """
        If self is model spectrum (errors are presumably zero), and S is a data
        spectrum (has errors) then this reproduces a scaled version of M2.
        There is no requirement for either to have the same wavelengths as
        interpolation is performed. However the resulting scaled model will
        have the wavelengths of the original model, not the data. If you want
        the model to share the same wavelengths, use model.interp(), either
        before or after calling this function.
        """
        if not isinstance(other, Spectrum):
            raise TypeError
        self._compare_units(other, 'xy')

        # if M and S already have same x-axis, this won't do much.
        S = other[other.e > 0]
        M = self if assume_same_x else self.interp(S)

        A = np.sum(S.y*M.y*S.ivar)/np.sum(M.y**2*S.ivar)

        return (self*A, A) if return_scaling_factor else self*A

    def scale_model_to_model(self, other, return_scaling_factor=False,
                             assume_same_x=False):
        """
        Similar to scale_model, but for scaling one model to another. Essentially
        this is for the case when the argument doesn't have errors.
        """
        if not isinstance(other, Spectrum):
            raise TypeError
        self._compare_units(other, 'xy')

        # if M and S already have same x-axis, this won't do much.
        S = other
        M = self if assume_same_x else self.interp(S)

        A = np.sum(S.y*M.y)/np.sum(M.y**2)

        return (self*A, A) if return_scaling_factor else self*A

    def scale_spectrum_to_spectrum(self, other, return_scaling_factor=False,
                                   assume_same_x=False):
        """
        Scales self to best fit other in their mutually overlapping region.
        """
        if not isinstance(other, Spectrum):
            raise TypeError
        self._compare_units(other, 'xy')

        x0 = max(S.x.min() for S in (self, other))
        x1 = min(S.x.max() for S in (self, other))
        Soc = other.clip(x0, x1)
        Ssi = self if assume_same_x else self.interp(Soc, kind='cubic')

        res = minimize(lambda A, S1, S2: (S1-A*S2).chi2, (1.0), args=(Soc, Ssi))
        A = float(res['x'][0])

        return (self*A, A) if return_scaling_factor else self*A

    def scale_to_AB_mag(self, band, mag):
        """
        Scales a spectrum to match an AB magnitude for some specific bandpass
        """
        mag0 = self.mag_calc_AB(band, Nmc=0)
        return self * 10**(0.4*(mag0-mag))

    def scale_to_Vega_mag(self, band, mag):
        """
        Scales a spectrum to match a Vega magnitude for some specific bandpass
        """
        mag0 = self.mag_calc_Vega(band, Nmc=0)
        return self * 10**(0.4*(mag0-mag))

    def convolve_gaussian(self, fwhm):
        """
        Convolves spectrum with a Gaussian of specified FWHM
        """
        S = self.copy()
        S.y = convolve_gaussian(S.x, S.y, fwhm)
        return S

    def convolve_gaussian_R(self, res):
        """
        Convolves spectrum with a Gaussian of specified resolving power
        """
        S = self.copy()
        S.y = convolve_gaussian(np.log(S.x), S.y, 1/res)
        return S

    def convolve_gaussian_powerlaw_R(self, x_points, R_points):
        """
        Convolves spectrum with a Gaussian whose resolving power varies
        as a function of wavelength. The resolving power is modelled
        as R(x) = A*x^b, where x will normally be wavelength.

        Convolution is performed on a x^b scale, with a FWHM of b/A.
        """
        b, ln_a = np.polyfit(np.log(x_points), np.log(R_points), deg=1)

        S = self.copy()
        S.y = convolve_gaussian(S.x**b, S.y, b*np.exp(-ln_a))
        return S


    def rot_broaden(self, vsini, n_half=10, method='flat', coefs=None):
        """
        Apply rotational broadening in km/s. The n_half parameter dictates the
        number of resolution elements in the convolution (n = 2*n_half+1).
        """
        n = 2*n_half + 1
        dv = 2*vsini/n
        S = self.copy()
        S.x_unit_to(u.AA)
        S.y_unit_to("erg/(s cm2)")  # needed to conserve flux
        xnew = logarange(*S.x01, 2.998e5/dv)
        S = S.interp(xnew, kind='cubic')
        S.y = rotational_broadening(S.y, n, method=method, coefs=coefs)
        S = S.interp(self, kind='linear')
        S.x_unit_to(self.x_unit)
        S.y_unit_to(self.y_unit)
        return S

    def polyfit(self, deg, weighted=True, logx=False, logy=False):
        """
        Fits a polynomial to a spectrum object.
        """
        x = np.log(self.x) if logx else self.x
        y = np.log(np.abs(self.y)) if logy else self.y
        e = np.abs(self.e/self.y) if logy else self.e
        weights = 1/e if weighted else None
        poly = np.polyfit(x, y, deg, w=weights)
        return poly, logx, logy, self.y_unit

    def polyval(self, polyres):
        """
        Generates a spectrum from polynomial coefficients with the same
        shape/units as self. polyres should be: poly, logx, logy, y_unit
        """
        poly, logx, logy, y_unit = polyres
        x = np.log(self.x) if logx else self.x
        y = np.polyval(poly, x)
        y = np.exp(y) if logy else y
        infonew = self.info
        infonew['y_unit'] = y_unit
        return Spectrum(self.x, y, 0, **infonew)

    def splfit(self, knots, weighted=True):
        """
        Fits a spline to a spectrum object. The returned object is a function
        for which wavelengths should be passed. Knots may be an integer for
        equidistantly spaced knots, or an interable of specified knots.
        """
        if not isinstance(knots, (int, tuple, list, np.ndarray)):
            raise TypeError
        if isinstance(knots, int):
            # space knots equidistantly
            knots = np.linspace(*self.x01, knots+2)[1:-2]

        w = 1/self.e if weighted else 1
        return LSQUnivariateSpline(self.x, self.y, knots, w)

    def split(self, W):
        """
        If W is an int/float, splits spectrum in two around W. If W is an
        interable of ints/floats, this will split into mutliple chunks instead.
        """
        if isinstance(W, (int, float)):
            W = -np.inf, W, np.inf
        elif isinstance(W, (list, tuple, np.ndarray)):
            if not all(isinstance(w, (int, float)) for w in W):
                raise TypeError("w must all be of type int/float")
            W = -np.inf, *sorted(W), np.inf
        else:
            raise TypeError("W must be int/float or iterable of those types")
        return [self.clip(*w_pair) for w_pair in zip(W[:-1], W[1:])]

    @classmethod
    def join(cls, SS, sort=False, name=None):
        """
        Joins a collection of spectra into a single spectrum. The name of the
        first spectrum is used as the new name. Can optionally sort the new
        spectrum by wavelengths.
        """
        S0 = SS[0]

        for S in SS:
            if not isinstance(S, Spectrum):
                raise TypeError('item is not Spectrum')
            S._compare_wave(S0)
            S._compare_units(S0, xy='xy')

        x = np.hstack([S.x for S in SS])
        y = np.hstack([S.y for S in SS])
        e = np.hstack([S.e for S in SS])
        S = cls(x, y, e, **S0.info)

        if name is not None:
            S.name = name
        if sort:
            S = S[np.argsort(x)]

        return S

    @classmethod
    def mean(cls, SS):
        """
        Calculate the weighted mean spectrum of a list/tuple of spectra.
        All spectra should have identical wavelengths.
        """
        S0 = SS[0]
        for S in SS:
            if not isinstance(S, cls):
                raise TypeError('item is not Spectrum')
            S._compare_units(S0, xy='xy')
            S._compare_x(S0)

        Y = np.array([S.y for S in SS])
        IV = np.array([S.ivar for S in SS])

        IVbar = np.sum(IV, axis=0)
        Ybar = np.sum(Y*IV, axis=0) / IVbar
        Ebar = 1.0 / np.sqrt(IVbar)

        return cls(S0.x, Ybar, Ebar, **S0.info)

    def closest_x(self, x0):
        """
        Returns the pixel index closest in wavelength to x0
        """
        return np.argmin(np.abs(self.x-x0))

    def isnan(self):
        """
        Returns truth-array showing pixels with nans (either x, y, or e)
        """
        return np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.e)

    def isinf(self):
        """
        Returns truth-array showing pixels with infs (either x, y, or e)
        """
        return np.isinf(self.x) | np.isinf(self.y) | np.isinf(self.e)

    def plot(self, *args, kind='y', scale=1, auto_ylims=True, **kwargs):
        """
        Plots the spectrum with matplotlib and passes *args/**kwargs. 'kind'
        should be one of 'y', 'e', 'var', 'ivar', 'y_e', 'SN', 'magAB',
        'magABe'. plt.show() and other mpl functions still need to be used
        separately.
        """
        allowed = "y e var ivar y_e y_e2 SN magAB magABe".split()
        if kind not in allowed:
            raise ValueError(f"kind must be one of: {allowed}")

        if kind not in {"y", "e"} and scale != 1:
            raise ValueError("Only flux can be rescaled")

        y_plot = getattr(self, kind)
        ll = plt.plot(self.x, scale*y_plot, *args, **kwargs)

        # default y limits (if not already set)
        ax = plt.gca()
        if auto_ylims and ax.get_autoscaley_on():
            ylo, yhi = ax.get_ylim()
            if kind.startswith('magAB') and ylo < yhi:
                plt.ylim(yhi, ylo)
            else:
                plt.ylim(0, yhi)
            ax.set_autoscaley_on(True)
        return ll
