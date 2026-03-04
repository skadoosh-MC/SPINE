import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pickle as pkl
import os
import random
from scipy.interpolate import interpolate
from scipy.interpolate import splrep, splev
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as sp
from copy import deepcopy
import scipy.fft as fft
from scipy import integrate
from mpmath import quad
from colossus.cosmology import cosmology
# from astropy import units as u
import camb
from camb import model, initialpower
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import math
from spine_pk.growth import GrowthCalculator

class Power:
    kmin = 1e-2
    kmax = 2

    def __init__(self, kl, pkl, h, omega_m, omega_b, ns, sigma_8):
        """
        Initialize the input parameters.
        Attributes
        ----------
        kl: array
            k [(h/Mpc)] for the linear power spectrum
        pkl: array
            The linear power spectrum [(Mpc/h)^3]
        h: float
            Hubble constant, H0, divided by 100 km/s/Mpc 
        Omegam: float
            Matter density of the universe at z = 0
        Omegab: float
            Baryon denisty of the universe at z = 0
        ns: float
            Spectral tilt of the primordial power spectrum 
        sigma8: float
            Root-mean-square density fluctuation when the linearly 
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        """
        self.kl = kl
        self.pkl = pkl
        self.h = h
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.ns = ns
        self.sigma_8 = sigma_8


    def sigma8_to_As(self, old_equation=False):
        """
        Compute the emulated conversion sigma8 -> As as given in Bartlett et al. 2023
    
        Args:
            :sigma8 (float): Root-mean-square density fluctuation when the linearly
                evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
            :Om (float): The z=0 total matter density parameter, Omega_m
            :Ob (float): The z=0 baryonic density parameter, Omega_b
            :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
            :ns (float): Spectral tilt of primordial power spectrum
            :old_equation (bool, default=False): Whether to use the version of the sigma8
                emulator which appeared in v1 of the paper on arXiv (True) or the final
                published version (and v2 on arXiv).
    
        Returns:
            :As (float): 10^9 times the amplitude of the primordial P(k)
        """
        if old_equation:
            a = [0.161320734729, 0.343134609906, -
                 7.859274, 18.200232, 3.666163, 0.003359]

            As = ((self.sigma8 - a[5]) /
                  (a[2] * self.omega_b + np.log(a[3] * self.omega_m)) /
                  np.log(a[4] * self.h) -
                  a[1] * self.ns) / a[0]

        else:
            a = [0.51172, 0.04593, 0.73983, 1.56738, 1.16846,
                 0.59348, 0.19994, 25.09218, 9.36909, 0.00011]

            f = (
                a[0] * self.omega_m + a[1] * self.h +
                a[2] * (
                    (self.omega_m - a[3] * self.omega_b)
                    * (np.log(a[4] * self.omega_m) - a[5] * self.ns)
                    * (self.ns + a[6] * self.h *
                       (a[7] * self.omega_b - a[8] * self.ns +
                        np.log(a[9] * self.h)))
                )
            )

            As = (self.sigma_8 / f) ** 2

        return As


    def set_cosmo(self):
        """
        Computes the cosmology from CAMB based on input cosmology.
        Returns
        -----------
        rdrag: float
            Sound horizon at the drag redshift in units of [(h/Mpc)]
        sigma12:
            Root-mean-square density fluctuation when the linearly evolved field is 
            smoothed with a top-hat filter of radius 12 Mpc
        """
        amp = self.sigma8_to_As()

        params_camb = camb.set_params(H0=self.h*100, 
                                      ombh2=self.omega_b*self.h**2,
                                      omch2=(self.omega_m - self.omega_b)*self.h**2,
                                      omk=0,
                                      w=-1,
                                      ns=self.ns,
                                      halofit_version='original',
                                      As=amp*1e-9,
                                      DoLensing=False)

        params_camb.set_matter_power(redshifts=[0], kmax=10)
        results = camb.get_results(params_camb)

        sigma12 = results.get_sigmaR(12, z_indices=0, hubble_units=False)
        additional_param = camb.get_background(params_camb)
        rdrag = results.sound_horizon(additional_param.get_derived_params()['zdrag']) * (params_camb.H0/100)

        return rdrag, sigma12


    def get_linear_nowiggle(self,
                            range_imin=np.array([80,150]),
                            range_imax=np.array([200,300]),
                            threshold=0.04,
                            offset=-25):
        """
        Computes the no wiggle linear power spectrum.
        Attributes
        ----------
        range_imin: array
            Start of the BAO feature
        range_imax: array
            End of the BAO feature
        threshold: float
            Boundary for where the BAO oscillation needs to stop in the second derivative 
            of the even part of the discrete sine transform
        offset: int
            Fixed shift applied at the start of the BAO removal window to ensure all
            of the BAO is encapsulated.
        Returns
        ---------
        kvec: array
            Array of wavenumbers [(h/Mpc)] normalised to the sound horizon
        pkl_nw: array
            The no wiggle linear power spectrum [(Mpc/h)^3]
        interp_pk: array
            Linear power spectrum at kvec values [(Mpc/h)^3]
        f: scipy.interp1d object
        """
        rdrag, _ = self.set_cosmo()

        kr = np.linspace(0.005, 1000., 2**16)
        kvec = kr / rdrag

        x = self.kl
        y = self.pkl

        f = sp.interp1d(x, y, kind='quadratic', fill_value='extrapolate')

        interp_pk = f(kvec)
        interp_pk[interp_pk < 0] = 1e-5

        xvec = np.log(kvec * interp_pk)
        xvec = fft.dst(xvec, type=2)
        frec = np.arange(len(kvec)/2)

        even = xvec[::2]
        odd = xvec[1::2]

        even_spline = UnivariateSpline(frec, even, k=3, s=0)

        result = even_spline.derivative(n=2)(frec)

        if np.any(np.isnan(result)):
            print(np.any(np.isnan(interp_pk)))
            print(rdrag)
            print(self.h, self.omega_m, self.omega_b, self.ns)
            raise Exception()

        imin_cut = np.argmin(
            even_spline.derivative(n=2)(
                np.arange(range_imin[0], range_imin[1])
            )
        ) + range_imin[0] + offset

        try:
            imax_cut = (
                np.where(
                    even_spline.derivative(n=2)(
                        np.arange(range_imax[0], range_imax[1])
                    ) < threshold
                )[0][0] + range_imax[0]
            )
        except IndexError:
            print(threshold)
            print(np.min(
                even_spline.derivative(n=2)(
                    np.arange(range_imax[0], range_imax[1])
                )
            ))
            imax_cut = range_imax[1]

        window = np.arange(imin_cut, imax_cut)

        r = np.delete(frec, window)

        even_nobumb = np.delete(even, window)
        odd_nobumb = np.delete(odd, window)

        even_nobumb_spline = UnivariateSpline(r, (r+1)**2 * even_nobumb, k=3, s=0)
        odd_nobumb_spline = UnivariateSpline(r, (r+1)**2 * odd_nobumb, k=3, s=0)

        xvec[::2] = even_nobumb_spline(frec)/(frec + 1)**2
        xvec[1::2] = odd_nobumb_spline(frec)/(frec + 1)**2

        xvec = fft.idst(xvec, type=2)
        pkl_nw = np.exp(xvec) / kvec

        return kvec, pkl_nw, interp_pk, f


    def get_sigmav(self):
        """
        Calculates the displacement of the matter density field in linear theory/BAO damping factor.
        Returns
        ---------
        sigmav: float
            BAO damping factor
        """
        kvec, pklnw, _, f = self.get_linear_nowiggle()

        function = lambda kvec: f(kvec) * (1/(6*np.pi**2))
        sigmav_quad = integrate.quad(function, 0.005, 10, limit=10000)
        sigmav = np.sqrt(sigmav_quad[0])

        return sigmav


    def get_linear_damped(self):
        """
        Calculates the damped BAO linear power spectrum.
        Retruns
        ---------
        pkdamp: array
            Damped BAO linear theory power spectrum [(Mpc/h)^3]
        """
        sigmav = self.get_sigmav()
        kvec, pkl_nw, interp_pk, _ = self.get_linear_nowiggle()
        gauss = np.exp(-(kvec*sigmav)**2 * (1/2))
        pkdamp = pkl_nw + ((interp_pk - pkl_nw) * gauss)

        return pkdamp


    def get_additional_params(self, kmap):
        """
        Calculates parameters additionally needed in the final equation. Calculated via COLOSSUS.
        Attributes
        ----------
        kmap: array
            The mapped k [(h/Mpc)] values according to the Peacock & Dodds 1996 formalism
        Returns
        ----------
        growthsupfactor: float
            The linear growth suppression factor for z=0
        n_L: array
            The slope of the late time power spectrum
        """
        kvec, pklnw, _, _ = self.get_linear_nowiggle()

        params_colossus = {'flat':True, 
                           'H0':self.h*100,
                           'Ob0':self.omega_b,
                           'ns':self.ns,
                           'Om0':self.omega_m,
                           'w0':-1,
                           'sigma8':self.sigma_8}

        cosmo = cosmology.setCosmology('QuijoteSR', **params_colossus)

        growthsupfactor = cosmo.growthFactorUnnormalized(0)

        n_L = cosmo.matterPowerSpectrum(kmap/2, z=0, derivative=True, model='eisenstein98_zb')

        return growthsupfactor, n_L
