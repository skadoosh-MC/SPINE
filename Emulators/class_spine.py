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
from scipy import fft
from scipy import integrate
from mpmath import quad
from colossus.cosmology import cosmology
from astropy import units as u
import camb
from camb import model, initialpower
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import math
from growth import GrowthCalculator
from scipy.integrate import simpson

class Power:
    def __init__(self, kl, pkl, knl, pknl, h, omega_m, omega_b, ns, sigma_8):
        """
        Initialize the input parameters.
        Attributes
        ----------
        kl: float
            Array of linear wavenumbers
        pkl: float
            Array of linear power spectrum values
        knl: float
            Array of nonlinear wavenumbers
        pknl: float
            Array of nonlinear power spectrum values
        h: float
            Reduced Hubble parameter
        omega_m: float
            Total matter density
        omega_b: float
            Baryon density
        ns: float
            Spectral index ---> Slope of the primordial power spectrum
        sigma_8: float
            Variance of matter fluctuations in spheres of 8h^-1 Mpc
        """
        self.kl = kl
        self.pkl = pkl
        self.knl = knl
        self.pknl = pknl
        self.h = h
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.ns = ns
        self.sigma_8 = sigma_8

    def set_cosmo(self):
        """
        Computes the cosmology from CAMB based on input cosmology.
        Returns
        -----------
        rdrag: float
            Sound horizon at the drag redshift in units of h^-1 Mpc
        sigma12:
            Root-mean-square density fluctuation when the linearly evolved field is smoothed with a top-hat filter of radius 12 Mpc
        growthrate:
            Growth rate of structure
        """
        #From Bartlett et al. 2023 sigma8 to As emulator
        amp = sigma8_to_As(self.sigma_8, self.omega_m, self.omega_b, self.h, self.ns)

        params_camb = camb.set_params(H0=self.h*100, ombh2=self.omega_b*self.h**2, omch2=(self.omega_m - self.omega_b)*self.h**2, omk=0, w=-1,
                                      ns=self.ns, halofit_version='original', As=amp*1e-9, DoLensing=False)
        params_camb.set_matter_power(redshifts=[0], kmax=10)
        results = camb.get_results(params_camb)

        sigma12 = results.get_sigmaR(12, z_indices=0, hubble_units=False)
        growthrate = results.get_fsigma8()

        additional_param = camb.get_background(params_camb)    #getting the drag redshift
        #Sound horizon at drag redshift - camb gives results in Mpc

        #Divide by h to get in h Mpc
        rdrag = results.sound_horizon(additional_param.get_derived_params()['zdrag'])*(params_camb.H0/100)

        return rdrag, sigma12, growthrate

    def get_linear_nowiggle(self, range_imin = np.array([80,150]), range_imax = np.array([200, 300]), threshold = 0.04, offset = -25):
        """
        Computes the no wiggle linear power spectrum.
        Attributes
        ----------
        range_imin: array
            Where the BAO signal starts
        range_imax: array
            Where the BAO signal ends
        threshold: float
            ????
        offset: int
            Amount to
        Returns
        ---------
        kvec: array
            Array of wavenumbers normalised to the sound horizon
        pkl_nw: array
            The no wiggle linear power spectrum
        interp_pk: array
            Linear power spectrum at kvec values
        f: scipy.interp1d object
        """
        rdrag = self.set_cosmo()
        kr = np.linspace(0.005, 1000., 2**16)


        kvec = kr/rdrag

        '''-----------------------------interpolation input--------------------------------------------'''
        x = self.kl  #known values of k
        y = self.pkl  #known values of the linear (wiggled) power spectrum
        #using scipy interp1d -
        f = sp.interp1d(x, y, kind='quadratic', fill_value='extrapolate')  #f interpolates between x and y
        #find pk at kvec
        interp_pk = f(kvec)

        interp_pk[interp_pk < 0] = 1e-5
        '''------------------------------------No wiggle P(k)------------------------------------------'''
        xvec = np.log(kvec * interp_pk)
        #convert to correlation function - dst = discrete sine transform
        xvec = fft.dst(xvec, type=2)
        frec = np.arange(len(kvec)/2)
        even = xvec[::2]   #even array
        odd = xvec[1::2]   #odd array
        even_spline = UnivariateSpline(frec, even, k=3, s=0)

        result = even_spline.derivative(n=2)(frec)

        if np.any(np.isnan(result)):
            plt.subplot(121)
            plt.loglog(kvec, kvec * interp_pk)

            print(kvec)
            print(interp_pk)
            print(np.any(np.isnan(interp_pk)))

            plt.subplot(122)
            plt.loglog(kvec, xvec)

            print(xvec)
            print(rdrag)
            print(self.h, self.omega_m, self.omega_b, self.ns)
            raise Exception()

        #index of where the bao is in the data
        imin_cut = np.argmin(even_spline.derivative(n=2)(np.arange(range_imin[0],range_imin[1]))) + range_imin[0] + offset

        try:
            imax_cut = np.where(even_spline.derivative(n=2)(np.arange(range_imax[0],range_imax[1])) < threshold)[0][0] + range_imax[0]
        except IndexError:
            print(threshold)
            print(np.min(even_spline.derivative(n=2)(np.arange(range_imax[0],range_imax[1]))))
            imax_cut = range_imax[1]

        #range of where the bao is
        window = np.arange(imin_cut,imax_cut)

        #delete that window
        r = np.delete(frec, window)

        even_nobumb = np.delete(even, window)
        odd_nobumb = np.delete(odd, window)

        even_nobumb_spline = UnivariateSpline(r, (r+1)**2*even_nobumb, k=3, s=0)
        odd_nobumb_spline = UnivariateSpline(r, (r+1)**2*odd_nobumb, k=3, s=0)

        xvec[::2] = even_nobumb_spline(frec)/(frec + 1)**2
        xvec[1::2] = odd_nobumb_spline(frec)/(frec + 1)**2
        #convert back to power spectrum
        xvec = fft.idst(xvec, type=2)

        #no wiggle linear P(k)
        pkl_nw = np.exp(xvec) / kvec

        return kvec, pkl_nw, interp_pk, f  #f is the interpolation of the linear power spectrum

    def get_sigmav(self):
        """
        Calculates the displacement of the matter density field in linear theory/BAO damping factor.
        Returns
        ---------
        sigmav: float
            BAO damping factor
        """
        kvec, _, _, f = self.get_linear_nowiggle()

        #the integrand will be the interpolated power spectrum values
        #integrate the power spectrum for each cosmology
        #sigmav_quad[0] = area under curve - sigmav_quad[1] gives error on calculation
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
            Damped BAO linear theory power spectrum
        """
        sigmav = self.get_sigmav()
        kvec, pkl_nw, interp_pk, _ = self.get_linear_nowiggle()

        #calculate the gaussian damping term
        gauss = np.exp(-(kvec*sigmav)**2 * (1/2))

        #pk with damped BAOs
        pkdamp = pkl_nw + ((interp_pk - pkl_nw)*gauss)

        return pkdamp

    def get_nonlinear_nowiggle(self):
        """
        Calculates the no wiggle nonlinear power spectrum.
        Returns
        ---------
        interp_pknl: array
            Nonlinear power spectrum
        pknl_nw: array
            No wiggle nonlinear power spectrum
        """
        kvec, pkl_nw, _, _ = self.get_linear_nowiggle()
        pkdamp = self.get_linear_damped()

        xnl = self.knl
        ynl = self.pknl
        fnl = sp.interp1d(xnl, ynl, kind='quadratic', fill_value='extrapolate')

        #find pk at kvec
        interp_pknl = fnl(kvec)

        #Limit at >0
        interp_pknl[interp_pknl < 0] = 0
        pknl_nw = interp_pknl*(pkl_nw/pkdamp)

        return interp_pknl, pknl_nw

    def get_additional_params(self, kmap):
        """
        Calculates parameters additionally needed in the final equation. Calculated via COLOSSUS.
        Attributes
        ----------
        kmap: array
            The mapped kvec values according to the Peacock & Dodds 1996 formalism
        Returns
        ----------
        growthsupfactor: float
            The linear growth suppression factor for z=0
        n_L: array
            The slope of the late time power spectrum
        """
        params_colossus = {'flat':True, 'H0':self.h*100, 'Ob0':self.omega_b, 'ns':self.ns,
                           'Om0':self.omega_m, 'w0':-1, 'sigma8':self.sigma_8}
        cosmo = cosmology.setCosmology('QuijoteSR', **params_colossus)

        growthsupfactor = cosmo.growthFactorUnnormalized(0)    #growth factor -> g(Omega) = D(a)/a ---> does this give D(z)/a?????
        n_L = cosmo.matterPowerSpectrum(kmap/2, z = 0, derivative=True, model='eisenstein98_zb')

        return growthsupfactor, n_L
