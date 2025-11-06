import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import logging
log = logging.getLogger(__name__)

class GrowthCalculator:
    """Computes the linear growth of structure for a given cosmology.

    This class solves the differential equation for the linear growth factor
    for a flat or non-flat universe with a w0-wa dark energy model. It
    pre-computes the solution over a range of scale factors and uses spline
    interpolation to provide fast and accurate values for various growth-related
    quantities like D(z), f(z), and derived parameters required by the emulator.

    Parameters
    ----------
    cospar : dict
        A dictionary of cosmological parameters. Expected keys are:
        'h' : float, the Hubble parameter H0 / 100.
        'omega_k' : float, the curvature density parameter, omega_k.
        'omega_c' : float, the physical cold dark matter density, omega_c * h^2.
        'omega_b' : float, the physical baryon density, omega_b * h^2.
        'w_0' : float, the dark energy equation of state parameter.
        'w_a' : float, the dark energy equation of state evolution parameter.

    Attributes
    ----------
    ga_spline : scipy.interpolate.CubicSpline
        A spline for the modified growth function g(ln a) = D(a)/a.
    dga_spline : scipy.interpolate.CubicSpline
        A spline for the derivative dg/d(ln a).
    xz_spline : scipy.interpolate.CubicSpline
        An inverse spline to find redshift z as a function of ln(D).
    """
    def __init__(self, cospar, ln_a_min=-7.0, ln_a_max=0.53, num_steps=1701):
        log.info("GrowthCalculator initialized.")
        # Set cosmological parameters
        self.h0 = cospar['h']
        self.ok0 = cospar['omega_k']/self.h0**2
        self.ocb0 = (cospar['omega_c'] + cospar['omega_b']) / self.h0**2
        self.onuh2 = 0. # Assuming no massive neutrinos
        self.om0 = self.ocb0 + self.onuh2 / self.h0**2
        self.or0 = 0. # Assuming negligible radiation
        self.ode0 = (1. - self.ok0 - self.om0 - self.or0) #cospar['omega_de']
        #self.neff = cospar['Num_nu'] if self.onuh2 == 0 else cospar['Num_nu'] + 1.0
        self.neff = 3.046
        self.w0 = cospar['w_0']
        self.wa = cospar['w_a']
        self.clight = 299792.458  # Speed of light in km/s
        
        # Arrays for storing values
        # x = ln(a) from ln_a_min to ln_a_max
        self.xa = np.linspace(ln_a_min, ln_a_max, num_steps)
        #self.ga = np.zeros_like(self.xa)
        #self.dga = np.zeros_like(self.xa)

        if np.any(np.isinf(self.xa)) or np.any(np.isnan(self.xa)):
            raise ValueError("Invalid values in x: check input ranges.")
        
        # Initialize growth factor
        self._setup_growth()

    def _setup_growth(self):
        """Solves the growth ODE and initializes the spline interpolators.
        
        This is an internal method called automatically during initialization.
        """
        log.info("Solving growth ODE...")
        # Initialize growth factor and its derivative
        y0 = [1.0, 0.0]  # Initial condition: g(ln(a)) = 1 at ln(a) = -5, g'(ln(a)) = 0

        # Solve the ODE using scipy's solve_ivp
        sol = solve_ivp(self._fderiv, [self.xa[0], self.xa[-1]], y0, t_eval=self.xa, 
                        method='RK45', rtol=1e-8, atol=1e-10, 
                        max_step=0.01, dense_output=True)

        # Store results
        self.ga = sol.y[0]
        self.dga = sol.y[1]
        # Spline interpolation for ga and dga
        self.ga_spline = CubicSpline(self.xa, self.ga)
        self.dga_spline = CubicSpline(self.xa, self.dga)

        # Spline for x(z)
        z = 1.0 / np.exp(self.xa) - 1.
        lnD = np.log(self.ga / (1. + z))
        self.xz_spline = CubicSpline(lnD, z)
        log.debug("ODE solved and splines created.")

    def _fderiv(self, x, y):
        """Defines the system of first-order ODEs for the growth function g(ln a).

        Parameters
        ----------
        x : float
            The independent variable, x = ln(a).
        y : list or ndarray
            The state vector [g, dg/dx].

        Returns
        -------
        list
            The derivatives [dg/dx, d^2g/dx^2].
        """
        # ODE system for growth factor
        a = np.exp(x)  # Convert x = ln(a) to a
        # E^2 and Omega_k as a function of a
        e2 = self._E2(a)
        ok = self.ok0 / (e2 * a**2)
        # w(a)
        wat_a = self.w0 + self.wa * (1.0 - a)
        
        # Effective w for the dark energy density term
        w_eff = np.where(
            np.isclose(a, 1.0, atol=1e-6), 
            self.w0, 
            self.w0 + self.wa * (1.0 + (1.0 - a) / np.log(a))
        )
        # Dark energy density term
        ode = self.ode0 / (e2 * a**(3 * (1 + w_eff)))

        # Derivatives
        dy1 = y[1]
        dy2 = -(2.5 + 0.5 * (ok - 3 * wat_a * ode)) * y[1] - (2 * ok + 1.5 * (1 - wat_a) * ode) * y[0]
        return [dy1, dy2]

    def _E2(self, a):
        """Calculates the squared Hubble parameter normalized by H0, E(a)^2 = (H(a)/H0)^2.

        Parameters
        ----------
        a : float or ndarray
            The scale factor.

        Returns
        -------
        float or ndarray
            The value of E(a)^2.
        """
        w_eff = np.where(
            np.isclose(a, 1.0, atol=1e-6), 
            self.w0, 
            self.w0 + self.wa * (1.0 + (1.0 - a) / np.log(a))
        )

        # E^2 as a function of a
        arg = 187000.0 * a * self.onuh2
        E2 = (self.ocb0 / a**3 +
              self.ok0 / a**2 +
              self.ode0 / a**(3 * (1 + w_eff)) +
              self.or0 / a**4 * (1 + 0.2271 * self.neff * self.func(arg)))
        return E2
    
    def g(self, z):
        """Computes the growth suppression factor g(z) = D(z) * (1+z).

        This quantity is equivalent to g(a) = D(a) / a.

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        float or ndarray
            The value of the modified growth function g(z).
        """
        # Interpolates and returns g(z) = D(z)(1+z)
        x = -np.log(1 + z)
        return self.ga_spline(x)

    def dgdlna(self, z):
        """Computes the derivative of the suppression factor, dg/d(ln a).

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        float or ndarray
            The value of the derivative dg/d(ln a).
        """
        # Interpolates and returns dg/dlna(z)
        x = -np.log(1 + z)
        return self.dga_spline(x)

    def fgrowth(self, z):
        """Computes the logarithmic growth rate, f(z) = d(ln D) / d(ln a).

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        float or ndarray
            The growth rate f(z).
        """
        # Returns the growth rate f(z) = dln(D)/dln(a)
        return self.dgdlna(z) / self.g(z) + 1.0

    def X(self, z):
        """Computes the combination X(z) = Omega_m(z) / f(z)^2.

        This quantity is a key input parameter for the emulator's response to
        dark energy and modified gravity.

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        float or ndarray
            The value of X(z).
        """
        # Returns the growth rate f(z) = dln(D)/dln(a)
        a = 1./(1. + z)
        Om_z = self.om0/a**3/self._E2(a)
        return Om_z/self.fgrowth(z)**2 

    def X_tau(self, tau):
        """Computes x(z)=Omega_m(z)/f(z)^2 as a function of the time 
        variable tau = ln(D(z)).

        Parameters
        ----------
        tau : float or ndarray
            The time variable, defined as the natural logarithm of the
            linear growth factor, ln(D).

        Returns
        -------
        float or ndarray
            The value of X at the redshift corresponding to the given tau.
        """
        # Returns X=Omega_m(z)/f(z)^2 at tau = ln(D)
        z = self.xz_spline(tau)
        Xval = self.X(z)
        return Xval

    def Dgrowth(self, z):
        """Computes the linear growth factor D(z).

        The growth factor is normalized such that D(a) = a during the
        matter-dominated era at high redshift. 

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        float or ndarray
            The linear growth factor D(z).
        """
        # Returns D = g(z)/(1+z)
        return self.g(z)/(1+z)
    
    @staticmethod
    def func(y):
        # Function based on equation from Komatsu et al. (2009)
        a, p = 0.3173, 1.83
        pinv = 1.0 / p
        return (1.0 + (a * y)**p)**pinv


