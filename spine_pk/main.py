import warnings
warnings.filterwarnings(
    "once",
    message="Input kl extends beyond SPINE/SPINEX emulation range"
)
from Emulators.class_spinex import Power, XPower
from Emulators.utils import dimensionless_pk, get_kl, power_spectrum
from Emulators.growth import GrowthCalculator
import numpy as np
from scipy.interpolate import UnivariateSpline
import warnings
from numba import njit
from random import random

COSMO_BOUNDS = {"h": (0.5, 0.89),
                "Omegam": (0.1, 0.5),
                "Omegab": (0.03, 0.07),
                "ns": (0.8, 1.2),
                "sigma8": (0.6, 1.0)}

def create_default_power(kl, pkl, h, Omegam, Omegab, ns, sigma8, model):
    '''Creates class instance when called
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
    model: string
        Which model to use. 
        I) 'spine' uses the parameters θ = {h, Omegam, Omegab, ns, sigma8, nL, ga} where:
        nL: Slope of the late time power spectrum calculated at kmap/2 where 
            kmap is  calculated using the Peacock and Dodds 1996 formalism
        ga: Growth suppression factor defined as the suppression in growth when comapred to 
            an Einstein de Sitter Universe. It is defined as g(a) = D(a)/a.
            
        II) 'spinex' uses the parameters θX = {omegam, fb, ns, sigma12, nL, X}, where:
        omegam : Physical matter density
        fb     : Baryon fraction defined as Omegab/Omegam
        sigma12: Root-mean-square density fluctuation when the linearly 
                 evolved field is smoothed with a top-hat filter of radius 12 Mpc
        X      : XTilde: Encodes information about the dependance of the nonlinear evolution of the 
        density field.
    Returns
    -------
    Class instance to calculate the repective parameters.
    '''
    #------------------------------------------------------------------------------------------------------
    #check for supicious units
    if np.mean(pkl) < 1e-10 or np.mean(pkl) > 1e6:
        warnings.warn("Input P(k) values look suspicious. Expected units are (Mpc/h)^3.", RuntimeWarning)
    #------------------------------------------------------------------------------------------------------

    if model == 'spine':
        return Power(kl, pkl, h, Omegam, Omegab, ns, sigma8)
    elif model == 'spinex':
        return XPower(kl, pkl, h, Omegam, Omegab, ns, sigma8)

@njit
def f_spine(Delta_l, h, Omegam, Omegab, ns, sigma8, nl, ga):
    '''Symbolic expression for the SPINE model
    Attributes
    ----------
    deltal: array
        The dimensionless linear power spectrum
    h: float
        Hubble constant, H0, divided by 100 km/s/Mpc 
    Omegam: float
        Matter density of the universe at z = 0
    Omegab: float
        Baryon denisty of the universe at z = 0
    ns: float
        Spectral tile of the primordial spectrum
    sigma8: float
        Root-mean-square density fluctuation when the linearly 
        evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    nL: array
        Slope of the late-time power spectrum calculated at kmap/2 where 
        kmap is  calculated using the Peacock and Dodds 1996 formalism
    ga: float
        Growth suppression factor defined as the suppression in growth when comapred to 
        an Einstein de Sitter Universe. It is defined as g(a) = D(a)/a.
    Returns
    -------
    full: array
        The dewiggled dimensionless nonlinear power spectrum from SPINE
    '''
    a1 = 162.45 + ((-nl**3 - (3.76/Delta_l))/(Omegab-Omegam))
    PI = -nl + (((9.13**nl * ns)**Delta_l * a1)/(sigma8*ga))
    fdel = (4.16**nl * ga**ns * (PI**Delta_l + 4.24))**nl
    fdel2 = Delta_l*(fdel-0.16) - nl
    full = Delta_l + (Delta_l**2.64 * fdel2)

    return full

@njit
def f_spinex(deltal, omegam, fb, ns, sigma12, nL, X):
    '''Symbolic expression for the SPINEX model
    Attributes
    ----------
    deltal: array
        The dimensionless linear power spectrum
    omegam: float
        physical matter density of the universe
    fb: float
        Baryon fraction of the universe defined as Oegam/Omegab
    ns: float
        Spectral tilt of the primordial spectrum
    sigma12: float
        Root-mean-square density fluctuation when the linearly 
        evolved field is smoothed with a top-hat filter of radius 12 Mpc
    nL: array
        Slope of the late-time power spectrum calculated at kmap/2 where 
        kmap is an array of wavevectors calculated using the Peacock and 
        Dodds 1996 formalism
    X: float
        XTilde: Encodes the evolution of the dependency of the nonlinear evolution of the
        density field
    Returns
    -------
    full: array
        The dewiggled dimensionless nonlinear power spectrum prediction from SPINEX
    '''
    k1 = (((ns**X + fb**(-0.154) - 0.652)**sigma12 + X + fb)*X)**(nL**2)
    k2 = (fb - X**(-15.169))/(sigma12)
    k = 0.085*(k1 + k2)
    e = (X + fb - 0.339)/(deltal**2)
    term2 = ((deltal**4.305)/(ns)) * (k + e)
    full = deltal + term2

    return full

def get_spectrum(kl, pkl, h, Omegam, Omegab, ns, sigma8, model, k_max = 2.0):
    '''Calculates the final prediction for the nonlinear matter power spectrum
    based on model choice
    Attributes
    ----------
    kl: array
        k [(h/Mpc)] for the linear matter power spectrum
    pkl: array
        The linear power spectrum [(Mpc/h)^3]
    h: float
        Hubble constant, H0, divided by 100 km/s/Mpc 
    Omegam: float
        Matter density of the universe at z = 0
    Omegab: float
        Baryon denisty of the universe at z = 0
    ns: float
        Spectral tilt of the primordial spectrum 
    sigma8: float
        Root-mean-square density fluctuation when the linearly 
        evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    model: string
        Which model to use. 
        I) 'spine' uses the parameters θ = {h, Omegam, Omegab, ns, sigma8, nL, ga} where:
        nL: Slope of the late time power spectrum calculated at kmap/2 where 
            kmap is  calculated using the Peacock and Dodds 1996 formalism
        ga: Growth suppression factor defined as the suppression in growth when comapred to 
            an Einstein de Sitter Universe. It is defined as g(a) = D(a)/a.
            
        II) 'spinex' uses the parameters θX = {omegam, fb, ns, sigma12, nL, X}, where:
        omegam : Physical matter density
        fb     : Baryon fraction defined as Omegab/Omegam
        sigma12: Root-mean-square density fluctuation when the linearly 
                 evolved field is smoothed with a top-hat filter of radius 12 Mpc
        X      : XTilde: Encodes information about the dependance of the nonlinear evolution of the 
        density field. 
    k_max: int
        The value at which to cut off k.
    Returns:
    --------
    knl: array
        The nonlinear wavevector [(h/Mpc)]
    model_pknl: array
        The model prediction for the nonlinear power spectrum [(Mpc/h)^3]
    model_pknl_nw: array
        The model prediction for the dewiggled nonlinear power spectrum [(Mpc/h)^3]
    '''
    #--------------------------------------------------------------------------------------
    #check for parameter range violation
    for name, (low, high) in COSMO_BOUNDS.items():
        val = locals()[name]
        if not (low <= val <= high):
            raise ValueError(f"Parameter {name}={val} is outside [{low}, {high}]")
    #--------------------------------------------------------------------------------------

    #create instance of power class
    if model == 'spine':
        class_instance = create_default_power(kl, pkl, h, Omegam, Omegab, ns, sigma8, model='spine')
    elif model == 'spinex':
        class_instance = create_default_power(kl, pkl, h, Omegam, Omegab, ns, sigma8, model='spinex')
        _, sigma12 = class_instance.set_cosmo()
        X = class_instance.get_xtilde(z = 0)
        fb = Omegab/Omegam
        omegam = Omegam * h**2

    #get the wavevector, linear no wiggle power spectrum, true og linear power spectrum & linear damped spectrum
    kvec, pklnw, _, _ = class_instance.get_linear_nowiggle()
    pkldamp = class_instance.get_linear_damped()

    # --- truncate arrays at k_max ---
    mask = kvec <= k_max
    kvec = kvec[mask]
    pklnw = pklnw[mask]
    pkldamp = pkldamp[mask]

    #convert input to dimensionless P(k) - do not get the mapped inputs 
    deltal = dimensionless_pk(kvec, pklnw)

    #get the growth factor and the slope of the late time power spectrum at the mapped k
    ga, nl = class_instance.get_additional_params(kvec)

    #get the prediction for the nonlinear nowiggle power spectrum
    if model == 'spine':
        delta_nlnw_pred = f_spine(deltal, h, Omegam, Omegab, ns, sigma8, nl, ga)
    elif model == 'spinex':
        delta_nlnw_pred = f_spinex(deltal, omegam, fb, ns, sigma12, nl, X)

    #get knl <--- the mapped version
    knl = kvec*(np.cbrt(1+delta_nlnw_pred))

    #convert to nonlinear no wiggle power spectrum
    model_pknl_nw = power_spectrum(knl, delta_nlnw_pred)

    #rewiggle the power spectrum
    eps = 1e-8 
    safe_pklnw = np.maximum(pklnw, eps)
    model_pknl = model_pknl_nw * (pkldamp / safe_pklnw)

    #return kvector, PkNL and no wiggle PkNL
    return knl, model_pknl, model_pknl_nw

# def emulate_pknl(kl, pkl, h, Omegam, Omegab, ns, sigma8, model):
    # """Calculates the emulated nonlinear matter power spectrum
    # Attributes
    # ----------
    # kl: array
    #     k (h/Mpc) for the linear matter power spectrum
    # pkl: array
    #     linear power spectrum
    # h: float
    #     Hubble constant, H0, divided by 100 km/s/Mpc 
    # Omegam: float
    #     Matter density of the universe
    # Omegab: float
    #     Baryon denisty of the universe
    # ns: float
    #     Spectral index 
    # sigma8: float
    #     Root-mean-square density fluctuation when the linearly 
    #     evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    # model: 
    #     Chose between SPINE or SPINEX
    # Returns
    # -------
    # kl: array
    #     array of wavevector
    # spine_pknl: array
    #     The emulated nonlinear power spectrum
    # spine_pknlnw: array
    #     The emulated nonlinear power spectrum that is dewiggled
    # """
    # k_raw, _pknl_raw, _pknlnw_raw = get_spectrum(kl, pkl, h, Omegam, Omegab, ns, sigma8, model)

    # #make it so that the input and output have the same shape
    # spine_pknl = np.interp(kl, k_raw, _pknl_raw)
    # spine_pknlnw = np.interp(kl, k_raw, _pknlnw_raw)

    # return kl, spine_pknl, spine_pknlnw

def emulate_pknl(kl, pkl, h, Omegam, Omegab, ns, sigma8, model):
    """
    Calculates the emulated nonlinear matter power spectrum. Automatically truncates beyond the SPINE/SPINEX 
    valid k-range (0.01 <= k <= 2 h/Mpc) and interpolates to match input kl array.
    Attributes
    ----------
    kl: array
        k [(h/Mpc)] for the linear matter power spectrum
    pkl: array
        The linear power spectrum [(Mpc/h)^3]
    h: float
        Hubble constant, H0, divided by 100 km/s/Mpc 
    Omegam: float
        Matter density of the universe at z = 0
    Omegab: float
        Baryon denisty of the universe at z = 0
    ns: float
        Spectral tilt of the primordial spectrum
    sigma8: float
        Root-mean-square density fluctuation when the linearly 
        evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    model: string
        Which model to use. 
        I) 'spine' uses the parameters θ = {h, Omegam, Omegab, ns, sigma8, nL, ga} where:
        nL: Slope of the late time power spectrum calculated at kmap/2 where 
            kmap is  calculated using the Peacock and Dodds 1996 formalism
        ga: Growth suppression factor defined as the suppression in growth when comapred to 
            an Einstein de Sitter Universe. It is defined as g(a) = D(a)/a.
            
        II) 'spinex' uses the parameters θX = {omegam, fb, ns, sigma12, nL, X}, where:
        omegam : Physical matter density
        fb     : Baryon fraction defined as Omegab/Omegam
        sigma12: Root-mean-square density fluctuation when the linearly 
                 evolved field is smoothed with a top-hat filter of radius 12 Mpc
        X      : XTilde: Encodes information about the dependance of the nonlinear evolution of the 
        density field.
    Returns
    -------
    kl : array
        The wavevectro at whihc the nonlinear power spectrum is evaluated
    pknl : array
        The nonlinear power spectrum [(Mpc/h)^3]
    pknl_nw : array
        The nonlinear no-wiggle power spectrum [(Mpc/h)^3]
    """
    #determine the effective k_max based on the user input and SPINE limits
    k_min, k_max = 1e-2, 2.0
    k_max_eff = min(np.max(kl), k_max)

    #compute full spectrum up to k_max_eff
    knl_full, pknl_full, pknl_nw_full = get_spectrum(kl, pkl, h, Omegam, Omegab, ns, sigma8, model=model, k_max=k_max_eff)

    #interpolate results back to user kl array
    pknl = np.interp(kl, knl_full, pknl_full, left=pknl_full[0], right=pknl_full[-1])
    pknl_nw = np.interp(kl, knl_full, pknl_nw_full, left=pknl_nw_full[0], right=pknl_nw_full[-1])

    #warn if user kl exceeds SPINE/SPINEX limits
    if np.any(kl < k_min) or np.any(kl > k_max):
        warnings.warn(f"Input kl extends beyond SPINE/SPINEX emulation range [{k_min}, {k_max}] h/Mpc. Extrapolation applied.", UserWarning)

    return kl, pknl, pknl_nw
