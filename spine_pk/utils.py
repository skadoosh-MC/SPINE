import numpy as np
import math

def get_kl(kl, pkl):
    """
    Gives the mapped k vector values according to the PD96 formulation
    Attributes
    ----------
    kl: array
        k [(h/Mpc)] values which need to be mapped using PD96 formalism
    pkl: array
        The linear power spectrum [(Mpc/h)^3] values which need to be mapped
    Retruns
    --------
    kmap: array
        wave vector [(h/Mpc)] values mapped using PD96 formalism
    """
    kmap = (1/np.cbrt((1 + ((kl**3 * pkl)/(2*np.pi**2))))) * kl
    return kmap
    
def dimensionless_pk(k,p):
    """
    Calculates the dimensionless power spectrum
    Attributes
    ----------
    k: array
        k [(h/Mpc)]
    pk: array
        Linear or Nonlinear power spectrum [(Mpc/h)^3]
    Returns
    -------
    temp: array
        The dimensionless linear or nonlinear power spectrum
    """
    temp = (k**3 * p)/(2*np.pi**2)
    return temp

def power_spectrum(k, delta):
    """
    Converts the dimensionless linear/nonlinear power spectrum into linear/nonlinear power spectrum
    Attributes
    ----------
    k: array
        k [(h/Mpc)]
    delta: array
        The dimensionless linear/nonlinear power spectrum
    Returns
    -------
    pk: array
        Regular linear/nonlinear power spectrum
    """
    pk = (delta*2*np.pi**2)/(k**3)
    return pk
