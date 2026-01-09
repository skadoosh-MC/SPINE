import numpy as np
import pytest
import camb

@pytest.fixture(scope="session")
def linear_power():
    """
    Provides a CAMB-generated linear power spectrum
    that is physically consistent.
    """
    pars = camb.set_params(
        H0=67.5,
        ombh2=0.022,
        omch2=0.122,
        ns=0.965,
        As=2.1e-9
    )
    pars.set_matter_power(redshifts=[0], kmax=3.0)

    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=1e-3, maxkh=3.0, npoints=300
    )

    return {
        "k": kh,
        "pk": pk[0],
        "h": 0.675,
        "Omegam": (0.022 + 0.122) / 0.675**2,
        "Omegab": 0.022 / 0.675**2,
        "ns": 0.965,
        "sigma8": results.get_sigma8()
    }
