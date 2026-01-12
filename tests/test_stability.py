# tests/test_stability.py
import pytest
import numpy as np
from Emulators.main import emulate_pknl

#----------------------------------------------------------------------
#fixture: create a default SPINE emulator setup
#----------------------------------------------------------------------
@pytest.fixture
def spine_model():
    """
    Returns
    ---------
    Callable function that wraps emulate_pknl with default parameters.
    """
    def _model(kl=None, model="spine"):
        # default k-vector
        if kl is None:
            kl = np.linspace(0.01, 3.0, 100)  # includes extrapolation beyond k=2
        # default cosmology parameters within COSMO_BOUNDS
        h = 0.7
        Omegam = 0.3
        Omegab = 0.05
        ns = 0.96
        sigma8 = 0.8

        # call emulator
        return emulate_pknl(kl, kl**ns, h, Omegam, Omegab, ns, sigma8, model)
    return _model

#----------------------------------------------------------------------
#test 1: Check that the extrapolated spectrum is finite
#----------------------------------------------------------------------
def test_extrapolation_is_finite(spine_model):
    kl, pknl, pknl_nw = spine_model(model="spine")
    assert np.all(np.isfinite(pknl)), "Nonlinear P(k) contains NaNs or infs"
    assert np.all(np.isfinite(pknl_nw)), "Nonlinear no-wiggle P(k) contains NaNs or infs"

#----------------------------------------------------------------------
#test 2: Check that the spectrum is smooth (no huge jumps)
#----------------------------------------------------------------------
def test_spectrum_is_smooth(spine_model):
    kl, pknl, _ = spine_model(model="spine")
    diffs = np.diff(pknl) / pknl[:-1]  # fractional differences
    max_jump = np.max(np.abs(diffs))
    assert max_jump < 1.0, "Spectrum has sudden large jumps; might not be smooth"

#----------------------------------------------------------------------
#test 3: Check that SPINEX model also returns finite spectrum
#----------------------------------------------------------------------
def test_spinex_is_finite(spine_model):
    kl, pknl, pknl_nw = spine_model(model="spinex")
    assert np.all(np.isfinite(pknl)), "SpinEx P(k) contains NaNs or infs"
    assert np.all(np.isfinite(pknl_nw)), "SpinEx no-wiggle P(k) contains NaNs or infs"
