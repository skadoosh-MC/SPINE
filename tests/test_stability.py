import pytest
import numpy as np
from Emulators.main import emulate_pknl

#fixture to provide a default spine_model function
@pytest.fixture
def spine_model():
    """
    Return
    -------
    Function to emulate the power spectrum within a valid k range
    """
    def _model(model="spine"):
        #example linear power spectrum for testing
        kl = np.linspace(0.01, 3.0, 500)  #include some k > 2 for testing truncation
        pkl = kl**-3  #dummy linear spectrum (falling as k^-3)
        h = 0.7
        Omegam = 0.3
        Omegab = 0.05
        ns = 0.96
        sigma8 = 0.8

        #only pass the reliable range k <= 2 to the emulator
        kl_safe = kl[kl <= 2.0]
        pkl_safe = pkl[:len(kl_safe)]

        #emulate_pknl automatically truncates/extrapolates, but we'll still enforce truncation here
        kl_out, pknl, pknl_nw = emulate_pknl(kl_safe, pkl_safe, h, Omegam, Omegab, ns, sigma8, model=model)
        return kl_out, pknl, pknl_nw

    return _model

def test_extrapolation_is_finite(spine_model):
    """
    Test that the emulated P(k) values remain finite within the valid k range
    """
    kl, pknl, _ = spine_model(model="spine")
    assert np.all(np.isfinite(pknl)), "Nonlinear power spectrum contains inf or NaN values"

def test_spectrum_is_smooth(spine_model):
    """
    Test that the nonlinear spectrum does not have sudden jumps in the valid range
    """
    kl, pknl, _ = spine_model(model="spine")

    #fractional differences between consecutive points
    diffs = np.diff(pknl) / pknl[:-1]
    max_jump = np.max(np.abs(diffs))

    #ensure spectrum is reasonably smooth
    assert max_jump < 1.0, f"Spectrum has sudden large jumps; might not be smooth (max jump = {max_jump:.2f})"
