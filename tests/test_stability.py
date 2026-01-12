import numpy as np
import pytest

# Extrapolation must be numerically stable -------------

def test_extrapolation_is_finite(spine_model):
    """
    Outside the training range (k < 0.01 or k > 2),
    the emulator may be inaccurate but must remain
    numerically well-behaved.
    """

    k = np.concatenate([
        np.logspace(-4, -2, 80),   # well below training range
        np.logspace(0.3, 2.0, 80)  # well above training range
    ])

    P = spine_model.emulate(k)

    assert np.all(np.isfinite(P)), "Extrapolation produced NaN or Inf values"
    assert np.all(P > 0), "Power spectrum became non-positive in extrapolation"

# Spectrum must be smooth inside the training range ----------

def test_spectrum_is_smooth(spine_model):
    """
    Inside the validated range, the spectrum should
    not show unphysical oscillations or spikes.
    """

    k = np.logspace(-2, 0.3, 300)   # 0.01 < k < 2
    P = spine_model.emulate(k)

    # Relative change between neighboring bins
    rel = np.abs(np.diff(P) / P[:-1])

    # Median change should be modest
    assert np.median(rel) < 0.5, "Spectrum shows unphysical oscillations"
