import numpy as np
from Emulators.main import emulate_pknl

def test_bao_rewiggling(linear_power):
    c = linear_power

    k, pknl, pknl_nw = emulate_pknl(
        c["k"],
        c["pk"],
        c["h"],
        c["Omegam"],
        c["Omegab"],
        c["ns"],
        c["sigma8"],
        model="spine"
    )

    # Wiggles must change the spectrum
    diff = np.std(pknl - pknl_nw)
    assert diff > 0
