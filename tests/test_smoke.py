import numpy as np
import pytest
from emulators.main import emulate_pknl

@pytest.mark.parametrize("model", ["spine", "spinex"])
@pytest.mark.slow
def test_models_produce_output(linear_power, model):
    c = linear_power

    k, pknl, pknl_nw = emulate_pknl(
        c["k"],
        c["pk"],
        c["h"],
        c["Omegam"],
        c["Omegab"],
        c["ns"],
        c["sigma8"],
        model=model
    )

    assert len(k) > 10
    assert np.all(np.isfinite(pknl))
    assert np.all(np.isfinite(pknl_nw))
    assert np.mean(pknl) > 0
