import pytest
from otsensitivity import cosi


def test_cosi(ishigami):
    model, sample, data = ishigami

    cosi_ = cosi(sample, data)
    assert cosi_ == pytest.approx([0.326, 0.444, 0.014], abs=0.2)
