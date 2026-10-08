import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.estimator import primal_usvp, mlwe_bits, min_std_mlwe, max_beta_msis, msis_bits


def test_kyber_primal():
    assert primal_usvp(512, 512, 3329, 1.2247) == 406
    assert primal_usvp(768, 768, 3329, 1.0) == 624
    assert primal_usvp(1024, 1024, 3329, 1.0) == 874


def test_kyber_levels():
    assert abs(mlwe_bits(256, 2, 2, 3329, 1.2247) - 118) < 2
    assert abs(mlwe_bits(256, 3, 3, 3329, 1.0) - 182) < 2
    assert abs(mlwe_bits(256, 4, 4, 3329, 1.0) - 256) < 3


def test_closed_forms_are_conservative():
    q = (1 << 20) - 3
    s = min_std_mlwe(6 * 128, 8 * 128, q)
    assert mlwe_bits(128, 6, 8, q, s * 1.001) >= 128
    b = max_beta_msis(7 * 128, 14 * 128, q)
    assert msis_bits(128, 7, 14, q, b) >= 128
