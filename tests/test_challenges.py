import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.challenges import (random_challenges, embed_sq, spectral_threshold, spectral_ratio, kappa_min,
                                  challenge_space_log2, hash_to_challenge, spectral_sq, worst_unfiltered_s1)


def test_threshold_equation():
    for n in (64, 128, 256):
        x = spectral_ratio(n)
        assert abs(x - math.log(x) - math.log(n * math.e)) < 1e-9


def test_fraction_at_least_half():
    rng = np.random.default_rng(14)
    n, kappa = 128, kappa_min(128, 128, 1.0)
    assert kappa == 31 and challenge_space_log2(n, kappa) - 1 >= 128
    g = random_challenges(rng, n, kappa, 50000)
    frac = np.mean(np.max(embed_sq(g), axis=1) <= spectral_threshold(n, kappa))
    assert frac >= 0.5


def test_moment_generating_function():
    n = 128
    for j in (0, 17, 63):
        for theta in (0.0, 0.5, 1.0):
            ang = theta + math.pi * (2 * j + 1) * np.arange(n) / n
            a = np.cos(ang)
            for t in (0.5, 2.0, 5.0, 12.0):
                assert np.mean(np.cosh(t * a)) <= math.exp(t * t / 4) * (1 + 1e-12)


def test_hash_to_challenge():
    thr = spectral_threshold(128, 31)
    for i in range(20):
        g = hash_to_challenge(bytes([i]) * 8, 128, 31, thr)
        assert np.count_nonzero(g) == 31 and spectral_sq(g) <= thr


def test_worst_case():
    g = np.zeros(128, dtype=np.int64)
    g[:31] = 1
    assert abs(spectral_sq(g) - worst_unfiltered_s1(128, 31) ** 2) < 1e-6
