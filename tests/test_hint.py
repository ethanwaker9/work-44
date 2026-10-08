import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.hint import (posterior, brute_posterior, gaussian_pmf_on, tight_sigma0_sq, split_distance,
                            uniform_hint_attack, uniform_hint_count, log_det_width_sq, jensen_width_sq,
                            reduction_secret)
from tighthint.challenges import random_challenges, embed_sq, spectral_threshold


def test_posterior_lemma():
    M = np.array([[2.0, 1.0], [0.0, 3.0]])
    h = np.array([7.0, -4.0])
    X, p = brute_posterior(M, h, 6.0, 9.0, 40)
    Sigma, mu = posterior(M, h, 6.0, 9.0)
    q = gaussian_pmf_on(X.astype(float), Sigma, mu)
    assert np.max(np.abs(p - q)) < 1e-12


def test_tight_split():
    eps = 2.0 ** -64
    eta = math.sqrt(math.log(2 * (1 + 1 / eps)) / math.pi)
    for P in (64.0, 144.0):
        s02 = tight_sigma0_sq(P, eta)
        assert s02 >= P - 2 * eta * eta and s02 >= P / 2
        assert split_distance(P, s02) <= 8 * eps
        assert split_distance(P, 0.995 * P) > 8 * eps


def test_reduction_moments():
    rng = np.random.default_rng(11)
    M = np.array([[1.0, 2.0]])
    s1, s2 = 20.0, 30.0
    Sigma, _ = posterior(M, np.array([0.0]), s1, s2)
    P = float(np.min(np.linalg.eigvalsh(Sigma)))
    eta = 3.0
    s0 = math.sqrt(tight_sigma0_sq(P, eta))
    h = np.array([13.0])
    Sigma, mu = posterior(M, h, s1, s2)
    xs = np.array([reduction_secret(rng, M, h, s1, s2, s0, np.rint(rng.normal(0, s0 / math.sqrt(2 * math.pi), 2)).astype(np.int64), eta) for _ in range(20000)])
    assert np.max(np.abs(xs.mean(axis=0) - mu)) < 0.3
    cov = np.cov(xs.T)
    assert np.max(np.abs(cov - Sigma / (2 * math.pi))) / np.max(Sigma / (2 * math.pi)) < 0.08


def test_uniform_attack():
    rng = np.random.default_rng(12)
    u, N = 16, 256
    Q = uniform_hint_count(u, N, 20)
    x = rng.integers(-1000, 1000, N)
    hints = x + rng.integers(-u, u + 1, size=(Q, N))
    est, _ = uniform_hint_attack(hints, u)
    assert np.array_equal(est, x)


def test_determinant_bound():
    rng = np.random.default_rng(13)
    s1, s2, eta, kappa = 22.21, 248.78, math.sqrt(31.73), 31
    thr = spectral_threshold(128, kappa)
    g = random_challenges(rng, 128, kappa, 2000)
    x = embed_sq(g)
    x = x[np.max(x, axis=1) <= thr]
    j = jensen_width_sq(s1, s2, kappa, eta)
    assert all(log_det_width_sq(row, s1, s2, eta) >= j - 1e-9 for row in x)
