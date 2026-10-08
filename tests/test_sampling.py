import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.sampling import DiscreteGaussian, CenteredSampler, dgauss_pmf


def tv(samples, s, c):
    lo, hi = int(samples.min()), int(samples.max())
    xs = np.arange(lo - 5, hi + 6)
    p = dgauss_pmf(s, c, xs)
    emp = np.bincount(samples - (lo - 5), minlength=len(xs))[: len(xs)] / len(samples)
    return 0.5 * np.abs(emp - p).sum()


def test_discrete_gaussian():
    rng = np.random.default_rng(5)
    for s in (8.0, 22.21, 248.78):
        x = DiscreteGaussian(s).sample(rng, 400000)
        assert tv(x, s, 0.0) < 0.02 + 3 * math.sqrt(s / 400000.0)
        assert abs(np.var(x) / (s * s / (2 * math.pi)) - 1) < 0.02


def test_centered_sampler():
    rng = np.random.default_rng(6)
    cs = CenteredSampler(7.0)
    for c in (0.0, 0.25, 0.5, 0.9):
        x = cs.sample(rng, np.full(300000, c))
        assert tv(x, 7.0, c) < 0.02
