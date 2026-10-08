import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint import su


def test_bound_dominates_exact():
    for N, T, c, alpha, a in ((256, 4, 1, 2.0, 20), (512, 6, 2, 4.0, 60), (1024, 8, 3, 8.0, 200), (1024, 3, 1, 2.0, 100)):
        bound, eps = su.renyi_bound(N, T, c, alpha, a)
        exact, e1, e2 = su.renyi_restricted(N, T, c, alpha, a)
        assert exact <= bound
        assert abs(e1 - eps) < 1e-12 and abs(e2 - eps) < 1e-12


def test_kappa_limits():
    assert abs(su.kappa_continuous_exact(4) - 1.03858) < 1e-5
    assert abs(su.kappa_discrete(2 ** 10, 8) - su.kappa_continuous_exact(8)) < 1e-5
    assert abs(su.kappa_discrete(2 ** 12, 2) - su.kappa_discrete(2 ** 8, 2) - math.log(16) / 3) < 0.05


def test_fisher_is_exact_constant():
    N, T, c, alpha = 2 ** 12, 8, 1, 2.0
    exact = su.renyi_restricted(N, T, c, alpha, 400)[0]
    predicted = alpha * c * c * su.kappa_discrete(N, T) / (2 * su.su_variance(N, T))
    assert abs(exact / predicted - 1) < 0.01
