import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.coding import rice_encode, rice_decode, fixed_encode, fixed_decode, big_rice_encode, big_rice_decode


def test_rice():
    rng = np.random.default_rng(7)
    for k in (0, 3, 6, 11):
        v = np.rint(rng.normal(0, 2 ** k + 1, 3000)).astype(np.int64)
        data, _ = rice_encode(v, k)
        assert np.array_equal(rice_decode(data, v.size, k), v)


def test_fixed():
    rng = np.random.default_rng(8)
    v = rng.integers(-95217, 95218, 2000)
    data, _ = fixed_encode(v, 95217)
    assert np.array_equal(fixed_decode(data, v.size, 95217), v)


def test_big_rice():
    rng = np.random.default_rng(9)
    v = np.array([int(x) * (1 << 130) + int(y) for x, y in zip(rng.integers(-50, 50, 200), rng.integers(0, 1 << 60, 200))], dtype=object)
    data, _ = big_rice_encode(v, 135)
    out = big_rice_decode(data, v.size, 135)
    assert all(int(a) == int(b) for a, b in zip(out, v))
