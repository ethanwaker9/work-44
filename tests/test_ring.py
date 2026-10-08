import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.ring import Ring, BigRing, make_ring, prime_5mod8, schoolbook, is_prime


def test_primes():
    for bits in (20, 24, 37, 46):
        q = prime_5mod8(bits)
        assert q % 8 == 5 and is_prime(q) and q < (1 << bits)


def test_multiplication():
    rng = np.random.default_rng(1)
    for bits in (20, 24, 31, 37, 46, 55, 61):
        q = prime_5mod8(bits)
        R = make_ring(128, q)
        a = R.uniform(rng, (1,))[0]
        b = R.uniform(rng, (1,))[0]
        assert all(int(x) == int(y) for x, y in zip(R.mul(a, b), schoolbook(a, b, q)))


def test_big_multiplication():
    rng = np.random.default_rng(2)
    q = prime_5mod8(213)
    R = make_ring(128, q)
    assert isinstance(R, BigRing)
    a = R.uniform(rng, (1,))[0]
    b = R.uniform(rng, (1,))[0]
    assert all(int(x) == int(y) for x, y in zip(R.mul(a, b), schoolbook(a, b, q)))


def test_matvec():
    rng = np.random.default_rng(3)
    q = prime_5mod8(20)
    R = Ring(128, q)
    M = R.uniform(rng, (3, 4))
    v = R.uniform(rng, (4,))
    out = R.matvec(M, v)
    for i in range(3):
        acc = np.zeros(128, dtype=np.int64)
        for j in range(4):
            acc = (acc + schoolbook(M[i, j], v[j], q)) % q
        assert np.array_equal(out[i], acc)


def test_spectral_norm():
    rng = np.random.default_rng(4)
    R = Ring(64, prime_5mod8(20))
    a = rng.integers(-3, 4, 64)
    rot = np.array([[(a[(i - j) % 64] if i >= j else -a[(i - j) % 64]) for j in range(64)] for i in range(64)], dtype=float)
    assert abs(np.linalg.norm(rot, 2) ** 2 - R.spectral_sq(a)) < 1e-6
    assert abs(np.sum(np.abs(R.embedding(a)) ** 2) - 64 * np.sum(a * a)) < 1e-6
