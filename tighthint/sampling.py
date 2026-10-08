import math
import os
import numpy as np
from scipy.special import erf, erfc


def make_rng(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(16), "little")
    return np.random.default_rng(seed)


class DiscreteGaussian:
    def __init__(self, s, center=0.0):
        self.s = float(s)
        self.center = float(center)
        self.sd = self.s / math.sqrt(2 * math.pi)
        self._build()

    def _proposal_pmf(self, x):
        x = np.asarray(x, dtype=np.float64)
        k = self.sd * math.sqrt(2)
        z1 = (x + 0.5 - self.center) / k
        z0 = (x - 0.5 - self.center) / k
        right = 0.5 * (erfc(z0) - erfc(z1))
        left = 0.5 * (erfc(-z1) - erfc(-z0))
        mid = 0.5 * (erf(z1) - erf(z0))
        return np.where(z0 >= 0, right, np.where(z1 <= 0, left, mid))

    def _target(self, x):
        return np.exp(-np.pi * (x - self.center) ** 2 / self.s ** 2)

    def _build(self):
        span = int(math.ceil(9 * self.sd)) + 2
        xs = np.arange(math.floor(self.center) - span, math.floor(self.center) + span + 1)
        ratio = self._target(xs) / np.maximum(self._proposal_pmf(xs), 1e-300)
        self.m = float(np.max(ratio)) * (1 + 1e-12)

    def sample(self, rng, size):
        size = int(np.prod(size)) if not isinstance(size, int) else size
        out = np.empty(0, dtype=np.int64)
        need = size
        while need > 0:
            batch = int(need * 1.2) + 16
            x = np.rint(rng.normal(self.center, self.sd, batch)).astype(np.int64)
            acc = rng.random(batch) < self._target(x) / (self.m * np.maximum(self._proposal_pmf(x), 1e-300))
            out = np.concatenate([out, x[acc]])
            need = size - out.shape[0]
        return out[:size]


def dgauss(rng, s, shape):
    shape = tuple(shape) if not isinstance(shape, int) else (shape,)
    return DiscreteGaussian(s).sample(rng, int(np.prod(shape))).reshape(shape)


def uniform_centered(rng, bound, shape):
    return rng.integers(-bound, bound + 1, size=shape, dtype=np.int64)


def ternary(rng, shape):
    return rng.integers(-1, 2, size=shape, dtype=np.int64)


def sum_of_uniforms(rng, u, T, shape):
    half = 1 << (u - 1)
    acc = np.zeros(shape, dtype=np.int64)
    for _ in range(T):
        acc += rng.integers(-half, half, size=shape, dtype=np.int64)
    return acc


def cbd(rng, eta, shape):
    a = rng.integers(0, 2, size=tuple(shape) + (eta,), dtype=np.int64).sum(axis=-1)
    b = rng.integers(0, 2, size=tuple(shape) + (eta,), dtype=np.int64).sum(axis=-1)
    return a - b


def dgauss_pmf(s, center, xs):
    w = np.exp(-np.pi * (np.asarray(xs, dtype=np.float64) - center) ** 2 / s ** 2)
    return w / w.sum()


class CenteredSampler:
    def __init__(self, r):
        self.r = float(r)
        self.sd = self.r / math.sqrt(2 * math.pi)
        grid = np.linspace(0.0, 1.0, 257)
        span = int(math.ceil(9 * self.sd)) + 3
        xs = np.arange(-span, span + 2)
        best = 0.0
        for c in grid:
            ratio = self._target(xs, c) / np.maximum(self._proposal(xs, c), 1e-300)
            best = max(best, float(np.max(ratio)))
        self.m = best * 1.02

    def _target(self, x, c):
        return np.exp(-np.pi * (x - c) ** 2 / self.r ** 2)

    def _proposal(self, x, c):
        k = self.sd * math.sqrt(2)
        z1 = (x + 0.5 - c) / k
        z0 = (x - 0.5 - c) / k
        right = 0.5 * (erfc(z0) - erfc(z1))
        left = 0.5 * (erfc(-z1) - erfc(-z0))
        mid = 0.5 * (erf(z1) - erf(z0))
        return np.where(z0 >= 0, right, np.where(z1 <= 0, left, mid))

    def sample(self, rng, centers):
        c = np.asarray(centers, dtype=np.float64).ravel()
        out = np.empty(c.size, dtype=np.int64)
        todo = np.arange(c.size)
        while todo.size:
            cc = c[todo]
            x = np.rint(rng.normal(cc, self.sd)).astype(np.int64)
            acc = rng.random(todo.size) < self._target(x, cc) / (self.m * np.maximum(self._proposal(x, cc), 1e-300))
            out[todo[acc]] = x[acc]
            todo = todo[~acc]
        return out.reshape(np.shape(centers))


def big_gauss(rng, s, size):
    sd = s / math.sqrt(2 * math.pi)
    e = max(0, int(math.floor(math.log2(sd))) - 50)
    g = rng.normal(0.0, sd / 2.0 ** e, size)
    out = np.empty(size, dtype=object)
    lows = rng.integers(0, 1 << min(e, 62), size=size) if e > 0 else np.zeros(size, dtype=np.int64)
    for i in range(size):
        hi = int(math.floor(g[i]))
        low = int(lows[i])
        if e > 62:
            low = (low << (e - 62)) | int(rng.integers(0, 1 << (e - 62)))
        out[i] = (hi << e) + low
    return out
