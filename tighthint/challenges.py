import hashlib
import math
import numpy as np


def log2_comb(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def challenge_space_log2(n, kappa):
    return log2_comb(n, kappa) + kappa


def binary_space_log2(n, kappa):
    return log2_comb(n, kappa)


def kappa_min(n, lam, loss_bits=1.0):
    for kappa in range(1, n + 1):
        if challenge_space_log2(n, kappa) - loss_bits >= lam:
            return kappa
    return None


def kappa_min_binary(n, lam):
    for kappa in range(1, n // 2 + 1):
        if binary_space_log2(n, kappa) >= lam:
            return kappa
    return None


def spectral_ratio(n):
    target = math.log(n * math.e)
    lo, hi = 1.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - math.log(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def spectral_ratio_delta(n, delta):
    target = math.log(n * math.e / (2.0 * delta))
    lo, hi = 1.0, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - math.log(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spectral_threshold(n, kappa):
    return spectral_ratio(n) * kappa


def hoeffding_bc2(n, kappa, frac=0.5):
    return 2.0 * kappa * math.log(2.0 * n / (1.0 - frac))


def worst_unfiltered_s1(n, kappa):
    return math.sin(kappa * math.pi / (2 * n)) / math.sin(math.pi / (2 * n))


def embed_sq(gamma):
    n = gamma.shape[-1]
    tw = np.exp(1j * np.pi * np.arange(n) / n)
    return np.abs(np.fft.fft(gamma * tw, axis=-1)) ** 2


def spectral_sq(gamma):
    return float(np.max(embed_sq(gamma)))


def expand(data, nbytes):
    return hashlib.shake_256(data).digest(nbytes)


def sparse_from_stream(stream, n, kappa, signed=True):
    pos = []
    used = set()
    sign_bits = int.from_bytes(stream[:16], "little")
    idx = 16
    mask = (1 << max(1, (n - 1).bit_length())) - 1
    while len(pos) < kappa:
        if idx + 2 > len(stream):
            stream = stream + hashlib.shake_256(stream).digest(len(stream))
        v = int.from_bytes(stream[idx: idx + 2], "little") & mask
        idx += 2
        if v >= n or v in used:
            continue
        used.add(v)
        pos.append(v)
    g = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(pos):
        g[p] = (1 if (sign_bits >> (i % 128)) & 1 else -1) if signed else 1
    return g


def hash_to_challenge(data, n, kappa, bc2=None):
    ctr = 0
    while True:
        h = expand(data + ctr.to_bytes(4, "little"), 16 + 8 * n)
        g = sparse_from_stream(h, n, kappa)
        if bc2 is None or spectral_sq(g) <= bc2:
            return g
        ctr += 1


def hash_to_binary(data, n, kappa):
    h = expand(data + b"bin", 16 + 8 * n)
    return sparse_from_stream(h, n, kappa, signed=False)


def hash_to_coset(data, n):
    h = expand(data + b"coset", (n + 7) // 8)
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8), bitorder="little")[:n]
    return bits.astype(np.int64)


def random_challenge(rng, n, kappa):
    g = np.zeros(n, dtype=np.int64)
    pos = rng.choice(n, size=kappa, replace=False)
    g[pos] = rng.choice(np.array([-1, 1]), size=kappa)
    return g


def random_challenges(rng, n, kappa, count):
    g = np.zeros((count, n), dtype=np.int64)
    keys = rng.random((count, n))
    pos = np.argsort(keys, axis=1)[:, :kappa]
    signs = rng.choice(np.array([-1, 1]), size=(count, kappa))
    np.put_along_axis(g, pos, signs, axis=1)
    return g


def spectral_quantile(rng, n, kappa, frac, trials):
    vals = np.max(embed_sq(random_challenges(rng, n, kappa, trials)), axis=1)
    return float(np.quantile(vals, frac))


def spectral_fraction(rng, n, kappa, bc2, trials):
    vals = np.max(embed_sq(random_challenges(rng, n, kappa, trials)), axis=1)
    return float(np.mean(vals <= bc2))
