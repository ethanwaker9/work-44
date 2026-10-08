import math
from fractions import Fraction
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def su_pmf(N, T):
    p = np.ones(N, dtype=np.float64) / N
    out = p.copy()
    for _ in range(T - 1):
        out = np.convolve(out, p)
    return out


def su_cdf_first_piece(N, T, y):
    if y < 0:
        return 0.0
    return math.comb(y + T, T) / N ** T


def su_variance(N, T):
    return T * (N * N - 1) / 12.0


def log_ratio(N, T):
    P = su_pmf(N, T)
    return np.log(P[1:]) - np.log(P[:-1])


def fisher(N, T):
    P = su_pmf(N, T)
    g = log_ratio(N, T)
    return float(np.sum(P[1:] * g * g))


def kappa_discrete(N, T):
    return fisher(N, T) * su_variance(N, T)


def gamma_edge(T, a):
    return math.log(1.0 + (T - 1) / (a + 1.0))


def tail_mass(N, T, a, c):
    c = abs(c)
    return su_cdf_first_piece(N, T, a - 1) + su_cdf_first_piece(N, T, a + c - 1)


def renyi_bound(N, T, c, alpha, a):
    c = abs(c)
    I = fisher(N, T)
    G = gamma_edge(T, a)
    eps = tail_mass(N, T, a, c)
    val = 0.5 * alpha * c * c * I * math.exp(max(alpha, 2.0) * c * G) / (1.0 - eps)
    return val, eps


def renyi_restricted(N, T, c, alpha, a):
    c = abs(c)
    P = su_pmf(N, T)
    L = len(P) - 1
    lo, hi = a + c, L - a
    xs = np.arange(lo, hi + 1)
    lp = np.log(P[xs])
    lq = np.log(P[xs - c])
    PB = float(np.exp(lp).sum())
    QB = float(np.exp(lq).sum())
    t = alpha * lp + (1.0 - alpha) * lq
    m = float(np.max(t))
    lphi = m + math.log(float(np.sum(np.exp(t - m))))
    val = (lphi - alpha * math.log(PB) + (alpha - 1) * math.log(QB)) / (alpha - 1)
    return val, 1 - PB, 1 - QB


def renyi_bound_asym(N, T, c, alpha, a, kappa_t):
    c = abs(c)
    sig2 = su_variance(N, T)
    G = gamma_edge(T, a)
    eps = tail_mass(N, T, a, c)
    return 0.5 * alpha * c * c * (kappa_t / sig2) * math.exp(max(alpha, 2.0) * c * G) / (1.0 - eps), eps


def tail_log2(N, T, y):
    if y < 0:
        return -float("inf")
    return (math.lgamma(y + T + 1) - math.lgamma(T + 1) - math.lgamma(y + 1) - T * math.log(N)) / math.log(2)


def best_a_asym(N, T, c, alpha, kappa_t, log2_eps):
    L = T * (N - 1)
    best = None
    lo, hi = 0, min(N - 2, N - abs(c), (L - abs(c)) // 2)
    for _ in range(200):
        if hi < lo:
            break
        mid = (lo + hi) // 2
        e1 = tail_log2(N, T, mid - 1)
        e2 = tail_log2(N, T, mid + abs(c) - 1)
        le = max(e1, e2) + 1
        if le <= log2_eps:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None
    sig2 = su_variance(N, T)
    G = gamma_edge(T, best)
    val = 0.5 * alpha * c * c * (kappa_t / sig2) * math.exp(max(alpha, 2.0) * abs(c) * G)
    return val, best


def irwin_hall_pdf(x, T):
    s = 0.0
    for k in range(0, int(math.floor(x)) + 1):
        if k > T:
            break
        s += (-1) ** k * math.comb(T, k) * (x - k) ** (T - 1)
    return s / math.factorial(T - 1)


def irwin_hall_dpdf(x, T):
    s = 0.0
    for k in range(0, int(math.floor(x)) + 1):
        if k > T:
            break
        s += (-1) ** k * math.comb(T, k) * (T - 1) * (x - k) ** (T - 2)
    return s / math.factorial(T - 1)


def kappa_continuous(T, pts_per_piece=4000):
    total = 0.0
    for piece in range(T):
        xs = np.linspace(piece, piece + 1, pts_per_piece + 1)
        mids = 0.5 * (xs[1:] + xs[:-1])
        w = xs[1] - xs[0]
        vals = []
        for x in mids:
            p = irwin_hall_pdf(x, T)
            d = irwin_hall_dpdf(x, T)
            vals.append(d * d / p if p > 0 else 0.0)
        total += w * sum(vals)
    return total * T / 12.0


def kappa_continuous_exact(T, dps=30):
    from mpmath import mp, mpf, quad, binomial, factorial, floor
    mp.dps = dps

    def pdf(x):
        s = mpf(0)
        for k in range(0, int(floor(x)) + 1):
            if k > T:
                break
            s += (-1) ** k * binomial(T, k) * (x - k) ** (T - 1)
        return s / factorial(T - 1)

    def dpdf(x):
        s = mpf(0)
        for k in range(0, int(floor(x)) + 1):
            if k > T:
                break
            s += (-1) ** k * binomial(T, k) * (T - 1) * (x - k) ** (T - 2)
        return s / factorial(T - 1)

    f = lambda x: dpdf(x) ** 2 / pdf(x) if pdf(x) > 0 else 0
    I = quad(f, [mpf(i) for i in range(T + 1)])
    return float(I * mpf(T) / 12)


def renyi_layered(N, T, c, alpha, a, b, kappa_t):
    c = abs(c)
    sig2 = su_variance(N, T)
    I = kappa_t / sig2
    beta = max(1.0, alpha - 1.0)
    Ga = gamma_edge(T, a)
    Gb = gamma_edge(T, b)
    eps = tail_mass(N, T, a, c)
    core = I * math.exp((c - 1) * Gb + beta * c * Gb)
    edge = math.exp(math.log(2.0) + tail_log2(N, T, b + c - 1) * math.log(2.0) + 2 * math.log(Ga) + beta * c * Ga)
    return 0.5 * alpha * c * c * (core + edge) / (1.0 - eps), eps


def layered_constant(N, T, c, alpha, kappa_t, grid=400):
    a = int((alpha - 1) * c) + 1
    hi = min(N - c - 2, (T * (N - 1) - c) // 2)
    best = None
    for i in range(grid + 1):
        b = int(round(a * (hi / a) ** (i / grid)))
        v, eps = renyi_layered(N, T, c, alpha, a, max(a, b), kappa_t)
        if best is None or v < best[0]:
            best = (v, max(a, b), eps)
    v, b, eps = best
    sig2 = su_variance(N, T)
    lplain = math.log(0.5 * alpha * c * c * kappa_t / sig2) + max(alpha, 2.0) * c * gamma_edge(T, a)
    scale = T * float(N) * float(N) / (alpha * c * c)
    e1, e2 = tail_log2(N, T, a - 1), tail_log2(N, T, a + c - 1)
    le = max(e1, e2) + math.log2(1.0 + 2.0 ** (min(e1, e2) - max(e1, e2)))
    return {"a": a, "b": b, "eps_log2": le, "layered": v * scale, "plain_log10": (lplain + math.log(scale)) / math.log(10)}
