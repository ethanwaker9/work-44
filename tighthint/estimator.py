import math
from functools import lru_cache
import numpy as np

CLASSICAL = 0.292
QUANTUM = 0.265
SIEVE_VECTORS = 0.2075
B_MIN = 40
B_MAX = 4000


@lru_cache(maxsize=None)
def log_delta(b):
    b = float(b)
    return math.log((b / (2 * math.pi * math.e)) * (math.pi * b) ** (1.0 / b)) / (2.0 * (b - 1.0))


def _bisect(pred):
    if not pred(B_MAX):
        return B_MAX
    lo, hi = B_MIN, B_MAX
    if pred(lo):
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def primal_success(b, n_sec, m_max, q, sd_err, sd_sec):
    m = np.arange(1, m_max + 1, dtype=np.float64)
    d = n_sec + m + 1
    ld = log_delta(b)
    log_vol = m * math.log(q) + n_sec * math.log(sd_err / sd_sec)
    lhs = math.log(sd_err) + 0.5 * math.log(b)
    rhs = (2 * b - d - 1) * ld + log_vol / d
    ok = (lhs <= rhs) & (d >= b)
    return bool(np.any(ok)) or b >= n_sec + m_max + 1


@lru_cache(maxsize=None)
def primal_usvp(n_sec, m_max, q, sd_err, sd_sec=None):
    if sd_sec is None:
        sd_sec = sd_err
    return _bisect(lambda b: primal_success(b, n_sec, m_max, q, sd_err, sd_sec))


def dual_cost_bits(b, n_sec, m_max, q, sd_err, sd_sec, rate):
    m = np.arange(1, m_max + 1, dtype=np.float64)
    scale = sd_err / sd_sec
    d = n_sec + m
    ld = log_delta(b)
    log_len = d * ld + (n_sec * math.log(q / scale)) / d
    tau = np.exp(log_len) * sd_err / q
    log2_eps = -2 * math.pi * math.pi * tau * tau / math.log(2)
    reps = np.maximum(0.0, -2 * log2_eps - SIEVE_VECTORS * b)
    return float(np.min(rate * b + reps))


@lru_cache(maxsize=None)
def dual_bits(n_sec, m_max, q, sd_err, sd_sec=None, quantum=False):
    if sd_sec is None:
        sd_sec = sd_err
    rate = QUANTUM if quantum else CLASSICAL
    best = float("inf")
    for b in range(B_MIN, B_MAX, 5):
        if rate * b >= best:
            break
        best = min(best, dual_cost_bits(b, n_sec, m_max, q, sd_err, sd_sec, rate))
    return best


def sis_success(b, rows, cols, q, beta):
    lq = math.log(q)
    ld = log_delta(b)
    dstar = math.sqrt(rows * lq / ld)
    cands = {cols, max(rows + 1, min(cols, int(dstar))), max(rows + 1, min(cols, int(dstar) + 1))}
    best = min(d * ld + rows * lq / d for d in cands)
    return best <= math.log(beta)


@lru_cache(maxsize=None)
def sis_l2(rows, cols, q, beta):
    if beta >= q:
        return B_MIN
    return _bisect(lambda b: sis_success(b, rows, cols, q, beta))


def bits(b, quantum=False):
    return (QUANTUM if quantum else CLASSICAL) * b


def mlwe_bits(n, d_sec, m_samples, q, sd, sd_sec=None, quantum=False):
    p = bits(primal_usvp(d_sec * n, m_samples * n, q, sd, sd_sec), quantum)
    d = dual_bits(d_sec * n, m_samples * n, q, sd, sd_sec, quantum)
    return min(p, d)


def mlwe_primal_bits(n, d_sec, m_samples, q, sd, sd_sec=None, quantum=False):
    return bits(primal_usvp(d_sec * n, m_samples * n, q, sd, sd_sec), quantum)


def msis_bits(n, rows, cols, q, beta, quantum=False):
    return bits(sis_l2(rows * n, cols * n, q, beta), quantum)


def _bstar(lam, rate):
    return int(math.floor(lam / rate - 1e-12))


@lru_cache(maxsize=None)
def min_std_mlwe(n_sec, m_max, q, lam=128, rate=CLASSICAL, primal_only=False):
    b = _bstar(lam, rate)
    if b >= n_sec + m_max + 1:
        return float("inf")
    m = np.arange(1, m_max + 1, dtype=np.float64)
    d = n_sec + m + 1
    ok = d >= b
    ld = log_delta(b)
    rhs = (2 * b - d - 1) * ld + m * math.log(q) / d - 0.5 * math.log(b)
    primal = math.exp(float(np.max(np.where(ok, rhs, -np.inf))))
    if primal_only:
        return primal
    bs = np.arange(B_MIN, b + 1, dtype=np.float64)
    lds = np.array([log_delta(int(x)) for x in bs])
    need = (lam - (rate - SIEVE_VECTORS) * bs) * math.log(2) / (4 * math.pi ** 2)
    need = np.maximum(need, 0.0)
    mm = m[None, :]
    dd = n_sec + mm
    log_len = dd * lds[:, None] + (n_sec * math.log(q)) / dd
    val = 0.5 * np.log(need[:, None]) + math.log(q) - log_len
    dual = math.exp(float(np.max(val)))
    return max(primal, dual)


@lru_cache(maxsize=None)
def max_beta_msis(rows, cols, q, lam=128, rate=CLASSICAL):
    b = _bstar(lam, rate)
    lq = math.log(q)
    ld = log_delta(b)
    dstar = math.sqrt(rows * lq / ld)
    cands = {cols, max(rows + 1, min(cols, int(dstar))), max(rows + 1, min(cols, int(dstar) + 1))}
    best = min(d * ld + rows * lq / d for d in cands)
    return min(math.exp(best), float(q)) * (1 - 1e-9)
