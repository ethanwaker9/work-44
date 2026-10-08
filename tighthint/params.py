import json
import math
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erfc
from .estimator import mlwe_bits, msis_bits, min_std_mlwe, max_beta_msis, CLASSICAL
from .ring import prime_5mod8
from .challenges import (kappa_min, kappa_min_binary, spectral_threshold, worst_unfiltered_s1)

LAM = 128
CHAL = None
SPEC = None
RATE = CLASSICAL
TAIL = None
PRIMAL_ONLY = False
MULTI_Q = 1
K = 1
TERNARY_SD = math.sqrt(2.0 / 3.0)
METHODS = ["uniform", "gauss", "onebit", "bimodal", "gpg", "flood", "klss", "enp", "ours", "ours_e"]
ABLATION = ["enp", "enp_spec", "ours", "ours_e"]
LABELS = {
    "uniform": "Uniform [Lyu09]",
    "gauss": "Gaussian [Lyu12]",
    "onebit": "OneBit [LNS21]",
    "bimodal": "Bimodal [DDLL13]",
    "gpg": "G+G [DPS23]",
    "flood": "Flooding",
    "klss": "Hint [KLSS23]",
    "enp": "MatrixHint [ENP24]",
    "ours": "Ours",
    "ours_e": "Ours-E",
    "enp_spec": "MatrixHint + spectral set",
    "lossless": "Lossless limit",
}


def tail_t(lam=None):
    if lam is None:
        lam = TAIL if TAIL is not None else LAM
    return lam * math.log(2) + math.log(2)


def eta(n, D, lam=None):
    lam = LAM if lam is None else lam
    eps = 2.0 ** (-lam - 4) / D
    return math.sqrt(math.log(2 * n * (1 + 1 / eps)) / math.pi)


def hkz_sq(tr, op, t):
    return tr + 2 * math.sqrt(tr * op * t) + 2 * op * t


def gauss_norm(s, m, t):
    v = s * s / (2 * math.pi)
    return math.sqrt(hkz_sq(m * v, v, t))


def ternary_norm(m, t):
    return math.sqrt(2.0 * m / 3.0 + math.sqrt(m * t / 2.0))


def chi_bound(v, dof2, target_logp):
    best = float("inf")
    for i in range(1, 400):
        th = (i / 400.0) / (2 * v)
        lp = math.log(1 - 2 * th * v) * (-dof2)
        tval = (lp - target_logp) / th
        best = min(best, tval)
    return best


def key_s1_bound(n, D):
    v = n / 3.0
    return math.sqrt(chi_bound(v, D, math.log(0.5 / (n / 2))) + 1.0)


def rice_len(std, k):
    if std <= 0:
        return 2.0 + k
    if std < 64:
        span = int(math.ceil(14 * std)) + 2
        xs = np.arange(-span, span + 1)
        w = np.exp(-0.5 * (xs / std) ** 2)
        w /= w.sum()
        return float(np.sum(w * (2 + k + (np.abs(xs) >> k))))
    L = float(2 ** k)
    j = np.arange(1, int(20 * std / L) + 3)
    return 2.0 + k + float(np.sum(erfc(j * L / (std * math.sqrt(2)))))


def rice_bits(std):
    k0 = max(0, int(math.floor(math.log2(max(std, 1.0)))))
    best = min((rice_len(std, k), k) for k in range(max(0, k0 - 2), k0 + 3))
    return best


@lru_cache(maxsize=None)
def q_of(logq):
    return prime_5mod8(logq)


@lru_cache(maxsize=None)
def min_std(n, nu, samples, logq):
    return min_std_mlwe(nu * n, samples * n, q_of(logq), LAM, RATE, PRIMAL_ONLY)


@lru_cache(maxsize=None)
def max_beta(n, mu, D, logq):
    return max_beta_msis(mu * n, D * n, q_of(logq), LAM, RATE)


def set_lambda(lam):
    global LAM
    LAM = lam
    min_std.cache_clear()
    max_beta.cache_clear()


def set_variant(rate=CLASSICAL, tail=None, primal_only=False, q_proofs=1, chal=None, spec=None):
    global RATE, TAIL, PRIMAL_ONLY, MULTI_Q, CHAL, SPEC
    RATE, TAIL, PRIMAL_ONLY, MULTI_Q, CHAL, SPEC = rate, tail, primal_only, q_proofs, chal, spec
    min_std.cache_clear()
    max_beta.cache_clear()


def t_many(Q, lam, n, filtered=False):
    target = (lam * math.log(2) + math.log(n / 2) + (Q * math.log(2) if filtered else 0.0)) / Q
    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - 1 - math.log(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi * Q


def chal_bits():
    return LAM if CHAL is None else CHAL


def challenge_info(method, n):
    if SPEC is not None and method in ("ours", "ours_e"):
        return {"kappa": SPEC[0], "weight": SPEC[0], "s1": math.sqrt(SPEC[1]), "kind": "spectral"}
    if method in ("ours", "ours_e", "gauss", "onebit", "enp_spec", "lossless"):
        kappa = kappa_min(n, chal_bits(), 1.0)
        if kappa is None:
            return None
        bc2 = spectral_threshold(n, kappa)
        return {"kappa": kappa, "weight": kappa, "s1": math.sqrt(bc2), "kind": "spectral"}
    if method in ("uniform", "klss", "enp", "flood", "ours_u"):
        kappa = kappa_min(n, chal_bits(), 0.0)
        if kappa is None:
            return None
        return {"kappa": kappa, "weight": kappa, "s1": worst_unfiltered_s1(n, kappa), "kind": "sparse"}
    if method == "bimodal":
        kappa = kappa_min_binary(n, chal_bits())
        if kappa is None:
            return None
        return {"kappa": kappa, "weight": kappa, "s1": worst_unfiltered_s1(n, kappa), "kind": "binary"}
    if method == "gpg":
        if n < chal_bits():
            return None
        return {"kappa": n, "weight": None, "s1": None, "kind": "coset"}
    raise ValueError(method)


def hint_alloc(P, bc2, kappa, m, t, extra=None):
    def beta_of(la):
        a = math.exp(la)
        if 1.0 / a >= 1.0 / P:
            return float("inf"), a, float("inf")
        b = bc2 / (1.0 / P - 1.0 / a)
        tr = m * (b + kappa * a) / (2 * math.pi)
        op = (b + bc2 * a) / (2 * math.pi)
        return math.sqrt(hkz_sq(tr, op, t)), a, b
    res = minimize_scalar(lambda la: beta_of(la)[0], bounds=(math.log(P) + 1e-9, math.log(P) + 12), method="bounded")
    beta, a, b = beta_of(res.x)
    return beta, math.sqrt(a), math.sqrt(b)


def ellip_alloc(P0, W2, bc2, kappa, m, t):
    def beta_of(la):
        a = math.exp(la)
        if 1.0 / a >= 1.0 / P0 or 1.0 / a >= 1.0 / W2:
            return float("inf"), a, float("inf")
        b = max(bc2 / (1.0 / P0 - 1.0 / a), kappa / (1.0 / W2 - 1.0 / a))
        tr = m * (b + kappa * a) / (2 * math.pi)
        op = (b + bc2 * a) / (2 * math.pi)
        return math.sqrt(hkz_sq(tr, op, t)), a, b
    lo = math.log(max(P0, W2)) + 1e-9
    res = minimize_scalar(lambda la: beta_of(la)[0], bounds=(lo, lo + 12), method="bounded")
    beta, a, b = beta_of(res.x)
    return beta, math.sqrt(a), math.sqrt(b)


def evaluate(method, n, logq, mu, nu, width=None):
    ch = challenge_info(method, n)
    if ch is None:
        return None
    t = tail_t()
    D = mu + nu + K
    lifted = method in ("bimodal", "gpg")
    m = (D + 1) * n if lifted else D * n
    q = q_of(logq)
    e = eta(n, D)
    out = {"method": method, "n": n, "logq": logq, "q": q, "mu": mu, "nu": nu, "D": D, "kappa": ch["kappa"], "eta": e}
    reps = 1.0
    uniform_z = False
    if method in ("uniform", "gauss", "onebit", "bimodal", "gpg", "flood"):
        hid_sd = TERNARY_SD
        Br = ternary_norm(D * n, t)
        BS = key_s1_bound(n, D + (1 if lifted else 0))
        if method in ("gauss", "onebit", "flood"):
            T = min(ch["s1"] * Br, BS * math.sqrt(ch["weight"]))
        elif method == "bimodal":
            T = BS * math.sqrt(ch["weight"])
        out["T"] = T if method != "uniform" and method != "gpg" else None
        if method == "uniform":
            kap = ch["kappa"]
            g1 = kap
            while ((2 * (g1 - kap) + 1) / (2 * g1 + 1)) ** m < math.exp(-1):
                g1 = int(g1 * 1.01) + 1
            lo_, hi_ = kap, g1
            while hi_ - lo_ > 1:
                mid = (lo_ + hi_) // 2
                if ((2 * (mid - kap) + 1) / (2 * mid + 1)) ** m >= math.exp(-1):
                    hi_ = mid
                else:
                    lo_ = mid
            g1 = hi_
            reps = 1.0 / (((2 * (g1 - kap) + 1) / (2 * g1 + 1)) ** m)
            bound = g1 - kap
            beta = math.sqrt(m) * bound
            out.update({"gamma1": g1, "binf": bound})
            zbits = math.ceil(math.log2(2 * bound + 1))
            uniform_z = True
            zstd = None
        elif method == "gauss":
            gam = 13.0
            s_std = gam * T
            reps = math.exp(14 / gam + 1 / (2 * gam * gam))
        elif method in ("onebit", "bimodal"):
            gam = 0.675
            s_std = gam * T
            M = math.exp(1 / (2 * gam * gam))
            reps = 2 * M if method == "onebit" else M
        elif method == "gpg":
            eps = 2.0 ** (-LAM - 1)
            ss = math.sqrt(2 * math.log(n - 1 + 2 * n / eps) / math.pi)
            s_std = 8 * BS * ss / math.sqrt(2 * math.pi)
            out["gpg_s"] = ss
            out["BS"] = BS
        elif method == "flood":
            s_param = math.sqrt(math.pi / 2) * T * 2.0 ** LAM
            s_std = s_param / math.sqrt(2 * math.pi)
        if not uniform_z:
            s_param = s_std * math.sqrt(2 * math.pi)
            beta = gauss_norm(s_param, m, t) + (T if method == "flood" else 0.0)
            zstd = s_std
            out["s"] = s_param
    else:
        kappa = ch["kappa"]
        s1max = ch["s1"]
        if method == "klss":
            B = MULTI_Q * kappa * kappa
        elif method == "ours_u":
            B = t_many(MULTI_Q, LAM, n) * kappa
        elif method in ("ours", "lossless"):
            B = min(MULTI_Q * s1max * s1max, t_many(MULTI_Q, LAM, n, True) * kappa)
        else:
            B = MULTI_Q * s1max * s1max
        need = min_std(n, nu, mu + K, logq) if width is None else width
        if method in ("klss", "enp", "enp_spec"):
            sig0 = max(need * math.sqrt(2 * math.pi), math.sqrt(2) * e)
            P = 2 * sig0 * sig0
            beta, s1, s2 = hint_alloc_klss(P, B, s1max, kappa, m, t)
            hid_sd = sig0 / math.sqrt(2 * math.pi)
        elif method in ("ours", "ours_u", "lossless"):
            sig0 = max(need * math.sqrt(2 * math.pi), math.sqrt(2) * e)
            P = max(sig0 ** 2, 4 * e * e) if method == "lossless" else sig0 ** 4 / (sig0 ** 2 - e * e)
            beta, s1, s2 = hint_alloc_klss(P, B, s1max, kappa, m, t)
            hid_sd = sig0 / math.sqrt(2 * math.pi)
        else:
            w = need * math.sqrt(2 * math.pi)
            W2 = w * w + 2 * e * e
            P0 = 4 * e * e
            beta, s1, s2 = ellip_alloc(P0, W2, B, kappa, m, t)
            weff2 = 1.0 / (1.0 / s1 ** 2 + kappa / s2 ** 2) - 2 * e * e
            hid_sd = math.sqrt(weff2) / math.sqrt(2 * math.pi)
        out.update({"sigma1": s1, "sigma2": s2, "P": P if method != "ours_e" else None})
        zstd = math.sqrt((s2 * s2 + kappa * s1 * s1) / (2 * math.pi))
    out["hid_sd"] = hid_sd
    out["zstd"] = zstd
    out["beta"] = beta
    out["reps"] = reps
    if lifted:
        bmsis = 8 * math.sqrt(n) * beta * beta
    else:
        bmsis = 8 * ch["s1"] * beta
    out["beta_msis"] = bmsis
    if uniform_z:
        bits_coef = zbits
        out["rice_k"] = None
    else:
        bits_coef, rk = rice_bits(zstd)
        out["rice_k"] = rk
    out["bits_coef"] = bits_coef
    out["proof_bytes"] = 32 + math.ceil(m * bits_coef / 8)
    out["commit_bytes"] = math.ceil((mu + K) * n * logq / 8)
    out["q_proofs"] = MULTI_Q
    out["total_bytes"] = out["proof_bytes"] + math.ceil(out["commit_bytes"] / MULTI_Q)
    return out


def hint_alloc_klss(P, B, s1max, kappa, m, t):
    def beta_of(la):
        a = math.exp(la)
        if 1.0 / a >= 1.0 / P:
            return float("inf"), a, float("inf")
        b = B / (1.0 / P - 1.0 / a)
        tr = m * (b + kappa * a) / (2 * math.pi)
        op = (b + s1max * s1max * a) / (2 * math.pi)
        return math.sqrt(hkz_sq(tr, op, t)), a, b
    res = minimize_scalar(lambda la: beta_of(la)[0], bounds=(math.log(P) + 1e-9, math.log(P) + 12), method="bounded")
    beta, a, b = beta_of(res.x)
    return beta, math.sqrt(a), math.sqrt(b)


def feasible(o):
    if o is None:
        return False
    n, logq, mu, nu = o["n"], o["logq"], o["mu"], o["nu"]
    if o["hid_sd"] < min_std(n, nu, mu + K, logq):
        return False
    if o["beta_msis"] > max_beta(n, mu, o["D"], logq):
        return False
    if o["method"] in ("bimodal", "gpg") and 2 * o["beta"] >= math.sqrt(o["q"]):
        return False
    return True


def attach_security(o):
    q = o["q"]
    o["hiding_bits"] = mlwe_bits(o["n"], o["nu"], o["mu"] + K, q, round(o["hid_sd"], 6))
    o["binding_bits"] = msis_bits(o["n"], o["mu"], o["D"], q, o["beta_msis"])
    o["hiding_bits_q"] = mlwe_bits(o["n"], o["nu"], o["mu"] + K, q, round(o["hid_sd"], 6), quantum=True)
    o["binding_bits_q"] = msis_bits(o["n"], o["mu"], o["D"], q, o["beta_msis"], quantum=True)
    return o


def min_nu_ternary(n, logq, mu, nu_max):
    if TERNARY_SD < min_std(n, nu_max, mu + K, logq):
        return None
    lo, hi = 0, nu_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if TERNARY_SD >= min_std(n, mid, mu + K, logq):
            hi = mid
        else:
            lo = mid
    return hi


def search(method, ns=(128, 256), logqs=None, mus=None, nus=None):
    ternary = method in ("uniform", "gauss", "onebit", "bimodal", "gpg", "flood")
    if logqs is None:
        logqs = range(120, 260) if method == "flood" else range(14, 60)
    if mus is None:
        mus = range(1, 40) if method == "flood" else range(1, 15)
    if nus is None:
        nus = range(1, 20)
    nu_max = 400 if method == "flood" else max(nus)
    best = None
    for n in ns:
        if challenge_info(method, n) is None:
            continue
        for logq in logqs:
            for mu in mus:
                if best is not None and math.ceil((mu + K) * n * logq / 8 / MULTI_Q) >= best["total_bytes"]:
                    continue
                if ternary:
                    nu0 = min_nu_ternary(n, logq, mu, nu_max)
                    cand = [] if nu0 is None else [nu0]
                else:
                    cand = [nu for nu in nus if min_std(n, nu, mu + K, logq) < float("inf")]
                for nu in cand:
                    o = evaluate(method, n, logq, mu, nu)
                    if o is None or not feasible(o):
                        continue
                    if best is None or o["total_bytes"] < best["total_bytes"]:
                        best = o
    if best is not None:
        attach_security(best)
    return best


def search_all(path=None, methods=METHODS):
    res = {}
    for mth in methods:
        res[mth] = search(mth)
    if path is not None:
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=float)
    return res
