import json
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint import su
from tighthint.challenges import (random_challenges, embed_sq, spectral_threshold, hoeffding_bc2,
                                  worst_unfiltered_s1, kappa_min, challenge_space_log2)
from tighthint.hint import tight_sigma0_sq
from tighthint.params import eta

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def factor_curve():
    xs = np.logspace(math.log10(4.0), 4, 60)
    ours = [tight_sigma0_sq(x, 1.0) / x for x in xs]
    return {"x": xs.tolist(), "ours": ours, "klss": [0.5] * len(xs), "lower": [1 - 2 / x for x in xs]}


def challenge_stats(n=128, trials=200000, seed=7):
    rng = np.random.default_rng(seed)
    kappa = kappa_min(n, 128, 1.0)
    vals = []
    for _ in range(trials // 20000):
        g = random_challenges(rng, n, kappa, 20000)
        vals.append(np.max(embed_sq(g), axis=1))
    v = np.concatenate(vals)
    thr = spectral_threshold(n, kappa)
    return {
        "n": n, "kappa": kappa, "log2C": challenge_space_log2(n, kappa),
        "rigorous": thr, "hoeffding": hoeffding_bc2(n, kappa), "worst_unfiltered": worst_unfiltered_s1(n, kappa) ** 2,
        "l1": kappa * kappa, "fraction": float(np.mean(v <= thr)),
        "quantiles": {str(f): float(np.quantile(v, f)) for f in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)},
        "hist": np.histogram(v, bins=80, range=(0, 500))[0].tolist(), "hist_range": [0, 500], "max": float(v.max()),
    }


def logdet_stats(params, n=128, trials=20000, seed=11):
    rng = np.random.default_rng(seed)
    s1, s2 = params["sigma1"], params["sigma2"]
    kappa = params["kappa"]
    D = params["D"]
    e2 = eta(n, D) ** 2
    thr = spectral_threshold(n, kappa)
    g = random_challenges(rng, n, kappa, trials)
    x = embed_sq(g)
    keep = np.max(x, axis=1) <= thr
    x = x[keep]
    p = 1.0 / (1.0 / s1 ** 2 + x / s2 ** 2)
    det = np.exp(np.mean(np.log(p - 2 * e2), axis=1))
    worst = np.min(p, axis=1) - 2 * e2
    jensen = 1.0 / (1.0 / s1 ** 2 + kappa / s2 ** 2) - 2 * e2
    spherical = tight_sigma0_sq(1.0 / (1.0 / s1 ** 2 + thr / s2 ** 2), math.sqrt(e2))
    h3 = 1.0 / (1.0 / s1 ** 2 + kappa / s2 ** 2)
    return {
        "sigma1": s1, "sigma2": s2, "eta2": e2, "jensen": jensen, "spherical": spherical, "heuristic3": h3,
        "det_min": float(det.min()), "det_mean": float(det.mean()), "worst_mean": float(worst.mean()),
        "det_hist": np.histogram(det, bins=60)[0].tolist(), "det_edges": np.histogram(det, bins=60)[1].tolist(),
        "worst_hist": np.histogram(worst, bins=60)[0].tolist(), "worst_edges": np.histogram(worst, bins=60)[1].tolist(),
    }


def kappa_table():
    rows = []
    for T in range(2, 17):
        row = {"T": T}
        row["cont"] = su.kappa_continuous_exact(T) if T >= 3 else float("inf")
        for u in (6, 10, 14):
            row["N%d" % u] = su.kappa_discrete(2 ** u, T)
        rows.append(row)
    return rows


def best_a(N, T, c, alpha, eps_max):
    L = T * (N - 1)
    best = None
    a = 0
    step = max(1, N // 4096)
    while a <= N - 2 and a + abs(c) <= N and 2 * a + abs(c) <= L:
        val, eps = su.renyi_bound(N, T, c, alpha, a)
        if eps <= eps_max and (best is None or val < best[0]):
            best = (val, eps, a)
        a += step
    return best


def su_compare():
    rows = []
    for T in (4, 6, 8, 16):
        for u in (10, 14):
            N = 2 ** u
            sig2 = su.su_variance(N, T)
            for c in (1, 4, 16):
                for alpha in (2.0, 8.0, 32.0):
                    if alpha * c * T * 8 > N:
                        continue
                    b = best_a(N, T, c, alpha, 2.0 ** -40)
                    if b is None:
                        continue
                    ours, eps, a = b
                    exact = su.renyi_restricted(N, T, c, alpha, a)[0]
                    rows.append({
                        "T": T, "u": u, "c": c, "alpha": alpha, "a": a, "eps": eps,
                        "ours": ours, "exact": exact, "gauss": alpha * c * c / (2 * sig2),
                        "lemma42": alpha * (T * c / N) ** 2 / 2,
                        "conj": 6 * alpha * c * c * (1 + 2 / (alpha - 1)) / (T * N * N),
                        "leading": alpha * c * c * su.kappa_discrete(N, T) / (2 * sig2),
                    })
    return rows


def su_curve(N=2 ** 12, c=2, alpha=8.0):
    out = []
    for T in range(3, 33):
        sig2 = su.su_variance(N, T)
        b = best_a(N, T, c, alpha, 2.0 ** -40)
        if b is None:
            continue
        ours, eps, a = b
        exact = su.renyi_restricted(N, T, c, alpha, a)[0]
        g = alpha * c * c / (2 * sig2)
        out.append({"T": T, "ours": ours / g, "exact": exact / g, "lemma42": (alpha * (T * c / N) ** 2 / 2) / g,
                    "conj": (6 * alpha * c * c * (1 + 2 / (alpha - 1)) / (T * N * N)) / g})
    return out


def hint_rates(Nsec=256, u=16, qs=None, trials=40, seed=3):
    rng = np.random.default_rng(seed)
    if qs is None:
        qs = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    var = ((2 * u + 1) ** 2 - 1) / 12.0
    res = {"q": qs, "uniform": [], "gauss": [], "su3": [], "su4": [], "var": var, "u": u, "N": Nsec}
    for Q in qs:
        acc = {"uniform": 0.0, "gauss": 0.0, "su3": 0.0, "su4": 0.0}
        for _ in range(trials):
            x = np.rint(rng.normal(0, 40, Nsec)).astype(np.int64)
            yu = rng.integers(-u, u + 1, size=(Q, Nsec))
            hu = x + yu
            est = np.max(hu, axis=0) - u
            acc["uniform"] += np.mean(est == x)
            yg = np.rint(rng.normal(0, math.sqrt(var), (Q, Nsec))).astype(np.int64)
            est = np.rint(np.mean(x + yg, axis=0)).astype(np.int64)
            acc["gauss"] += np.mean(est == x)
            for T, key in ((3, "su3"), (4, "su4")):
                w = int(round(math.sqrt(12 * var / T + 1)))
                ys = sum(rng.integers(0, w, size=(Q, Nsec)) for _ in range(T)) - T * (w - 1) // 2
                est = np.rint(np.mean(x + ys, axis=0) - (T * (w - 1) / 2 - T * (w - 1) // 2)).astype(np.int64)
                acc[key] += np.mean(est == x)
        for k in acc:
            res[k].append(acc[k] / trials)
    return res


def split_tv(Ps=(64.0, 144.0, 400.0), mu=0.3, points=41):
    from mpmath import mp, mpf, exp, pi, fsum
    mp.dps = 60

    def dg(s2, c, lo, hi):
        xs = list(range(lo, hi + 1))
        w = [exp(-pi * (mpf(x) - c) ** 2 / s2) for x in xs]
        Z = fsum(w)
        return {x: wi / Z for x, wi in zip(xs, w)}

    def tv(P, s02):
        R = int(12 * math.sqrt(P)) + 5
        a = dg(mpf(s02), mpf(0), -R, R)
        b = dg(mpf(P) - mpf(s02), mpf(mu), -R, R)
        conv = {}
        for x, px in a.items():
            for y, py in b.items():
                conv[x + y] = conv.get(x + y, 0) + px * py
        t = dg(mpf(P), mpf(mu), -2 * R, 2 * R)
        keys = set(conv) | set(t)
        return float(fsum(abs(conv.get(k, 0) - t.get(k, 0)) for k in keys) / 2)

    eps = 2.0 ** -64
    eta2 = math.log(2 * (1 + 1 / eps)) / math.pi
    out = {"eta2": eta2, "curves": []}
    for P in Ps:
        fr = np.linspace(0.02, 0.98, points)
        vals = [tv(P, f * P) for f in fr]
        out["curves"].append({"P": P, "frac": fr.tolist(), "tv": vals,
                              "theta": tight_sigma0_sq(P, math.sqrt(eta2)) / P,
                              "tv_theta": tv(P, tight_sigma0_sq(P, math.sqrt(eta2)))})
    return out


def su_table():
    rows = []
    for T in (4, 6, 8, 16):
        k = su.kappa_continuous_exact(T)
        for c, alpha in ((1, 2.0), (1, 16.0), (8, 2.0), (8, 16.0)):
            N = 2 ** 16
            b = best_a(N, T, c, alpha, 2.0 ** -40)
            if b is None:
                continue
            ours, eps, a = b
            exact = su.renyi_restricted(N, T, c, alpha, a)[0]
            sig2 = su.su_variance(N, T)
            g = alpha * c * c / (2 * sig2)
            rows.append({"T": T, "c": c, "alpha": alpha, "a": a, "log2eps": math.log2(eps) if eps > 0 else -999,
                         "exact": exact / g, "ours": ours / g, "lemma42": (alpha * (T * c / N) ** 2 / 2) / g,
                         "conj": (6 * alpha * c * c * (1 + 2 / (alpha - 1)) / (T * N * N)) / g, "kappa": k})
    return rows


def t_many(Q, lam=128, n=128):
    target = (lam * math.log(2) + math.log(n / 2)) / Q
    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - 1 - math.log(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi * Q


def many_hints(n=128, kappa=31, qs=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), tuples=200, seed=5):
    rng = np.random.default_rng(seed)
    out = {"q": list(qs), "bound": [], "median": [], "max": [], "n": n, "kappa": kappa}
    for Q in qs:
        vals = []
        for _ in range(tuples):
            g = random_challenges(rng, n, kappa, Q)
            vals.append(float(np.max(np.sum(embed_sq(g), axis=0))) / (Q * kappa))
        out["bound"].append(t_many(Q, 128, n) / Q)
        out["median"].append(float(np.median(vals)))
        out["max"].append(float(np.max(vals)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    params = json.load(open(os.path.join(OUT, "params.json")))
    res = {
        "factor": factor_curve(),
        "challenges": challenge_stats(),
        "logdet": logdet_stats(params["ours"]),
        "kappa": kappa_table(),
        "su": su_compare(),
        "su_curve": su_curve(),
        "rates": hint_rates(),
        "split": split_tv(),
        "su_table": su_table(),
        "many": many_hints(),
    }
    with open(os.path.join(OUT, "theory.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v if k in ("challenges",) else "ok") for k, v in res.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
