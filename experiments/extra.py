import json
import math
import os
import sys
import numpy as np
from scipy.special import erf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint import params as P
from tighthint import su
from tighthint.estimator import mlwe_bits, CLASSICAL, QUANTUM
from tighthint.hint import tight_sigma0_sq
from tighthint.challenges import (kappa_min, spectral_ratio, spectral_threshold, hoeffding_bc2, random_challenges,
                                  embed_sq, challenge_space_log2, worst_unfiltered_s1, log2_comb)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def sandwich_curves():
    xs = np.logspace(math.log10(4.0), 3, 80)
    lower = [tight_sigma0_sq(x, 1.0) / x for x in xs]
    upper = [x / (x - 1.0) for x in xs]
    spherical_upper = [(x + 4.0 / 3.0) / x for x in xs]
    klss = [0.5] * len(xs)
    return {"x": xs.tolist(), "lower": lower, "upper": upper, "spherical_upper": spherical_upper, "klss": klss}


def bits_recovered():
    settings = [
        ("Module, $n=256$, $d=m=2$", 256, 2, 2, 8380417),
        ("Module, $n=256$, $d=m=3$", 256, 3, 3, 8380417),
        ("Ring, $n=1024$, $d=m=1$", 1024, 1, 1, 2 ** 27 - 39),
        ("Ring, $n=2048$, $d=m=1$", 2048, 1, 1, 2 ** 54 - 33),
    ]
    rows = []
    for name, n, d, m, q in settings:
        eps = 2.0 ** -132 / (d + m)
        e2 = math.log(2 * n * (1 + 1 / eps)) / math.pi
        for ratio in (4, 8, 16, 64, 256):
            Pp = ratio * e2
            widths = {"klss": Pp / 2, "ours": tight_sigma0_sq(Pp, math.sqrt(e2)), "upper": Pp + e2 * Pp / (Pp - e2)}
            bits = {k: mlwe_bits(n, d, m, q, round(math.sqrt(v / (2 * math.pi)), 6)) for k, v in widths.items()}
            rows.append({"name": name, "n": n, "d": d, "m": m, "logq": math.log2(q), "ratio": ratio, "eta2": e2,
                         "widths": widths, "bits": bits})
            print(name, ratio, {k: round(v, 1) for k, v in bits.items()}, flush=True)
    return rows


def challenge_families(trials=40000, seed=17):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (128, 256, 512, 1024):
        kappa = kappa_min(n, 128, 1.0)
        thr = spectral_threshold(n, kappa)
        vals = []
        for _ in range(trials // 10000):
            g = random_challenges(rng, n, kappa, 10000)
            vals.append(np.max(embed_sq(g), axis=1))
        v = np.concatenate(vals)
        rows.append({"n": n, "kappa": kappa, "log2C": challenge_space_log2(n, kappa), "x_n": spectral_ratio(n),
                     "rigorous": thr, "hoeffding": hoeffding_bc2(n, kappa), "worst": worst_unfiltered_s1(n, kappa) ** 2,
                     "l1": kappa * kappa, "fraction": float(np.mean(v <= thr)), "median": float(np.median(v)),
                     "q99": float(np.quantile(v, 0.99)), "max": float(v.max())})
        print(rows[-1], flush=True)
    return rows


def alphabet_study(n=128, lam=129):
    best = {}
    for k2 in range(0, 40):
        for k1 in range(0, n - k2 + 1):
            k0 = n - k1 - k2
            if k0 < 0:
                break
            lc = (math.lgamma(n + 1) - math.lgamma(k0 + 1) - math.lgamma(k1 + 1) - math.lgamma(k2 + 1)) / math.log(2) + k1 + k2
            if lc >= lam:
                norm = k1 + 4 * k2
                key = "ternary" if k2 == 0 else "with two"
                if key not in best or norm < best[key]["norm"]:
                    best[key] = {"k1": k1, "k2": k2, "norm": norm, "log2C": lc}
                break
    return best


def cbd_kappa(etas=(1, 2, 3, 4, 6, 8, 12, 16, 32)):
    out = []
    for eta in etas:
        k = np.arange(0, 2 * eta + 1)
        logp = np.array([math.lgamma(2 * eta + 1) - math.lgamma(int(i) + 1) - math.lgamma(2 * eta - int(i) + 1) for i in k]) - 2 * eta * math.log(2)
        p = np.exp(logp)
        g = logp[1:] - logp[:-1]
        I = float(np.sum(p[1:] * g * g))
        var = eta / 2.0
        out.append({"eta": eta, "kappa": var * I})
    return out


def kappa_large_T(Ts=(3, 4, 6, 8, 12, 16, 24, 32, 48)):
    out = []
    for T in Ts:
        k = su.kappa_continuous_exact(T)
        out.append({"T": T, "kappa": k, "scaled": (k - 1) * T * T, "edgeworth": 1 + 6.0 / (25 * T * T)})
        print(out[-1], flush=True)
    return out


def lower_bound_rows():
    rows = []
    for N, T, c, alpha, a in ((2 ** 12, 6, 1, 2.0, 700), (2 ** 12, 8, 1, 8.0, 1500), (2 ** 12, 8, 3, 4.0, 1500),
                              (2 ** 14, 8, 2, 16.0, 6000), (2 ** 14, 16, 4, 8.0, 30000)):
        Pm = su.su_pmf(N, T)
        L = len(Pm) - 1
        lg = np.log(Pm)
        g = lg[1:] - lg[:-1]
        G = su.gamma_edge(T, a)
        eps = su.tail_mass(N, T, a, c)
        xs = np.arange(a + c, L - a + 1)
        ell = lg[xs] - lg[xs - c]
        pB = Pm[xs].sum()
        E2 = float(np.sum(Pm[xs] * ell * ell) / pB)
        upper, _ = su.renyi_bound(N, T, c, alpha, a)
        exact = su.renyi_restricted(N, T, c, alpha, a)[0]
        I_tot = su.fisher(N, T)
        ys = np.arange(1, a + c)
        I_out = 2.0 * float(np.sum(Pm[ys] * g[ys - 1] ** 2))
        I_in = I_tot - I_out
        Elow = c * c * math.exp(-(c - 1) * G) * I_in
        v = (alpha - 1) * alpha / 2 * Elow * math.exp(-max(1.0, alpha - 1) * c * G)
        lower = alpha / 2 * Elow * math.exp(-max(1.0, alpha - 1) * c * G) / (1 + v)
        g0 = alpha * c * c / (2 * su.su_variance(N, T))
        rows.append({"N": N, "T": T, "c": c, "alpha": alpha, "a": a, "eps": eps, "lower": lower / g0,
                     "exact": exact / g0, "upper": upper / g0, "kappa": su.kappa_discrete(N, T), "E2": E2})
        print(rows[-1], flush=True)
    return rows


def map_rates(u=16, qs=None):
    K = 2 * u + 1
    if qs is None:
        qs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    var = (K * K - 1) / 12.0
    out = {"q": qs, "uniform": [], "gauss": [], "uniform_approx": [], "u": u}
    for Q in qs:
        tot = 0.0
        for r in range(0, K):
            if r == 0:
                pr = K * (1.0 / K) ** Q
            else:
                pr = (K - r) * (((r + 1) / K) ** Q - 2 * (r / K) ** Q + ((r - 1) / K) ** Q)
            tot += pr / (K - r)
        out["uniform"].append(tot)
        out["uniform_approx"].append(min(1.0, Q / K))
        out["gauss"].append(float(erf(0.5 * math.sqrt(Q) / (math.sqrt(var) * math.sqrt(2)))))
    return out


def multi_proof(qs=(1, 2, 4, 8, 16, 32, 64)):
    res = {}
    for Q in qs:
        P.set_variant(q_proofs=Q)
        res[Q] = {}
        for m in ("gauss", "uniform", "klss", "enp", "ours", "ours_u"):
            b = P.search(m, ns=(128,))
            res[Q][m] = b
            print(Q, m, None if b is None else (b["total_bytes"], b["proof_bytes"], b["commit_bytes"], b["logq"], b["mu"], b["nu"]), flush=True)
    P.set_variant()
    return res


def sensitivity():
    variants = {"base": dict(), "quantum": dict(rate=QUANTUM), "tail40": dict(tail=40), "primal": dict(primal_only=True)}
    res = {}
    for name, kw in variants.items():
        P.set_variant(**kw)
        res[name] = {}
        for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
            b = P.search(m)
            res[name][m] = None if b is None else {k: b[k] for k in ("total_bytes", "proof_bytes", "commit_bytes", "n", "logq", "mu", "nu")}
            print(name, m, res[name][m], flush=True)
    P.set_variant()
    return res


RACCOON = ((1, 8, 6, 41), (2, 4, 6, 41), (4, 2, 6, 41), (8, 4, 5, 40), (16, 2, 5, 40), (32, 4, 4, 39))


def raccoon_rows(omega=19, alphas=(2, 8, 32, 128)):
    kap = {}
    out = []
    for d, rep, ut, uw in RACCOON:
        T = d * (rep - 1) + 1
        if T not in kap:
            kap[T] = su.kappa_continuous_exact(T, 40 + 2 * T)
        N = 2 ** uw
        c = omega * T * 2 ** (ut - 1)
        row = {"d": d, "rep": rep, "T": T, "uw": uw, "ut": ut, "c": c, "kappa": kap[T], "alphas": {}}
        for al in alphas:
            r = su.layered_constant(N, T, c, al, kap[T])
            r["conj_eps_log2"] = (T * math.log2(al * c + T) - math.lgamma(T + 1) / math.log(2)) - T * uw
            row["alphas"][str(al)] = r
        out.append(row)
        print(row, flush=True)
    return out


def su_noise(T, N):
    p = su.su_pmf(N, T)
    off = (len(p) - 1) / 2.0
    return p, off


def dg_noise(var, width):
    xs = np.arange(-width, width + 1)
    w = np.exp(-xs * xs / (2 * var))
    return w / w.sum(), float(width)


def leak_su(trials=400, seed=31, qmax=4096):
    rng = np.random.default_rng(seed)
    qs = sorted(set(int(round(2 ** (k / 4))) for k in range(0, 4 * int(math.log2(qmax)) + 1)))
    cases = [("SU", 1, 49), ("SU", 2, 35), ("SU", 3, 28), ("SU", 4, 25), ("SU", 8, 17), ("DG", 0, 200)]
    out = []
    for kind, T, N in cases:
        if kind == "SU":
            p, off = su_noise(T, N)
            var = su.su_variance(N, T)
            fisher = su.fisher(N, T)
        else:
            var = float(N)
            p, off = dg_noise(var, int(12 * math.sqrt(var)))
            lp = np.log(p)
            g = lp[1:] - lp[:-1]
            fisher = float(np.sum(p[1:] * g * g))
        L = len(p) - 1
        cdf = np.cumsum(p)
        with np.errstate(divide="ignore"):
            lp = np.log(p)
        W = L + 2
        cand = np.arange(-W, W + 1)
        succ = []
        for Q in qs:
            ok = 0.0
            for _ in range(trials):
                y = np.searchsorted(cdf, rng.random(Q) * cdf[-1])
                idx = y[:, None] + cand[None, :]
                valid = (idx >= 0) & (idx <= L)
                ll = np.where(valid, lp[np.clip(idx, 0, L)], -np.inf).sum(axis=0)
                m = ll.max()
                post = np.exp(ll - m)
                ok += post.max() / post.sum()
            succ.append(ok / trials)
        pred = [float(erf(math.sqrt(Q * fisher) / (2 * math.sqrt(2)))) for Q in qs]
        out.append({"kind": kind, "T": T, "N": N, "var": var, "fisher": fisher, "kappa": fisher * var, "q": qs,
                    "succ": succ, "pred": pred})
        print(kind, T, N, var, fisher * var, list(zip(qs, [round(s, 3) for s in succ]))[::8], flush=True)
    return out


def limits(lams=(96, 128, 160, 192, 256)):
    out = {}
    for lam in lams:
        P.set_lambda(lam)
        out[str(lam)] = {}
        for m in ("enp_spec", "ours", "lossless"):
            b = P.search(m)
            out[str(lam)][m] = {k: b[k] for k in ("total_bytes", "proof_bytes", "commit_bytes", "n", "logq", "mu", "nu", "sigma1", "sigma2", "beta", "P")}
            print(lam, m, out[str(lam)][m], flush=True)
    P.set_lambda(128)
    return out


def certified(lam=128):
    P.set_lambda(lam)
    o = P.search("ours")
    n, q, mu, nu, K = o["n"], o["q"], o["mu"], o["nu"], P.K
    e2 = o["eta"] ** 2
    s1, s2, kappa = o["sigma1"], o["sigma2"], o["kappa"]
    bc2 = P.challenge_info("ours", n)["s1"] ** 2
    s1max2 = P.challenge_info("enp", n)["s1"] ** 2
    kap_u = P.challenge_info("enp", n)["kappa"]

    def floor(B):
        return 1.0 / (1.0 / s1 ** 2 + B / s2 ** 2)

    def bits(w2):
        if w2 is None or w2 <= 0:
            return None
        return mlwe_bits(n, nu, mu + K, q, round(math.sqrt(w2 / (2 * math.pi)), 6))

    rng = np.random.default_rng(5)
    gs = random_challenges(rng, n, kappa, 4000)
    dets, detv, dets_psi = [], [], []
    for g in gs:
        sl = embed_sq(g.astype(float))
        if sl.max() > bc2:
            continue
        p = 1.0 / (1.0 / s1 ** 2 + sl / s2 ** 2)
        dets.append(float(np.exp(np.mean(np.log(p - 2 * e2)))))
        detv.append(float(np.exp(np.mean(np.log(0.5 * p * (1 + np.sqrt(1 - 4 * e2 / p)))))))
        dets_psi.append(float(np.exp(np.mean(np.log(p * p / (p - e2))))))
    rows = []
    for name, B, fn in (("klss", kap_u * kap_u, lambda Pf: Pf / 2), ("enp", s1max2, lambda Pf: Pf / 2),
                        ("enp_spec", bc2, lambda Pf: Pf / 2), ("ours", bc2, lambda Pf: tight_sigma0_sq(Pf, math.sqrt(e2)))):
        Pf = floor(B)
        w2 = fn(Pf)
        ok = w2 >= 2 * e2 and Pf >= 4 * e2
        rows.append({"name": name, "B": B, "P": Pf, "w2": w2, "valid": ok, "bits": bits(w2) if ok else None})
    for name, w2 in (("elliptic_det", float(np.median(dets))), ("elliptic_vth", float(np.median(detv))),
                     ("heuristic", floor(kappa)), ("upper_det", float(np.median(dets_psi)))):
        rows.append({"name": name, "w2": w2, "valid": True, "bits": bits(w2)})
    for r in rows:
        print(r, flush=True)
    P.set_lambda(128)
    return {"params": {k: o[k] for k in ("n", "q", "mu", "nu", "sigma1", "sigma2", "kappa", "eta")}, "bc2": bc2, "rows": rows}


def msglen(ks=(1, 4, 16)):
    out = {}
    for k in ks:
        P.K = k
        P.set_variant()
        out[str(k)] = {}
        for m in ("uniform", "gauss", "onebit", "klss", "enp", "ours", "ours_e"):
            b = P.search(m)
            out[str(k)][m] = None if b is None else {x: b[x] for x in ("total_bytes", "proof_bytes", "commit_bytes", "n", "logq", "mu", "nu")}
            print(k, m, out[str(k)][m], flush=True)
    P.K = 1
    P.set_variant()
    return out


def qrom_sets(chals=(192, 256)):
    out = {}
    for c in chals:
        P.set_variant(chal=c)
        out[str(c)] = {}
        for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
            b = P.search(m, ns=(128, 256, 512))
            out[str(c)][m] = None if b is None else {x: b[x] for x in ("total_bytes", "proof_bytes", "commit_bytes", "n", "logq", "mu", "nu", "kappa")}
            print(c, m, out[str(c)][m], flush=True)
    P.set_variant()
    return out


def pointwise_check(p1=8.0, p2=24.0, theta=0.3, L=40, W=22, alphas=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5)):
    from scipy.special import logsumexp

    def dg_log(S, c, X):
        Pm = np.linalg.inv(S)
        d = X - c
        return -math.pi * np.einsum("ij,jk,ik->i", d, Pm, d)

    def eps_of(s2, N=2):
        k = np.arange(-50, 51)
        return float(np.sum(np.exp(-math.pi * k * k * s2))) ** N - 1

    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    Sig = R @ np.diag([p1, p2]) @ R.T
    c1, c2 = np.zeros(2), np.array([0.3, -0.2])
    g = np.arange(-L, L + 1)
    X = np.array([[a, b] for a in g for b in g], dtype=float)
    Z = X[np.abs(X).max(axis=1) <= W]
    ldn = logsumexp(dg_log(Sig, c1 + c2, X))
    rows = []
    for a in alphas:
        S1, S2 = a * np.eye(2), Sig - a * np.eye(2)
        l1 = dg_log(S1, c1, X)
        l1 -= logsumexp(l1)
        l2n = logsumexp(dg_log(S2, c2, X))
        lp = np.array([logsumexp(l1 + dg_log(S2, c2, z[None, :] - X) - l2n) for z in Z])
        ld = dg_log(Sig, c1 + c2, Z) - ldn
        mask = ld > ld.max() - 25
        S3 = np.linalg.inv(np.linalg.inv(S1) + np.linalg.inv(S2))
        lm = float(np.linalg.eigvalsh(S3).min())
        e = eps_of(lm)
        rows.append({"a": a, "lmin3": lm, "eps": e, "bound": math.log((1 + e) / (1 - e)) if e < 1 else None,
                     "exact": float(np.abs(lp - ld)[mask].max()), "sd": float(0.5 * np.abs(np.exp(lp) - np.exp(ld)).sum())})
        print(rows[-1], flush=True)
    return {"p1": p1, "p2": p2, "rows": rows}


def renyi_to_gauss(P, mean, var, alpha):
    xs = np.arange(len(P)) - mean
    L = int(max(np.abs(xs).max() * 3, 20 * math.sqrt(var)))
    ys = np.arange(-L, L + 1)
    Z = np.exp(-ys * ys / (2 * var)).sum()
    q = np.exp(-xs * xs / (2 * var)) / Z
    return math.log(float(np.sum(P ** alpha * q ** (1 - alpha)))) / (alpha - 1)


def cbd_pmf(eta):
    return np.array([math.comb(2 * eta, k) for k in range(2 * eta + 1)], dtype=float) / 4 ** eta


def renyi_gauss(alphas=(2, 4, 16)):
    laws = []
    for T, N in ((4, 128), (4, 256), (8, 64), (8, 256), (16, 64)):
        P = su.su_pmf(N, T)
        laws.append(("SU", T, N, P, (len(P) - 1) / 2.0, su.su_variance(N, T), -1.2 / T))
    for eta in (2, 3, 4, 8, 16):
        laws.append(("CBD", eta, None, cbd_pmf(eta), float(eta), eta / 2.0, -1.0 / eta))
    rows = []
    for kind, a, N, P, mean, var, k4 in laws:
        rows.append({"kind": kind, "param": a, "N": N, "var": var, "k4": k4,
                     "D": {str(al): renyi_to_gauss(P, mean, var, al) for al in alphas}, "pred2": k4 * k4 / 24.0})
        print(rows[-1], flush=True)
    kat = 1024 * renyi_to_gauss(cbd_pmf(4), 4.0, 2.0, 2) + 256 * renyi_to_gauss(su.su_pmf(128, 4), 254.0, su.su_variance(128, 4), 2)
    rac = 4608 * renyi_to_gauss(su.su_pmf(64, 8), 252.0, su.su_variance(64, 8), 2)
    return {"rows": rows, "katana_ln": kat, "raccoon_ln": rac}


def filtering(n=128, kappas=range(31, 39), lam=128):
    from tighthint.challenges import spectral_ratio_delta
    out = []
    for kappa in kappas:
        lc = challenge_space_log2(n, kappa)
        delta = 1.0 - 2.0 ** (lam - lc)
        x = spectral_ratio_delta(n, delta)
        row = {"kappa": kappa, "log2C": lc, "delta": delta, "x": x, "bc2": x * kappa, "obj": kappa * (math.sqrt(x) + 1) ** 2}
        P.set_variant(spec=(kappa, x * kappa))
        for m in ("ours", "ours_e"):
            b = P.search(m, ns=(n,))
            row[m] = None if b is None else b["total_bytes"]
        out.append(row)
        print(row, flush=True)
    P.set_variant()
    return out


def norms(runs=3000, seed=7):
    from tighthint.pok import Scheme
    params = json.load(open(os.path.join(OUT, "params.json")))
    out = {}
    for m in ("klss", "enp", "ours", "ours_e"):
        cfg = params[m]
        rng = np.random.default_rng(seed)
        S = Scheme(cfg)
        msg = S.R.uniform(rng, (P.K,))
        sq, att, bad = [], 0, 0
        for _ in range(runs):
            c, aux = S.commit(msg, rng)
            proof, a = S.prove(c, aux, rng)
            att += a - 1
            z = S.decode(proof[32:], S.D * S.n)
            sq.append(float(np.sum(np.asarray(z, dtype=np.float64) ** 2)))
            bad += 0 if S.verify(c, proof) else 1
        sq = np.array(sq)
        zq = math.sqrt(2 * 129 * math.log(2))
        out[m] = {"runs": runs, "beta": cfg["beta"], "mean": float(np.sqrt(sq).mean()), "max": float(np.sqrt(sq).max()),
                  "extrap": float(math.sqrt(sq.mean() + zq * sq.std())), "retries": att, "verify_fail": bad}
        print(m, out[m], flush=True)
    return out


def raccoon_curves(alpha=32, omega=19, points=60):
    ex = json.load(open(os.path.join(OUT, "extra.json")))
    kap = {r["T"]: r["kappa"] for r in ex["raccoon"]}
    out = []
    for d, rep, ut, uw in ((1, 8, 6, 41), (4, 2, 6, 41)):
        T = d * (rep - 1) + 1
        N = 2 ** uw
        c = omega * T * 2 ** (ut - 1)
        a0 = int((alpha - 1) * c) + 1
        rows = []
        for i in range(points):
            a = int(a0 * 2 ** (i * 0.25))
            if 2 * a + c > T * (N - 1) or a + c > N:
                break
            e1, e2 = su.tail_log2(N, T, a - 1), su.tail_log2(N, T, a + c - 1)
            le = max(e1, e2) + math.log2(1.0 + 2.0 ** (min(e1, e2) - max(e1, e2)))
            sig2 = su.su_variance(N, T)
            scale = T * float(N) * float(N) / (alpha * c * c)
            lplain = math.log(0.5 * alpha * c * c * kap[T] / sig2) + max(alpha, 2.0) * c * su.gamma_edge(T, a) + math.log(scale)
            hi = min(N - c - 2, (T * (N - 1) - c) // 2)
            best = None
            for j in range(301):
                b = max(a, int(round(a * (hi / a) ** (j / 300))))
                v, _ = su.renyi_layered(N, T, c, alpha, a, b, kap[T])
                best = v if best is None or v < best else best
            rows.append({"a": a, "eps_log2": le, "plain_log10": lplain / math.log(10), "layered": best * scale})
        out.append({"d": d, "T": T, "rows": rows})
        print(d, T, rows[0], rows[-1], flush=True)
    return out


def edge_check(N=256, trials=200, seed=41, target=0.99):
    rng = np.random.default_rng(seed)
    laws = [("CBD", 1, cbd_pmf(1)), ("CBD", 2, cbd_pmf(2)), ("CBD", 3, cbd_pmf(3)),
            ("SU", (4, 2), su.su_pmf(4, 2)), ("SU", (8, 2), su.su_pmf(8, 2)), ("SU", (4, 3), su.su_pmf(4, 3))]
    out = []
    for kind, par, P in laws:
        L = len(P) - 1
        pplus = float(P[-1])
        cdf = np.cumsum(P)
        pred = math.log(N / (1 - target)) / pplus
        Q = 1
        while True:
            ok = 0
            for _ in range(trials):
                y = np.searchsorted(cdf, rng.random((Q, N)) * cdf[-1])
                ok += int(np.all(y.max(axis=0) == L))
            if ok >= target * trials:
                break
            Q = int(math.ceil(Q * 1.05)) if Q > 20 else Q + 1
        out.append({"kind": kind, "par": par, "pplus": pplus, "pred": pred, "meas": Q})
        print(out[-1], flush=True)
    return out


def main(which=None):
    path = os.path.join(OUT, "extra.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    jobs = {"sandwich": sandwich_curves, "bits": bits_recovered, "families": challenge_families,
            "alphabet": alphabet_study, "cbd": cbd_kappa, "kappaT": kappa_large_T, "lower": lower_bound_rows,
            "rates": map_rates, "multi": multi_proof, "sensitivity": sensitivity,
            "raccoon": raccoon_rows, "leak": leak_su, "limits": limits, "certified": certified, "msglen": msglen, "qromsets": qrom_sets, "pointwise": pointwise_check,
            "renyigauss": renyi_gauss, "filtering": filtering, "norms": norms,
            "raccooncurves": raccoon_curves, "edge": edge_check,
            "certified256": lambda: certified(256)}
    for k, fn in jobs.items():
        if which and k not in which:
            continue
        res[k] = fn()
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:] or None)


def negacyclic_mul(a, b):
    n = len(a)
    tw = np.exp(1j * np.pi * np.arange(n) / n)
    c = np.fft.ifft(np.fft.fft(a * tw) * np.fft.fft(b * tw)) / tw
    return np.real(c)


def conj_auto(a):
    n = len(a)
    out = np.zeros(n, dtype=np.float64)
    out[0] = a[0]
    out[1:] = -a[1:][::-1]
    return out


def power_bounds(n=128, kappa=31, trials=3000, seed=21):
    rng = np.random.default_rng(seed)
    ks = (1, 2, 4, 8, 16)
    exact, bounds = [], {k: [] for k in ks}
    for _ in range(trials):
        g = random_challenges(rng, n, kappa, 1)[0].astype(np.float64)
        exact.append(math.sqrt(float(np.max(embed_sq(g)))))
        p = g.copy()
        k = 1
        while k <= max(ks):
            if k in ks:
                prod = negacyclic_mul(conj_auto(p), p) if k > 1 else negacyclic_mul(conj_auto(g), g)
                bounds[k].append(float(np.sum(np.abs(prod))) ** (1.0 / (2 * k)))
            p = negacyclic_mul(p, p) if k < max(ks) else p
            k *= 2
    exact = np.array(exact)
    out = {"exact_q99": float(np.quantile(exact ** 2, 0.99)), "exact_med": float(np.median(exact ** 2)), "rows": []}
    for k in ks:
        b = np.array(bounds[k])
        out["rows"].append({"k": k, "q99": float(np.quantile(b ** 2, 0.99)), "median_ratio": float(np.median(b / exact)),
                            "max_ratio": float(np.max(b / exact))})
    print(out, flush=True)
    return out


def katana_ratios(trials=4000, seed=23):
    rng = np.random.default_rng(seed)
    out = []
    for n, cols in ((256, 4), (256, 6), (256, 8), (512, 4), (1024, 2)):
        r = []
        for _ in range(trials):
            M = sum(rng.integers(0, 2, size=(cols, n)) for _ in range(4)) - sum(rng.integers(0, 2, size=(cols, n)) for _ in range(4))
            S = np.sum(embed_sq(M.astype(float)), axis=0)
            r.append(float(S.max() / S.mean()))
        r = np.array(r)
        out.append({"n": n, "cols": cols, "median": float(np.median(r)), "q99": float(np.quantile(r, 0.99)), "max": float(r.max())})
        print(out[-1], flush=True)
    return out


def main2():
    path = os.path.join(OUT, "extra.json")
    res = json.load(open(path))
    res["powers"] = power_bounds()
    res["katana"] = katana_ratios()
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
