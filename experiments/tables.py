import json
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ORDER = ["uniform", "gauss", "onebit", "bimodal", "gpg", "flood", "klss", "enp", "enp_spec", "ours", "ours_e"]
NAMES = {"uniform": "Uniform", "gauss": "Gaussian", "onebit": "OneBit", "bimodal": "Bimodal", "gpg": "G+G",
         "flood": "Flooding", "klss": "Hint", "enp": "MatrixHint", "enp_spec": "MatrixHint+spec.",
         "ours": "\\Scheme", "ours_e": "\\Scheme-E"}


def width(p):
    if p.get("sigma1"):
        return "$%.2f$ / $%.2f$" % (p["sigma1"], p["sigma2"])
    if p["method"] == "uniform":
        return "$\\gamma_1=%d$" % p["gamma1"]
    s = p["s"]
    if s > 1e6:
        return "$2^{%.1f}$" % math.log2(s)
    return "$%.1f$" % s


def params_table(P):
    rows = []
    for m in ORDER:
        p = P[m]
        rows.append(" & ".join([
            NAMES[m], "$%d$" % p["n"], "$%d$" % p["logq"], "$%d$" % p["mu"], "$%d$" % p["nu"], "$%d$" % p["kappa"],
            width(p), "$%.1f$" % p["hid_sd"], "$%.1f$" % math.log2(p["beta_msis"]),
            "$%.1f$/$%.1f$" % (p["hiding_bits"], p["hiding_bits_q"]), "$%.1f$/$%.1f$" % (p["binding_bits"], p["binding_bits_q"]),
        ]) + "\\\\")
    return "\n".join(rows)


def kappa_table(T):
    rows = []
    for r in T["kappa"]:
        c = "$\\infty$" if r["cont"] == float("inf") or r["cont"] > 1e9 else "$%.5f$" % r["cont"]
        rows.append("$%d$ & %s & $%.5f$ & $%.5f$ & $%.5f$\\\\" % (r["T"], c, r["N6"], r["N10"], r["N14"]))
    return "\n".join(rows)


def su_rows(T):
    rows = []
    for r in T["su_table"]:
        rows.append("$%d$ & $%d$ & $%d$ & $%d$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.2f$ & $%.2f$\\\\" % (
            r["T"], r["c"], int(r["alpha"]), r["a"], r["kappa"], r["exact"], r["ours"], r["conj"], r["lemma42"]))
    return "\n".join(rows)


def main():
    P = json.load(open(os.path.join(RES, "params.json")))
    T = json.load(open(os.path.join(RES, "theory.json")))
    out = "%% params\n" + params_table(P) + "\n%% kappa\n" + kappa_table(T) + "\n%% su\n" + su_rows(T) + "\n"
    with open(os.path.join(RES, "tables.tex"), "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()


def slot_bound(c, rhs):
    lo, hi = 1.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c * (mid - 1 - math.log(mid)) < rhs:
            lo = mid
        else:
            hi = mid
    return lo


def uniform_opt(K, Q):
    tot = K * (1.0 / K) ** Q / K
    for r in range(1, K):
        tot += (K - r) * (((r + 1) / K) ** Q - 2 * (r / K) ** Q + ((r - 1) / K) ** Q) / (K - r)
    return tot


def uniform_q(K, p):
    Q = 1
    while uniform_opt(K, Q) < p:
        Q += 1
    return Q


def appendix_tables():
    ex = json.load(open(os.path.join(RES, "extra.json")))
    out = {}
    rows = []
    for r in ex["bits"]:
        b = r["bits"]
        gap = b["upper"] - b["klss"]
        rec = "--" if gap <= 1e-9 or r["ratio"] == 4 else "$%d\\%%$" % round(100 * (b["ours"] - b["klss"]) / gap)
        rows.append("%s & $%d$ & $%.1f$ & $%.1f$ & $%.1f$ & %s\\\\" % (r["name"], r["ratio"], b["klss"], b["ours"], b["upper"], rec))
    out["BITSROWS"] = "\n".join(rows)
    rows = []
    for r in ex["families"]:
        rows.append("$%d$ & $%d$ & $%.2f$ & $%.1f$ & $%.1f\\%%$ & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$\\\\" % (
            r["n"], r["kappa"], r["x_n"], r["rigorous"], 100 * r["fraction"], r["median"], r["q99"], r["hoeffding"], r["worst"]))
    out["FAMROWS"] = "\n".join(rows)
    rows = []
    for Q in sorted(int(k) for k in ex["multi"]):
        d = ex["multi"][str(Q)]
        vals = [d[m]["total_bytes"] for m in ("uniform", "gauss", "klss", "enp")]
        ours = min(d["ours"]["total_bytes"], d["ours_u"]["total_bytes"])
        rows.append("$%d$ & " % Q + " & ".join("$%d$" % v for v in vals) + " & $\\mathbf{%d}$\\\\" % ours)
    out["MULTIROWS"] = "\n".join(rows)
    rows = []
    for r in ex["lower"]:
        rows.append("$%d$ & $%d$ & $%d$ & $%d$ & $%d$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$\\\\" % (
            round(math.log2(r["N"])), r["T"], r["c"], r["alpha"], r["a"], r["lower"], r["exact"], r["upper"], r["kappa"]))
    out["LOWERROWS"] = "\n".join(rows)
    rows = []
    for r in ex["kappaT"]:
        rows.append("$%d$ & $%.6f$ & $%.4f$ & $%.6f$\\\\" % (r["T"], r["kappa"], r["scaled"], r["edgeworth"]))
    out["KAPPAROWS"] = "\n".join(rows)
    etas = [r["eta"] for r in ex["cbd"]]
    out["CBDHEAD"] = "$\\eta_{\\mathrm{B}}$ & " + " & ".join("$%d$" % e for e in etas) + "\\\\"
    out["CBDROW"] = "$\\frac{\\eta_{\\mathrm{B}}}{2}I$ & " + " & ".join("$%.3f$" % r["kappa"] for r in ex["cbd"]) + "\\\\"
    from tighthint.challenges import challenge_space_log2, spectral_ratio
    rows = []
    for lc in (128, 192, 256):
        cells = []
        for n in (128, 256, 512, 1024):
            k = next((kk for kk in range(1, n + 1) if challenge_space_log2(n, kk) - 1 >= lc), None)
            cells.append("--" if k is None else "$%d/%.0f$" % (k, spectral_ratio(n) * k))
        rows.append("$%d$ & " % lc + " & ".join(cells) + "\\\\")
    out["QROMROWS"] = "\n".join(rows)
    lat = json.load(open(os.path.join(RES, "latency.json")))
    names = {"uniform": "Uniform", "gauss": "Gaussian", "onebit": "OneBit", "bimodal": "Bimodal", "gpg": "G+G",
             "klss": "Hint", "enp": "MatrixHint", "ours": "\\Scheme"}
    rows = []
    for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours"):
        p = np.array(lat[m]["prove"])
        q = np.percentile(p, [50, 90, 99, 100])
        fmt = "$\\mathbf{%.2f}$" if m == "ours" else "$%.2f$"
        rows.append(names[m] + " & $%d$ & " % len(p) + " & ".join(fmt % v for v in q) + "\\\\")
    out["LATROWS"] = "\n".join(rows)
    rob = ex["sensitivity"]
    rows = []
    for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
        cells = []
        for v in ("base", "quantum", "tail40", "primal"):
            x = rob[v][m]["total_bytes"]
            cells.append(("$\\mathbf{%d}$" if m.startswith("ours") else "$%d$") % x)
        rows.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
    out["ROBROWS"] = "\n".join(rows)
    scan = json.load(open(os.path.join(RES, "lambda_scan.json")))
    rows = []
    for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
        cells = []
        for lam in ("96", "128", "160", "192", "256"):
            b = scan[lam][m]
            cells.append("$%d$ \\scriptsize{(%d, %d, %d, %d)}" % (b["total_bytes"], b["n"], b["logq"], b["mu"], b["nu"]))
        rows.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
    out["LAMROWS"] = "\n".join(rows)

    rows = []
    for lam in ("96", "128", "160", "192", "256"):
        for m in ("ours", "ours_e"):
            o = scan[lam][m]
            rows.append(("$%s$ & " % lam if m == "ours" else " & ") + NAMES[m] + " & $%d$ & $%d$ & $(%d,%d)$ & $%d$ & $%.2f$ & $%.2f$ & $%d$ & $%d$ & $%d$ & $%d$ & $%.1f$ & $%.1f$\\\\" % (
                o["n"], o["logq"], o["mu"], o["nu"], o["kappa"], o["sigma1"], o["sigma2"], round(o["beta"]), o["proof_bytes"],
                o["commit_bytes"], o["total_bytes"], o["hiding_bits"], o["binding_bits"]))
    out["PARAMROWS"] = "\n".join(rows)
    pw = ex["powers"]
    rows = []
    for r in pw["rows"]:
        rows.append("$%d$ & $%.1f$ & $%.3f$ & $%.3f$\\\\" % (r["k"], r["q99"], r["median_ratio"], r["max_ratio"]))
    rows.append("exact & $%.1f$ & $1$ & $1$\\\\" % pw["exact_q99"])
    out["POWROWS"] = "\n".join(rows)
    rows = []
    for r in ex["katana"]:
        rows.append("$%d$ & $%d$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$\\\\" % (r["n"], r["cols"], r["median"], r["q99"], r["max"], slot_bound(r["cols"], math.log(r["n"]))))
    out["KATROWS"] = "\n".join(rows)

    if "leak" in ex:
        from scipy.special import erfinv
        rows = []
        for r in ex["leak"]:
            qs = np.log2(np.array(r["q"], dtype=float))
            sc = np.array(r["succ"])
            cells = []
            for p in (0.5, 0.9):
                meas = 2 ** float(np.interp(p, sc, qs)) if sc[-1] >= p else float("nan")
                pred = 8 * erfinv(p) ** 2 / r["fisher"] if r["fisher"] > 0 else uniform_q(r["N"], p)
                cells.append("$%d$" % round(meas))
                cells.append("--" if math.isnan(pred) else "$%d$" % round(pred))
            lab = "Gaussian" if r["kind"] == "DG" else "$\\SU(%d,%d)$" % (r["N"], r["T"])
            edge = "--" if r["kind"] == "DG" else "$2^{%.1f}$" % (r["T"] * math.log2(r["N"]))
            kap = "$\\infty$" if r["fisher"] == 0 else "$%.3f$" % r["kappa"]
            rows.append(lab + " & $%.1f$ & %s & " % (r["var"], kap) + " & ".join(cells) + " & " + edge + "\\\\")
        out["LEAKROWS"] = "\n".join(rows)
    if "raccoon" in ex:
        rows = []
        for r in ex["raccoon"]:
            al = r["alphas"]
            cells = ["$%.4f$" % al[a]["layered"] for a in ("2", "8", "32", "128")]
            rows.append("$%d$ & $%d$ & $%d$ & $%d$ & $%d$ & $%d$ & $%.4f$ & " % (r["d"], r["rep"], r["T"], r["ut"], r["uw"], r["c"], 6 * r["kappa"])
                        + " & ".join(cells) + " & $%.1f$ & $10^{%.1f}$\\\\" % (al["32"]["eps_log2"], al["8"]["plain_log10"]))
        out["RACROWS"] = "\n".join(rows)

    for key, tag in (("certified", "CERTROWS"), ("certified256", "CERT256ROWS")):
        if key not in ex:
            continue
        names = {"klss": "Hint~\\cite{KLSS23}", "enp": "MatrixHint~\\cite{ENP24}", "enp_spec": "equal split, $\\Cspec$",
                 "ours": "Theorem~\\ref{thm:tight}", "elliptic_det": "Corollary~\\ref{cor:ring}", "elliptic_vth": "Theorem~\\ref{thm:general}",
                 "heuristic": "Heuristic~3", "upper_det": "Theorem~\\ref{thm:minouter}"}
        rows = []
        for r in ex[key]["rows"]:
            B = "$%.1f$" % r["B"] if "B" in r else "--"
            Pf = "$%.1f$" % r["P"] if "P" in r else "--"
            bits = "--" if r["bits"] is None else "$%.1f$" % r["bits"]
            rows.append(names[r["name"]] + " & " + B + " & " + Pf + " & $%.1f$ & " % r["w2"] + bits + "\\\\")
        out[tag] = "\n".join(rows)
    if False:
        names = {"klss": "Hint~\\cite{KLSS23}", "enp": "MatrixHint~\\cite{ENP24}", "enp_spec": "equal split, $\\Cspec$",
                 "ours": "Theorem~\\ref{thm:tight}", "elliptic_det": "Corollary~\\ref{cor:ring}", "elliptic_vth": "Theorem~\\ref{thm:general}",
                 "heuristic": "Heuristic~3", "upper_det": "Theorem~\\ref{thm:minouter}"}
        rows = []
        for r in ex["certified"]["rows"]:
            B = "$%.1f$" % r["B"] if "B" in r else "--"
            Pf = "$%.1f$" % r["P"] if "P" in r else "--"
            bits = "--" if r["bits"] is None else "$%.1f$" % r["bits"]
            rows.append(names[r["name"]] + " & " + B + " & " + Pf + " & $%.1f$ & " % r["w2"] + bits + "\\\\")
        out["CERTROWS"] = "\n".join(rows)
    if "limits" in ex:
        rows = []
        for lam in sorted(ex["limits"], key=int):
            d = ex["limits"][lam]
            a, b, c = d["enp_spec"]["total_bytes"], d["ours"]["total_bytes"], d["lossless"]["total_bytes"]
            rows.append("$%s$ & $%d$ & $%d$ & $%d$ & $%.1f\\%%$\\\\" % (lam, a, b, c, 100.0 * (b - c) / b))
        out["LIMROWS"] = "\n".join(rows)

    if "msglen" in ex:
        ks = sorted(ex["msglen"], key=int)
        rows = []
        for m in ("uniform", "gauss", "onebit", "klss", "enp", "ours", "ours_e"):
            cells = []
            for k in ks:
                b = ex["msglen"][k][m]
                cells.append("--" if b is None else (("$\\mathbf{%d}$" if m.startswith("ours") else "$%d$") % b["total_bytes"]))
            rows.append(NAMES[m] + " & " + " & ".join(cells) + "\\\\")
        out["MSGROWS"] = "\n".join(rows)

    if "qromsets" in ex:
        base = json.load(open(os.path.join(RES, "params.json")))
        rows = []
        for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
            vals = [base[m]["total_bytes"]] + [ex["qromsets"][c][m]["total_bytes"] for c in ("192", "256")]
            fmt = "$\\mathbf{%d}$" if m.startswith("ours") else "$%d$"
            rows.append(NAMES[m] + " & " + " & ".join(fmt % v for v in vals) + "\\\\")
        out["QSIZEROWS"] = "\n".join(rows)

    if "pointwise" in ex:
        rows = []
        for r in ex["pointwise"]["rows"]:
            b = "--" if r["bound"] is None else "$%.4f$" % r["bound"]
            e = "$>1$" if r["eps"] >= 1 else "$%.4f$" % r["eps"]
            rows.append("$%.1f$ & $%.3f$ & %s & %s & $%.4f$ & $%.4f$\\\\" % (r["a"], r["lmin3"], e, b, r["exact"], r["sd"]))
        out["PWROWS"] = "\n".join(rows)

    if "renyigauss" in ex:
        rows = []
        for r in ex["renyigauss"]["rows"]:
            lab = "$\\SU(%d,%d)$" % (r["N"], r["param"]) if r["kind"] == "SU" else "CBD, $\\eta_{\\mathrm{B}}=%d$" % r["param"]
            rows.append(lab + " & $%g$ & " % r["var"] + " & ".join("$%.5f$" % r["D"][a] for a in ("2", "4", "16")) + " & $%.5f$\\\\" % r["pred2"])
        out["RGROWS"] = "\n".join(rows)

    if "filtering" in ex:
        rows = []
        for r in ex["filtering"]:
            rows.append("$%d$ & $%.1f$ & $%.4f$ & $%.3f$ & $%.1f$ & $%.1f$ & $%d$ & $%d$\\\\" % (
                r["kappa"], r["log2C"], r["delta"], r["x"], r["bc2"], r["obj"], r["ours"], r["ours_e"]))
        out["FILTROWS"] = "\n".join(rows)

    if "edge" in ex:
        rows = []
        for r in ex["edge"]:
            lab = "CBD, $\\eta_{\\mathrm{B}}=%d$" % r["par"] if r["kind"] == "CBD" else "$\\SU(%d,%d)$" % tuple(r["par"])
            rows.append(lab + " & $1/%d$ & $%d$ & $%d$ & $%.2f$\\\\" % (round(1 / r["pplus"]), round(r["pred"]), r["meas"], r["meas"] / r["pred"]))
        out["EDGEROWS"] = "\n".join(rows)
    lpath = os.path.join(RES, "lattice_attack.json")
    if os.path.exists(lpath):
        la = json.load(open(lpath))
        names = [("elliptic", "elliptical, whitened"), ("elliptic_naive", "elliptical, unwhitened"),
                 ("det", "spherical, $\\det(\\Sig)^{1/n}$"), ("min", "spherical, $\\lambda_{\\min}(\\Sig)$"),
                 ("prior", "spherical, $\\sigma_1^2$")]
        rows = []
        for cfg in la:
            p = cfg["params"]
            w = {"elliptic": p["det_w2"], "elliptic_naive": p["det_w2"], "det": p["det_w2"], "min": p["min_w2"], "prior": p["prior_w2"]}
            first = True
            for k, lab in names:
                if k not in cfg["runs"]:
                    continue
                bs = [x["b"] for x in cfg["runs"][k]]
                ok = [b for b in bs if b is not None]
                med = np.median([b if b is not None else 10 ** 6 for b in bs])
                medtxt = "$>60$" if med > 60 else "$%g$" % med
                meantxt = "$%.1f$" % np.mean(ok) if ok else "--"
                head = ("$%d$, $%d$, $%g$, $%g$" % (p["n"] * (p["d"] + p["m"]), p["q"], p["s1"], p["s2"])) if first else ""
                rows.append(head + " & " + lab + " & $%.1f$ & $%d/%d$ & %s & %s\\\\" % (w[k], len(ok), len(bs), meantxt, medtxt))
                first = False
            rows.append("\\midrule")
        out["LATTROWS"] = "\n".join(rows[:-1])
    with open(os.path.join(RES, "appendix_tables.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "appendix":
    for k, v in appendix_tables().items():
        print("==", k)
        print(v)
