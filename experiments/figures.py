import json
import math
import os
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("ps")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint import su
from tighthint.params import LABELS

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(RES, "figures")
ORDER = ["uniform", "gauss", "onebit", "bimodal", "gpg", "flood", "klss", "enp", "ours", "ours_e"]
COLORS = {"uniform": "#8c564b", "gauss": "#9467bd", "onebit": "#d62728", "bimodal": "#e377c2", "gpg": "#7f7f7f",
          "flood": "#bcbd22", "klss": "#ff7f0e", "enp": "#1f77b4", "ours": "#2ca02c", "ours_e": "#17becf",
          "enp_spec": "#aec7e8"}
SHORT = {"uniform": "Uniform", "gauss": "Gaussian", "onebit": "OneBit", "bimodal": "Bimodal", "gpg": "G+G",
         "flood": "Flooding", "klss": "Hint", "enp": "MatrixHint", "ours": "Ours", "ours_e": "Ours-E",
         "enp_spec": "MatrixHint+Spec"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "ps.useafm": False, "ps.fonttype": 42})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    eps = os.path.join(FIG, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    tool = shutil.which("epstopdf")
    if tool:
        subprocess.run([tool, eps, "--outfile=" + os.path.join(FIG, name + ".pdf")], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["ps2pdf", "-dEPSCrop", eps, os.path.join(FIG, name + ".pdf")], check=True)


def fig_sizes(params):
    ms = [m for m in ORDER if params.get(m)]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    y = np.arange(len(ms))
    cb = np.array([params[m]["commit_bytes"] for m in ms]) / 1024.0
    pb = np.array([params[m]["proof_bytes"] for m in ms]) / 1024.0
    ax.barh(y, cb, color="#c7c7c7", edgecolor="black", linewidth=0.4, label="commitment")
    ax.barh(y, pb, left=cb, color=[COLORS[m] for m in ms], edgecolor="black", linewidth=0.4, label="proof")
    for i, m in enumerate(ms):
        ax.text(cb[i] + pb[i], i, " %.2f" % (cb[i] + pb[i]), va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[m] for m in ms])
    ax.set_xscale("log")
    ax.set_xlim(1, 2000)
    ax.set_xlabel("communication (KiB, log scale)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_sizes")


def fig_time(bench):
    ms = [m for m in ORDER if m in bench]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(ms))
    w = 0.27
    for j, (key, lab, col) in enumerate([("commit_ms", "commit", "#c7c7c7"), ("prove_ms", "prove", "#4c72b0"),
                                         ("verify_ms", "verify", "#dd8452")]):
        ax.bar(x + (j - 1) * w, [bench[m][key] for m in ms], w, label=lab, color=col, edgecolor="black", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ms], rotation=45, ha="right")
    ax.set_ylabel("median time (ms, log scale)")
    ax.set_ylim(1, 3e4)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_time")


def fig_lambda(scan):
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    lams = sorted(int(k) for k in scan)
    for m in ["uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"]:
        xs, ys = [], []
        for lam in lams:
            v = scan[str(lam)].get(m)
            if v:
                xs.append(lam)
                ys.append(v["total_bytes"] / 1024.0)
        style = "-" if m.startswith("ours") else "--"
        ax.plot(xs, ys, style, marker="o", ms=2.5, lw=1.2 if m.startswith("ours") else 0.8, color=COLORS[m], label=SHORT[m])
    ax.set_xlabel(r"security level $\lambda$ (bits)")
    ax.set_ylabel("commitment + proof (KiB)")
    ax.set_yscale("log")
    ax.set_yticks([3, 5, 10, 20, 30])
    ax.set_yticklabels(["3", "5", "10", "20", "30"])
    ax.legend(frameon=False, ncol=3, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    save(fig, "fig_lambda")


def fig_ablation(params):
    ms = ["klss", "enp", "enp_spec", "ours", "ours_e"]
    labs = ["Hint", "MatrixHint", "+ spectral set", "+ tight reduction", "+ elliptical"]
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    tot = [params[m]["total_bytes"] for m in ms]
    ax.bar(range(len(ms)), tot, color=[COLORS[m] for m in ms], edgecolor="black", linewidth=0.4)
    for i, t in enumerate(tot):
        ax.text(i, t + 40, str(t), ha="center", fontsize=6)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylabel("total bytes")
    ax.set_ylim(0, max(tot) * 1.15)
    save(fig, "fig_ablation")


def fig_factor(th):
    f = th["factor"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(f["x"], f["ours"], color=COLORS["ours"], lw=1.4, label=r"Theorem 3.2: $\vartheta(P)/P$")
    ax.plot(f["x"], f["lower"], ":", color="black", lw=0.8, label=r"$1-2\eta^2/P$")
    ax.plot(f["x"], f["klss"], "--", color=COLORS["klss"], lw=1.2, label="factor two [KLSS23, ENP24]")
    ax.set_xscale("log")
    ax.set_xlabel(r"$P/\eta^2$")
    ax.set_ylabel(r"$\sigma_0^2/P$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_factor")


def fig_spectrum(th):
    c = th["challenges"]
    h = np.array(c["hist"])
    lo, hi = c["hist_range"]
    edges = np.linspace(lo, hi, len(h) + 1)
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.bar(edges[:-1], h / h.sum(), width=edges[1] - edges[0], align="edge", color="#9ecae1", edgecolor="#3182bd", linewidth=0.3)
    ymax = (h / h.sum()).max()
    for val, lab, col, ls in [(c["kappa"], r"$\kappa$", "black", ":"), (c["rigorous"], r"$\beta_c^2$", COLORS["ours"], "-"),
                              (c["hoeffding"], "Hoeffding", "#636363", "--")]:
        ax.axvline(val, color=col, ls=ls, lw=1.0)
        ax.text(val + 5, ymax * 0.55, lab, fontsize=6.5, color=col, rotation=90, va="bottom")
    ax.set_xlim(0, hi)
    ax.set_xlabel(r"$s_1(\gamma)^2=\max_j|\gamma(\zeta_j)|^2$")
    ax.set_ylabel("fraction of challenges")
    save(fig, "fig_spectrum")


def fig_logdet(th):
    L = th["logdet"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for key, col, lab in [("worst", "#fdae6b", "worst slot $p_{\\min}-2\\eta^2$"), ("det", "#74c476", "determinant width")]:
        h = np.array(L[key + "_hist"], dtype=float)
        e = np.array(L[key + "_edges"])
        ax.bar(e[:-1], h / h.sum(), width=e[1] - e[0], align="edge", color=col, alpha=0.8, label=lab, linewidth=0)
    for val, lab, ls in [(L["spherical"], r"$\vartheta(P)$", "--"), (L["jensen"], "Jensen bound", "-"), (L["heuristic3"], "Heuristic 3", ":")]:
        ax.axvline(val, color="black", ls=ls, lw=0.9)
        ax.text(val + 3, 0.97, lab, rotation=90, fontsize=6, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"width$^2$ of the MLWE secret after one hint")
    ax.set_ylabel("fraction")
    ax.legend(frameon=False, loc="center left", fontsize=6)
    save(fig, "fig_logdet")


def fig_su():
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    us = list(range(14, 44, 2))
    for T, col in [(4, "#d62728"), (6, "#ff7f0e"), (8, "#1f77b4"), (16, "#2ca02c")]:
        k = su.kappa_continuous_exact(T)
        xs, ys = [], []
        for u in us:
            r = su.best_a_asym(2 ** u, T, 1, 8.0, k, -64)
            if r is None:
                continue
            g = 8.0 / (2 * su.su_variance(2 ** u, T))
            xs.append(u)
            ys.append(r[0] / g)
        ax.plot(xs, ys, marker="o", ms=2, lw=1.0, color=col, label=r"$T=%d$" % T)
        ax.axhline(k, color=col, ls=":", lw=0.7)
    ax.axhline(1 + 2 / 7.0, color="black", ls="--", lw=0.8)
    ax.text(15, 1 + 2 / 7.0 + 0.02, r"Conjecture C.1 with $C=6$, $\alpha=8$", fontsize=6)
    ax.set_ylim(0.95, 1.6)
    ax.set_xlabel(r"$\log_2 N$")
    ax.set_ylabel(r"bound $/\,(\alpha c^2/2\sigma^2)$")
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=6)
    save(fig, "fig_su")


def fig_rates(th):
    r = th["rates"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for key, lab, col, ls in [("uniform", r"uniform ($T=1$)", "#d62728", "-"), ("su3", r"SU, $T=3$", "#ff7f0e", "--"),
                              ("su4", r"SU, $T=4$", "#1f77b4", "-."), ("gauss", "Gaussian", "#2ca02c", ":")]:
        ax.plot(r["q"], r[key], ls, marker="o", ms=2.5, color=col, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"number of hints $Q$")
    ax.set_ylabel("recovered fraction")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig_rates")


def fig_width():
    from tighthint.challenges import spectral_ratio, worst_unfiltered_s1
    n = 256
    ks = np.arange(4, 129, 2)
    xn = spectral_ratio(n)
    hint = ks + np.sqrt(ks)
    mhint = np.array([worst_unfiltered_s1(n, int(k)) for k in ks]) + np.sqrt(ks)
    ours = np.sqrt(ks) * (np.sqrt(xn) + 1)
    eta = 5.63
    rej = 0.675 * np.sqrt(2 * np.pi) * np.sqrt(ks * 2 * 16 * n / 3.0) / (2 * eta)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(ks, hint, "--", color=COLORS["klss"], label=r"Hint: $\kappa+\sqrt{\kappa}$")
    ax.plot(ks, mhint, "-.", color=COLORS["enp"], label=r"MatrixHint: $\max_{C_\kappa}s_1+\sqrt{\kappa}$")
    ax.plot(ks, ours, "-", lw=1.5, color=COLORS["ours"], label=r"Ours: $(\sqrt{x_n}+1)\sqrt{\kappa}$")
    ax.plot(ks, rej, ":", color=COLORS["onebit"], label=r"OneBit: $0.675\,T$, $m=16n$")
    ax.set_xlabel(r"challenge weight $\kappa$ ($n=256$)")
    ax.set_ylabel(r"$\sqrt{2\pi}\,\sigma_z/(2\eta)$")
    ax.set_yscale("log")
    ax.set_ylim(4, 2000)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=1)
    save(fig, "fig_width")


def fig_split(th):
    sp = th["split"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    cols = ["#9467bd", "#1f77b4", "#2ca02c"]
    for c, col in zip(sp["curves"], cols):
        y = np.log2(np.maximum(np.array(c["tv"]), 1e-300))
        ax.plot(c["frac"], y, color=col, lw=1.1, label=r"$P=%d$" % int(c["P"]))
        ax.axvline(c["theta"], color=col, ls=":", lw=0.9)
    ax.axhline(math.log2(8 * 2.0 ** -64), color="black", ls="--", lw=0.8, label=r"$8\varepsilon$")
    ax.axvline(0.5, color=COLORS["klss"], ls="--", lw=0.8)
    ax.set_xlabel(r"split $\sigma_0^2/P$")
    ax.set_ylabel(r"$\log_2$ statistical distance")
    ax.set_ylim(-200, 5)
    ax.legend(frameon=False, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_split")


def fig_many(th):
    m = th["many"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(m["q"], m["bound"], "-", color="black", lw=1.1, label=r"bound $t_Q/Q$, probability $1-2^{-128}$")
    ax.plot(m["q"], m["max"], "--", marker="o", ms=2.5, color=COLORS["onebit"], label="largest of 200 samples")
    ax.plot(m["q"], m["median"], ":", marker="s", ms=2.5, color=COLORS["ours"], label="median of 200 samples")
    ax.axhline(1.0, color="gray", lw=0.7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"number of independent ring hints $Q$")
    ax.set_ylabel(r"$s_1(\mathbf{M})^2/(Q\kappa)$")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "fig_many")


def fig_sandwich(ex):
    c = ex["sandwich"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(c["x"], c["upper"], "-", color="#d62728", lw=1.2, label=r"$\psi(p)/p$, sharp upper")
    ax.plot(c["x"], c["spherical_upper"], ":", color="#d62728", lw=1.0, label=r"$(p+\frac{4}{3}\eta^2)/p$, spherical upper")
    ax.plot(c["x"], c["lower"], "-", color=COLORS["ours"], lw=1.2, label=r"$\vartheta(p)/p$, sharp lower")
    ax.plot(c["x"], c["klss"], "--", color=COLORS["klss"], lw=1.0, label="equal split")
    ax.axhline(1.0, color="gray", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p/\eta^2$")
    ax.set_ylabel("width relative to $p$")
    ax.set_ylim(0.4, 1.4)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "fig_sandwich")


def fig_multi(ex):
    d = ex["multi"]
    qs = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for m, lab in [("uniform", "Uniform"), ("gauss", "Gaussian"), ("klss", "Hint"), ("enp", "MatrixHint"), ("ours", "SpecHint")]:
        ys = [d[str(q)][m]["total_bytes"] / 1024.0 for q in qs]
        ax.plot(qs, ys, "-" if m == "ours" else "--", marker="o", ms=2.5, lw=1.4 if m == "ours" else 0.9, color=COLORS[m], label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"proofs per commitment $Q$")
    ax.set_ylabel("KiB per proof")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "fig_multi")


def fig_map(ex):
    r = ex["rates"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.plot(r["q"], r["uniform"], "-", marker="o", ms=2.5, color="#d62728", label="uniform, exact optimum")
    ax.plot(r["q"], r["uniform_approx"], ":", color="#d62728", lw=0.9, label=r"$\min(1,Q/K)$")
    ax.plot(r["q"], r["gauss"], "-", marker="s", ms=2.5, color=COLORS["ours"], label="Gaussian, exact optimum")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"number of hints $Q$")
    ax.set_ylabel("optimal success probability")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "fig_map")


def fig_leak(ex):
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    cols = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4", 4: "#9467bd", 8: "#8c564b", 0: COLORS["ours"]}
    for r in ex["leak"]:
        lab = "Gaussian" if r["kind"] == "DG" else "$T=%d$" % r["T"]
        ax.plot(r["q"], r["succ"], "-", marker="o", ms=1.8, lw=1.0, color=cols[r["T"]], label=lab)
        if r["kind"] == "SU" and r["T"] in (2, 8):
            ax.plot(r["q"], r["pred"], ":", lw=0.9, color=cols[r["T"]])
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"number of hints $Q$")
    ax.set_ylabel("optimal success probability")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "fig_leak")


def fig_latency(lat):
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for m in ("ours", "enp", "klss", "gauss", "onebit", "gpg", "uniform", "bimodal"):
        p = np.sort(np.array(lat[m]["prove"]))
        y = np.arange(1, len(p) + 1) / len(p)
        ax.step(p, y, where="post", lw=1.4 if m == "ours" else 0.9, color=COLORS[m],
                label="SpecHint" if m == "ours" else SHORT[m])
    ax.set_xscale("log")
    ax.set_xlabel("proving time (ms)")
    ax.set_ylabel("fraction of proofs")
    ax.legend(frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig_latency")


def fig_lattice(lat):
    kinds = [("elliptic", "E"), ("elliptic_naive", "U"), ("det", "D"), ("min", "W"), ("prior", "P")]
    cols = 2 if len(lat) > 2 else len(lat)
    rows = (len(lat) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4, 1.7 * rows), sharey=True, squeeze=False)
    for idx, cfg in enumerate(lat):
        ax = axes[idx // cols][idx % cols]
        present = [(k, lab) for k, lab in kinds if k in cfg["runs"]]
        data = [[r["b"] if r["b"] is not None else 64 for r in cfg["runs"][k]] for k, _ in present]
        pos = list(range(1, len(present) + 1))
        ax.boxplot(data, positions=pos, widths=0.6, showfliers=True, flierprops={"markersize": 2},
                   medianprops={"color": "k", "lw": 1.0}, boxprops={"lw": 0.7}, whiskerprops={"lw": 0.7},
                   capprops={"lw": 0.7})
        ax.set_xticks(pos)
        ax.set_xticklabels([lab for _, lab in present], fontsize=6.5)
        p = cfg["params"]
        ax.set_title("$q=%d$, dim. $%d$, $Q=%d$" % (p["q"], p["n"] * (p["d"] + p["m"]), p.get("hints", 1)), fontsize=6.5)
        ax.axhline(60, color="#999999", lw=0.6, ls=":")
        if idx % cols == 0:
            ax.set_ylabel("success block size")
    for idx in range(len(lat), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout(pad=0.3)
    save(fig, "fig_lattice")


def fig_racc(ex):
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    cols = {8: "#1f77b4", 5: "#d62728"}
    for cur in ex["raccooncurves"]:
        xs = [r["eps_log2"] for r in cur["rows"]]
        ax.plot(xs, [10 ** r["plain_log10"] for r in cur["rows"]], "--", lw=1.0, color=cols[cur["T"]],
                label="Theorem, $T=%d$" % cur["T"])
        ax.plot(xs, [r["layered"] for r in cur["rows"]], "-", lw=1.4, color=cols[cur["T"]], label="layered, $T=%d$" % cur["T"])
    ax.axhline(6.0, color="#999999", lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_2$ of the smoothing mass per coordinate")
    ax.set_ylabel("provable constant")
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig_racc")


def fig_certified(ex):
    labels = {"klss": "Hint", "enp": "MatrixHint", "enp_spec": "equal split", "ours": "Theorem (tight)",
              "elliptic_det": "determinant", "elliptic_vth": "general priors", "heuristic": "Heuristic 3", "upper_det": "upper bound"}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.1), sharey=True)
    for ax, key, lam in ((axes[0], "certified", 128), (axes[1], "certified256", 256)):
        rows = ex[key]["rows"]
        names = [labels[r["name"]] for r in rows]
        vals = [r["bits"] if r["bits"] is not None else 0.0 for r in rows]
        cols = ["#bbbbbb" if r["bits"] is None else (COLORS["ours"] if r["name"] in ("ours", "elliptic_vth") else "#7f7f7f") for r in rows]
        ypos = list(range(len(rows)))[::-1]
        ax.barh(ypos, [v - lam + 20 if v else 0 for v in vals], left=lam - 20, color=cols, height=0.6)
        ax.axvline(lam, color="k", lw=0.7, ls=":")
        ax.set_xlim(lam - 20, lam + 30 if lam == 256 else lam + 20)
        ax.set_title(r"$\lambda=%d$" % lam, fontsize=7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("core-SVP bits", fontsize=7)
        for y, r in zip(ypos, rows):
            if r["bits"] is None:
                ax.text(lam - 19, y, "n/a", va="center", fontsize=5.5)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_certified")


def fig_filter(ex):
    rows = ex["filtering"]
    fig, ax1 = plt.subplots(figsize=(3.3, 2.0))
    ks = [r["kappa"] for r in rows]
    ax1.plot(ks, [r["ours"] for r in rows], "-o", ms=2.5, color=COLORS["ours"], label="SpecHint bytes")
    ax1.plot(ks, [r["ours_e"] for r in rows], "-s", ms=2.5, color=COLORS["ours_e"], label="SpecHint-E bytes")
    ax1.set_xlabel(r"challenge weight $\kappa$")
    ax1.set_ylabel("total bytes")
    ax2 = ax1.twinx()
    ax2.plot(ks, [r["bc2"] for r in rows], "--", color="#7f7f7f", lw=0.9, label=r"threshold $\beta_c^2$")
    ax2.set_ylabel(r"$\beta_c^2$")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6, loc="upper left")
    save(fig, "fig_filter")


def fig_pointwise(ex):
    rows = ex["pointwise"]["rows"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    a = [r["a"] for r in rows]
    ax.plot(a, [r["exact"] for r in rows], "-o", ms=2.5, color=COLORS["ours"], label=r"exact $\max|\ln(p/d)|$")
    ab = [(r["a"], r["bound"]) for r in rows if r["bound"] is not None]
    ax.plot([x for x, _ in ab], [y for _, y in ab], "--s", ms=2.5, color="#7f7f7f", label=r"bound $\ln\frac{1+\varepsilon}{1-\varepsilon}$")
    ax.plot(a, [r["sd"] for r in rows], ":^", ms=2.5, color="#1f77b4", label="statistical distance")
    ax.set_yscale("log")
    ax.set_xlabel(r"spherical part $a$ of the split")
    ax.set_ylabel("deviation")
    ax.legend(frameon=False, fontsize=6, loc="upper center")
    save(fig, "fig_pointwise")


def extra_figures():
    ex = json.load(open(os.path.join(RES, "extra.json")))
    if "leak" in ex:
        fig_leak(ex)
    if "raccooncurves" in ex:
        fig_racc(ex)
    if "certified" in ex and "certified256" in ex:
        fig_certified(ex)
    if "filtering" in ex:
        fig_filter(ex)
    if "pointwise" in ex:
        fig_pointwise(ex)
    if os.path.exists(os.path.join(RES, "latency.json")):
        fig_latency(json.load(open(os.path.join(RES, "latency.json"))))
    if os.path.exists(os.path.join(RES, "lattice_attack.json")):
        fig_lattice(json.load(open(os.path.join(RES, "lattice_attack.json"))))


def main():
    fig_width()
    if os.path.exists(os.path.join(RES, "extra.json")):
        ex = json.load(open(os.path.join(RES, "extra.json")))
        fig_sandwich(ex)
        fig_multi(ex)
        fig_map(ex)
        extra_figures()
    params = json.load(open(os.path.join(RES, "params.json")))
    th = json.load(open(os.path.join(RES, "theory.json")))
    fig_sizes(params)
    fig_ablation(params)
    fig_factor(th)
    fig_spectrum(th)
    fig_logdet(th)
    fig_su()
    fig_rates(th)
    fig_split(th)
    fig_many(th)
    if os.path.exists(os.path.join(RES, "bench.json")):
        fig_time(json.load(open(os.path.join(RES, "bench.json"))))
    if os.path.exists(os.path.join(RES, "lambda_scan.json")):
        fig_lambda(json.load(open(os.path.join(RES, "lambda_scan.json"))))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extra":
        extra_figures()
    else:
        main()
