import json
import math
import os
import sys
import time
import numpy as np
from fpylll import IntegerMatrix, BKZ, LLL, GSO
from fpylll.algorithms.bkz2 import BKZReduction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.challenges import random_challenge, embed_sq
from tighthint.sampling import CenteredSampler

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
STRATEGY = next((p for p in ("/opt/homebrew/share/fplll/strategies/default.json",
                             "/usr/local/share/fplll/strategies/default.json",
                             "/usr/share/fplll/strategies/default.json") if os.path.exists(p)), None)


def rot(a):
    n = len(a)
    R = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            k = i - j
            R[i, j] = a[k] if k >= 0 else -a[k + n]
    return R


def sample_elliptic(rng, S, blocks):
    n = S.shape[0]
    r = 1.2
    w, V = np.linalg.eigh(S - r * r * np.eye(n))
    L = V * np.sqrt(np.clip(w, 0, None))
    cs = CenteredSampler(r)
    out = []
    for _ in range(blocks):
        c = L @ rng.normal(0.0, 1.0 / math.sqrt(2 * math.pi), n)
        out.append(cs.sample(rng, c))
    return np.concatenate(out)


def ring_matrix(rng, n, rows, cols, q):
    A = np.zeros((rows * n, cols * n), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            a = rng.integers(0, q, n)
            A[i * n:(i + 1) * n, j * n:(j + 1) * n] = rot(a).astype(np.int64) % q
    return A


def instance(rng, n, d, m, q, kind, S_blk, width2):
    A = ring_matrix(rng, n, m, d, q)
    N = n * (m + d)
    if kind == "elliptic":
        x = sample_elliptic(rng, S_blk, m + d)
    else:
        x = sample_elliptic(rng, width2 * np.eye(n), m + d)
    e, s = x[: m * n], x[m * n:]
    b = (e + A @ s) % q
    return A, b, x


def embedding(A, b, q, W, F, M):
    mn, dn = A.shape
    N = mn + dn
    B = np.zeros((N + 1, N + 1), dtype=np.float64)
    B[:mn, :mn] = q * np.eye(mn)
    B[mn:N, :mn] = (-A.T) % q
    B[mn:N, mn:N] = np.eye(dn)
    B[N, :mn] = b
    B[N, N] = M
    if W is not None:
        B[:, :N] = B[:, :N] @ W.T
    return np.rint(F * B).astype(np.int64)


def base_embedding(A, b, q, M):
    mn, dn = A.shape
    N = mn + dn
    B = np.zeros((N + 1, N + 1), dtype=object)
    for i in range(mn):
        B[i, i] = q
    At = (-A.T) % q
    for j in range(dn):
        for i in range(mn):
            B[mn + j, i] = int(At[j, i])
        B[mn + j, mn + j] = 1
    for i in range(mn):
        B[N, i] = int(b[i])
    B[N, N] = int(M)
    return B


def found(U, Borig, target):
    rows = np.array([[U[i, j] for j in range(U.ncols)] for i in range(U.nrows)], dtype=object)
    V = rows.dot(Borig)
    t = np.array([int(v) for v in target], dtype=object)
    for r in V:
        if all(r[k] == t[k] for k in range(len(t))) or all(r[k] == -t[k] for k in range(len(t))):
            return True
    return False


def attack(Bint, Borig, target, bmax, step=2, tours=2):
    A = IntegerMatrix.from_matrix(Bint.tolist())
    U = IntegerMatrix.identity(A.nrows)
    M = GSO.Mat(A, U=U)
    M.update_gso()
    L = LLL.Reduction(M)
    L()
    if found(U, Borig, target):
        return 2
    bkz = BKZReduction(M)
    for b in range(4, bmax + 1, step):
        for _ in range(tours):
            bkz(BKZ.Param(block_size=b, strategies=STRATEGY, max_loops=1, flags=BKZ.MAX_LOOPS | BKZ.GH_BND) if STRATEGY else BKZ.Param(block_size=b, max_loops=1, flags=BKZ.MAX_LOOPS))
            if found(U, Borig, target):
                return b
    return None


KINDS = (("elliptic", True), ("det", False), ("min", False), ("prior", False), ("elliptic_naive", False))

CONFIGS = [dict(n=16, d=3, m=3, q=521, s1=10.0, s2=20.0, kappa=6, trials=24),
           dict(n=16, d=3, m=3, q=1031, s1=30.0, s2=15.0, kappa=6, trials=24),
           dict(n=16, d=4, m=4, q=1031, s1=20.0, s2=20.0, kappa=6, trials=12,
                kinds=("elliptic", "det", "min", "elliptic_naive")),
           dict(n=16, d=3, m=3, q=521, s1=10.0, s2=20.0 * math.sqrt(8), kappa=6, trials=24, hints=8,
                kinds=("elliptic", "det", "min", "elliptic_naive"))]


def run(n=16, d=3, m=3, q=521, s1=10.0, s2=20.0, kappa=6, trials=12, bmax=60, seed=2026, kinds=None, hints=1):
    rng = np.random.default_rng(seed)
    gammas = [random_challenge(rng, n, kappa) for _ in range(hints)]
    G = sum(rot(g).T @ rot(g) for g in gammas)
    S = np.linalg.inv(np.eye(n) / s1 ** 2 + G / s2 ** 2)
    ev = np.linalg.eigvalsh(S)
    det_w2 = float(np.exp(np.mean(np.log(ev))))
    min_w2 = float(ev.min())
    Winv_blk = np.linalg.cholesky(S)
    Wblk = np.linalg.inv(Winv_blk) * math.sqrt(det_w2)
    blocks = m + d
    W = np.kron(np.eye(blocks), Wblk)
    res = {"params": {"n": n, "d": d, "m": m, "q": q, "s1": s1, "s2": s2, "kappa": kappa, "hints": hints,
                      "det_w2": det_w2, "min_w2": min_w2, "prior_w2": s1 * s1,
                      "slots": sorted(sum(embed_sq(g) for g in gammas).tolist())}, "runs": {}}
    F = 16.0
    widths = {"elliptic": None, "det": det_w2, "min": min_w2, "prior": s1 * s1, "elliptic_naive": None}
    for kind, useW in KINDS:
        if kinds is not None and kind not in kinds:
            continue
        width2 = widths[kind]
        out = []
        for t in range(trials):
            A, b, x = instance(rng, n, d, m, q, "elliptic" if kind.startswith("elliptic") else "spherical", S, width2)
            sd = math.sqrt((det_w2 if kind.startswith("elliptic") else width2) / (2 * math.pi))
            Mc = max(1.0, round(sd))
            Wm = W if useW else None
            Bint = embedding(A, b, q, Wm, F, Mc)
            Borig = base_embedding(A, b, q, Mc)
            tgt = [int(v) for v in x] + [int(Mc)]
            t0 = time.time()
            bsucc = attack(Bint, Borig, tgt, bmax=bmax)
            out.append({"b": bsucc, "sec": time.time() - t0})
            print(kind, t, bsucc, round(time.time() - t0, 1), flush=True)
        res["runs"][kind] = out
    return res


if __name__ == "__main__":
    path = os.path.join(OUT, "lattice_attack.json")
    which = [int(a) for a in sys.argv[1:]] or list(range(len(CONFIGS)))
    allres = json.load(open(path)) if os.path.exists(path) else []
    for i in which:
        res = run(**CONFIGS[i])
        while len(allres) <= i:
            allres.append(None)
        allres[i] = res
        with open(path, "w") as f:
            json.dump(allres, f, indent=1)
