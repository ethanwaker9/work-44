import json
import os
import platform
import sys
import time
import tracemalloc
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.pok import Scheme
from tighthint.params import K, METHODS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
TRIALS = {"flood": 5, "gpg": 60, "bimodal": 60, "uniform": 60}


def peak_kib(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


def run(cfg, trials, seed=2026):
    rng = np.random.default_rng(seed)
    S = Scheme(cfg)
    msg = S.R.uniform(rng, (K,))
    tc, tp, tv, reps, ok, sizes = [], [], [], [], 0, []
    for _ in range(trials):
        t0 = time.perf_counter()
        c, aux = S.commit(msg, rng)
        t1 = time.perf_counter()
        proof, a = S.prove(c, aux, rng)
        t2 = time.perf_counter()
        good = S.verify(c, proof)
        t3 = time.perf_counter()
        tc.append(t1 - t0)
        tp.append(t2 - t1)
        tv.append(t3 - t2)
        reps.append(a)
        ok += int(good)
        sizes.append(len(proof))
    c, aux = S.commit(msg, rng)
    proof, _ = S.prove(c, aux, rng)
    mem_p = peak_kib(lambda: S.prove(c, aux, rng))
    mem_v = peak_kib(lambda: S.verify(c, proof))
    ms = lambda v: 1000.0 * float(np.median(v))
    return {
        "commit_ms": ms(tc), "prove_ms": ms(tp), "verify_ms": ms(tv),
        "prove_ms_mean": 1000.0 * float(np.mean(tp)), "verify_ms_mean": 1000.0 * float(np.mean(tv)),
        "reps": float(np.mean(reps)), "accept": ok / trials, "trials": trials,
        "proof_bytes_mean": float(np.mean(sizes)), "proof_bytes_max": int(np.max(sizes)),
        "commit_bytes": cfg["commit_bytes"], "mem_prove_kib": mem_p, "mem_verify_kib": mem_v,
    }


def main():
    params = json.load(open(os.path.join(OUT, "params.json")))
    res = {"platform": {"python": platform.python_version(), "machine": platform.machine(),
                        "processor": platform.processor(), "system": platform.system(), "numpy": np.__version__}}
    for m in METHODS + ["enp_spec"]:
        cfg = params.get(m)
        if cfg is None:
            continue
        r = run(cfg, TRIALS.get(m, 200))
        res[m] = r
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with open(os.path.join(OUT, "bench.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
    run_profile()


def profile(cfg, trials, seed=7):
    rng = np.random.default_rng(seed)
    S = Scheme(cfg)
    msg = S.R.uniform(rng, (K,))
    acc = {"mask": 0.0, "commit_w": 0.0, "challenge": 0.0, "product": 0.0, "reject": 0.0, "encode": 0.0}
    for _ in range(trials):
        c, aux = S.commit(msg, rng)
        while True:
            t0 = time.perf_counter()
            y = S.gpg_mask(rng, aux) if S.method == "gpg" else S.mask(rng)
            t1 = time.perf_counter()
            w = S.w_of(y, aux)
            t2 = time.perf_counter()
            d = S.digest(c, w)
            g = S.challenge(d)
            t3 = time.perf_counter()
            v = S.ring_mul_small(g, aux["r"])
            z = y + v
            t4 = time.perf_counter()
            ok = S.accept(z, v, rng) and S.norm_ok(z)
            t5 = time.perf_counter()
            acc["mask"] += t1 - t0
            acc["commit_w"] += t2 - t1
            acc["challenge"] += t3 - t2
            acc["product"] += t4 - t3
            acc["reject"] += t5 - t4
            if ok:
                t6 = time.perf_counter()
                S.encode(z)
                acc["encode"] += time.perf_counter() - t6
                break
    return {k: 1000.0 * v / trials for k, v in acc.items()}


def run_profile(methods=("gauss", "onebit", "klss", "enp", "ours")):
    params = json.load(open(os.path.join(OUT, "params.json")))
    res = {m: profile(params[m], 100) for m in methods}
    with open(os.path.join(OUT, "profile.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res
