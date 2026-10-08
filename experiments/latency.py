import json
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.pok import Scheme
from tighthint.params import K

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
RUNS = {"ours": 400, "enp": 400, "klss": 400, "onebit": 400, "gauss": 400, "uniform": 200, "bimodal": 150, "gpg": 150}


def run(cfg, runs, seed=99):
    rng = np.random.default_rng(seed)
    S = Scheme(cfg)
    msg = S.R.uniform(rng, (K,))
    prove, reps, verify = [], [], []
    for _ in range(runs):
        c, aux = S.commit(msg, rng)
        t0 = time.perf_counter()
        proof, a = S.prove(c, aux, rng)
        t1 = time.perf_counter()
        S.verify(c, proof)
        t2 = time.perf_counter()
        prove.append(1000 * (t1 - t0))
        verify.append(1000 * (t2 - t1))
        reps.append(a)
    return {"prove": prove, "verify": verify, "reps": reps}


def main():
    params = json.load(open(os.path.join(OUT, "params.json")))
    res = {}
    for m, r in RUNS.items():
        res[m] = run(params[m], r)
        p = np.array(res[m]["prove"])
        print(m, np.percentile(p, [50, 90, 99, 100]).round(2), np.bincount(res[m]["reps"])[:12], flush=True)
    with open(os.path.join(OUT, "latency.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
