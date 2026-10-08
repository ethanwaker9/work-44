import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint.pok import Scheme
from tighthint.params import K, METHODS

RES = os.path.join(os.path.dirname(__file__), "..", "results", "params.json")


def check(method, trials):
    cfg = json.load(open(RES))[method]
    rng = np.random.default_rng(15)
    S = Scheme(cfg)
    msg = S.R.uniform(rng, (K,))
    for _ in range(trials):
        c, aux = S.commit(msg, rng)
        proof, _ = S.prove(c, aux, rng)
        assert S.verify(c, proof)
        bad0 = S.R.add(c[0], np.eye(1, S.n, 0, dtype=np.int64)[0] if not S.big else np.array([1] + [0] * (S.n - 1), dtype=object))
        assert not S.verify((bad0, c[1]), proof)


def test_all_methods():
    for m in METHODS + ["enp_spec"]:
        check(m, 1 if m == "flood" else 3)
