import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tighthint import params as P

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main(scan=True):
    os.makedirs(OUT, exist_ok=True)
    P.set_lambda(128)
    res = P.search_all(os.path.join(OUT, "params.json"), P.METHODS + ["enp_spec"])
    for k, v in res.items():
        print(k, v["total_bytes"], v["proof_bytes"], v["commit_bytes"], v["n"], v["logq"], v["mu"], v["nu"])
    if not scan:
        return
    out = {}
    for lam in (96, 128, 160, 192, 256):
        P.set_lambda(lam)
        out[lam] = {}
        for m in ("uniform", "gauss", "onebit", "bimodal", "gpg", "klss", "enp", "ours", "ours_e"):
            b = P.search(m, ns=(128, 256, 512) if lam > 192 else (128, 256))
            out[lam][m] = b
            print(lam, m, None if b is None else (b["total_bytes"], b["n"], b["logq"], b["mu"], b["nu"]), flush=True)
    P.set_lambda(128)
    with open(os.path.join(OUT, "lambda_scan.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main(scan="--no-scan" not in sys.argv)
