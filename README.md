# Tight Hint

This repository contains our implementation package for the research
*On the Precise Hardness of Hint Module Learning With Errors*, which contains
* **SpecHint**, an abort-free proof of opening for BDLOP-style lattice commitments with
  spectral challenges, tight Hint-MLWE hiding and optimal masking;
* complete implementations of the eight existing masking methods it is compared with
  (uniform rejection, Gaussian rejection, one-bit rejection, bimodal rejection, G+G convolution,
  statistical flooding, Hint-MLWE masking and MatrixHint-MLWE masking);
* the parameter search, the compact lattice estimator, and the numerical experiments behind results of our research  (tight Hint-MLWE reduction, spectral challenge sets, smooth Rényi divergence of sums of uniforms, uniform hint noise).

## Requirements
Python 3.10 or newer (tested with 3.12), NumPy, SciPy, mpmath, Matplotlib, `epstopdf` (TeX Live) or Ghostscript `ps2pdf` for converting the EPS, pytest for the tests, fpylll for the lattice reduction experiment.
```bash
pip install numpy scipy mpmath matplotlib pytest
```

## Files and Contents
| Path | Content |
|------|---------|
| `tighthint/ring.py` | arithmetic in `Z_q[X]/(X^n+1)` by Kronecker substitution, including a big-modulus variant |
| `tighthint/sampling.py` | exact discrete Gaussian samplers (fixed and arbitrary centers), uniform, ternary and sum-of-uniform samplers |
| `tighthint/challenges.py` | sparse ternary challenges, spectral norms, the spectral threshold `x_n` and hash-to-challenge |
| `tighthint/estimator.py` | core-SVP estimates (primal uSVP, dual, MSIS in the l2 norm) and their closed forms used by the search |
| `tighthint/params.py` | width constraints of all methods and the exhaustive parameter search |
| `tighthint/pok.py` | commitment, prover and verifier of all ten proofs of opening, and a benchmark helper |
| `tighthint/coding.py` | Golomb–Rice and fixed length encoders for responses |
| `tighthint/hint.py` | posterior of Hint-MLWE, the tight split `theta(P)`, the reduction sampler, and exact split distances |
| `tighthint/su.py` | sums of uniforms, discrete Fisher information, the smooth Rényi bound and exact restricted divergences |
| `experiments/run_params.py` | parameter search at 128 bits and at 96, 160, 192 and 256 bits |
| `experiments/theory.py` | numerical experiments for Sections 3 and 4 of our work |
| `experiments/bench.py` | time, size, memory and time profile benchmarks of all methods |
| `experiments/latency.py` | distribution of proving and verification times over repeated proofs |
| `experiments/lattice_attack.py` | primal attack with fpylll BKZ on small elliptical and spherical instances with ring hints |
| `experiments/run_all.py` | runs all scripts above in order |
| `results/` | JSON outputs, figures and tables produced by the scripts |
| `tests/` | unit tests |

## Running the Experiments
```bash
python3 experiments/run_all.py
```
The running steps can also be run one by one:
```bash
python3 experiments/run_params.py
python3 experiments/theory.py
python3 experiments/bench.py
python3 experiments/latency.py
python3 experiments/extra.py
python3 experiments/lattice_attack.py
python3 experiments/figures.py
python3 experiments/tables.py
python3 experiments/tables.py appendix
```
A single appendix job of our research is run with `python3 experiments/extra.py <job>`, for example
`python3 experiments/extra.py raccoon`, and a single lattice configuration with
`python3 experiments/lattice_attack.py <index>`. The lattice experiment needs `fpylll`
and takes about one to three hours; the appendix jobs take between a few seconds and half an hour each. On an Apple M2, the parameter search takes about 15 minutes (most of it for the flooding baseline and
the security level scan), the theory experiments about 6 minutes, and the benchmarks about 5 minutes. All randomness in the experiments uses fixed seeds.

## Using the SpecHint
```python
import json
import numpy as np
from tighthint.pok import Scheme

cfg = json.load(open("results/params.json"))["ours"]
rng = np.random.default_rng()
scheme = Scheme(cfg)
message = scheme.R.uniform(rng, (1,))
commitment, opening = scheme.commit(message, rng)
proof, _ = scheme.prove(commitment, opening, rng)
assert scheme.verify(commitment, proof)
```
The configuration keys `ours`, `ours_e`, `klss`, `enp`, `enp_spec`, `onebit`, `gauss`, `uniform`,
`bimodal`, `gpg` and `flood` select SpecHint, its elliptical parameter set, and the baselines.
Our implementation is built for measurement and comparison as all methods share the same ring
arithmetic, samplers, hash function and encoder, so relative timings are normal, and it is not
hardened against side channels.

## Tests
```bash
python3 -m pytest -q tests
```
The tests check ring multiplication against schoolbook convolution for moduli from 20 to 213 bits,
the samplers against the exact discrete Gaussian, the encoders, the posterior lemma, the tight split
against exact statistical distances, the spectral challenge lemma, the smooth Rényi bound against exact
restricted divergences, the estimator against the Kyber parameter sets, and completeness and
soundness checks of every proof of opening.
