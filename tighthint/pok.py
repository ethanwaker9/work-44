import hashlib
import math
import time
import numpy as np
from .ring import make_ring, Ring
from .sampling import DiscreteGaussian, CenteredSampler, ternary, big_gauss
from .challenges import expand, sparse_from_stream, spectral_sq, hash_to_coset, embed_sq
from .coding import rice_encode, rice_decode, fixed_encode, fixed_decode, big_rice_encode, big_rice_decode
from . import params as PR
from .params import K, challenge_info, key_s1_bound, ternary_norm, tail_t

HINT = ("klss", "enp", "enp_spec", "ours", "ours_e")
REJ = ("gauss", "onebit")
LIFTED = ("bimodal", "gpg")


def twist(n):
    return np.exp(1j * np.pi * np.arange(n) / n)


def key_s1_sq(r):
    return float(np.max(np.sum(embed_sq(np.asarray(r, dtype=np.float64)), axis=0)))


class Scheme:
    def __init__(self, cfg, seed=b"tighthint-ck"):
        self.cfg = cfg
        self.method = cfg["method"]
        self.n = cfg["n"]
        self.q = cfg["q"]
        self.mu = cfg["mu"]
        self.nu = cfg["nu"]
        self.D = self.mu + self.nu + K
        self.seed = seed
        self.R = make_ring(self.n, self.q)
        self.big = not isinstance(self.R, Ring)
        ch = challenge_info(self.method, self.n)
        self.kappa = ch["kappa"]
        self.kind = ch["kind"]
        self.bc2 = ch["s1"] ** 2 if self.kind == "spectral" else None
        self.beta = cfg["beta"]
        self.t = tail_t()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "little"))
        mu, nu, D, n = self.mu, self.nu, self.D, self.n
        B0p = self.R.uniform(rng, (mu, nu + K))
        B1p = self.R.uniform(rng, (K, nu))
        dt = object if self.big else np.int64
        self.B0 = np.zeros((mu, D, n), dtype=dt)
        for i in range(mu):
            self.B0[i, i, 0] = 1
        self.B0[:, mu:, :] = B0p
        self.B1 = np.zeros((K, D, n), dtype=dt)
        self.B1[:, mu:mu + nu, :] = B1p
        for i in range(K):
            self.B1[i, mu + nu + i, 0] = 1
        self.B0pk = self.R.packmat(self.B0)
        self.B1pk = self.R.packmat(self.B1)
        m = self.method
        if m in HINT:
            self.g1 = DiscreteGaussian(cfg["sigma1"])
            self.g2 = DiscreteGaussian(cfg["sigma2"])
        elif m in REJ or m == "bimodal":
            self.g2 = DiscreteGaussian(cfg["s"])
            self.s_std = cfg["s"] / math.sqrt(2 * math.pi)
            self.T = cfg["T"]
            self.M = self._rej_m()
        elif m == "uniform":
            self.gamma1 = cfg["gamma1"]
            self.binf = cfg["binf"]
        elif m == "flood":
            self.s_flood = cfg["s"]
        if m in REJ or m in LIFTED or m in ("uniform", "flood"):
            self.Br = ternary_norm(D * n, self.t)
            self.BS = key_s1_bound(n, D + (1 if m in LIFTED else 0))
        if m == "gpg":
            self.sigma = cfg["s"]
            self.gs = cfg["gpg_s"]
            self.round_r = 1.2 * math.sqrt(math.log(2 * n * (1 + 2.0 ** (PR.LAM + 4))) / math.pi)
            self.cs = CenteredSampler(self.round_r)
            self.kg0 = DiscreteGaussian(self.gs, 0.0)
            self.kg1 = DiscreteGaussian(self.gs, -0.5)
        if m in LIFTED:
            self.R2 = Ring(n, 2 * self.q)
        self.rice_k = cfg.get("rice_k")

    def _rej_m(self):
        m = self.method
        if m == "gauss":
            g = 13.0
            return math.exp(14 / g + 1 / (2 * g * g))
        return math.exp(1 / (2 * 0.675 ** 2))

    def sample_r(self, rng):
        D, n = self.D, self.n
        if self.method in HINT:
            return self.g1.sample(rng, D * n).reshape(D, n)
        while True:
            r = ternary(rng, (D, n))
            if float(np.sum(r * r)) > self.Br ** 2:
                continue
            s1sq = key_s1_sq(r) + (1.0 if self.method in LIFTED else 0.0)
            if s1sq <= self.BS ** 2:
                return r

    def commit(self, msg, rng):
        r = self.sample_r(rng)
        rp = self.R.packvec(r)
        c0 = self.R.matvec_packed(self.B0pk, rp)
        c1 = self.R.add(self.R.matvec_packed(self.B1pk, rp), msg)
        return (c0, c1), self.prepare(r, (c0, c1))

    def prepare(self, r, c):
        aux = {"r": r}
        if self.method in LIFTED:
            n, q = self.n, self.q
            c0 = np.asarray(c[0], dtype=np.int64)
            A = np.zeros((self.mu, self.D + 1, n), dtype=np.int64)
            A[:, : self.D, :] = (2 * np.asarray(self.B0, dtype=np.int64)) % (2 * q)
            last = (-2 * c0) % (2 * q)
            last[:, 0] = (last[:, 0] + q) % (2 * q)
            A[:, self.D, :] = last
            aux["Apk"] = self.R2.packmat(A)
            rb = np.zeros((self.D + 1, n), dtype=np.int64)
            rb[: self.D] = r
            rb[self.D, 0] = 1
            aux["rbar"] = rb
            if self.method == "gpg":
                tw = twist(n)
                Rh = np.fft.fft(rb * tw, axis=1)
                rho = np.sum(np.abs(Rh) ** 2, axis=0)
                a = math.sqrt(self.sigma ** 2 - self.round_r ** 2)
                disc = a * a - 4 * self.gs ** 2 * rho
                if np.min(disc) <= 0:
                    raise ValueError("gpg covariance")
                aux["gpg"] = (Rh, (a - np.sqrt(disc)) / rho, a, tw)
        return aux

    def digest(self, c, w):
        h = hashlib.shake_256()
        h.update(self.seed)
        for part in (c[0], c[1], w):
            if self.big:
                h.update(b"".join(int(x).to_bytes((self.q.bit_length() + 8) // 8, "little") for x in np.ravel(part)))
            else:
                h.update(np.ascontiguousarray(part, dtype=np.int64).tobytes())
        return h.digest(32)

    def challenge(self, d):
        n = self.n
        if self.kind == "coset":
            return hash_to_coset(d, n)
        ctr = 0
        while True:
            stream = expand(d + ctr.to_bytes(4, "little"), 16 + 8 * n)
            g = sparse_from_stream(stream, n, self.kappa, signed=self.kind != "binary")
            if self.bc2 is None or spectral_sq(g) <= self.bc2:
                return g
            ctr += 1

    def mask(self, rng):
        m, D, n = self.method, self.D, self.n
        if m in HINT or m in REJ:
            return self.g2.sample(rng, D * n).reshape(D, n)
        if m == "uniform":
            return rng.integers(-self.gamma1, self.gamma1 + 1, size=(D, n), dtype=np.int64)
        if m == "flood":
            return big_gauss(rng, self.s_flood, D * n).reshape(D, n)
        if m == "bimodal":
            return self.g2.sample(rng, (D + 1) * n).reshape(D + 1, n)
        return None

    def gpg_mask(self, rng, aux):
        Rh, hmul, a, tw = aux["gpg"]
        n, D1 = self.n, self.D + 1
        g = rng.normal(0.0, 1.0 / math.sqrt(2 * math.pi), (D1, n))
        G = np.fft.fft(g * tw, axis=1)
        U = np.sum(np.conj(Rh) * G, axis=0)
        H = hmul * U
        corr = np.real(np.fft.ifft(Rh * H[None, :], axis=1) / tw)
        yc = a * g - corr
        return self.cs.sample(rng, yc)

    def ring_mul_small(self, g, r):
        R = self.R2 if self.method in LIFTED else self.R
        if self.big:
            return self.R.centered(self.R.scalar_vec(self.R.reduce(g), self.R.reduce(r)))
        return R.centered(R.scalar_vec(np.mod(g, R.q), np.mod(r, R.q)))

    def w_of(self, y, aux):
        if self.method in LIFTED:
            return self.R2.matvec_packed(aux["Apk"], self.R2.packvec(np.mod(y, self.R2.q)))
        if self.big:
            return self.R.matvec_packed(self.B0pk, self.R.packvec(self.R.reduce(y)))
        return self.R.matvec_packed(self.B0pk, self.R.packvec(np.mod(y, self.q)))

    def accept(self, z, v, rng):
        m = self.method
        if m in HINT or m in ("flood", "gpg"):
            return True
        if m == "uniform":
            return int(np.max(np.abs(z))) <= self.binf
        zz = z.astype(np.float64)
        vv = v.astype(np.float64)
        zv = float(np.sum(zz * vv))
        v2 = float(np.sum(vv * vv))
        s2 = self.s_std ** 2
        if m == "bimodal":
            x = zv / s2
            lc = abs(x) + math.log1p(math.exp(-2 * abs(x))) - math.log(2)
            return math.log(rng.random()) < -(math.log(self.M) - v2 / (2 * s2) + lc)
        if m == "onebit" and zv < 0:
            return False
        e = (v2 - 2 * zv) / (2 * s2)
        return math.log(rng.random()) < min(0.0, e - math.log(self.M))

    def norm_ok(self, z):
        if self.method == "uniform":
            return int(np.max(np.abs(z))) <= self.binf
        if self.big:
            return sum(int(x) * int(x) for x in np.ravel(z)) <= int(self.beta ** 2) + 1
        return float(np.sum(z.astype(np.float64) ** 2)) <= self.beta ** 2

    def encode(self, z):
        if self.method == "uniform":
            return fixed_encode(z, self.binf)[0]
        if self.big:
            return big_rice_encode(z, self.rice_k)[0]
        return rice_encode(z, self.rice_k)[0]

    def decode(self, data, count):
        if self.method == "uniform":
            return fixed_decode(data, count, self.binf)
        if self.big:
            return big_rice_decode(data, count, self.rice_k)
        return rice_decode(data, count, self.rice_k)

    def prove(self, c, aux, rng):
        attempts = 0
        lifted = self.method in LIFTED
        while True:
            attempts += 1
            y = self.gpg_mask(rng, aux) if self.method == "gpg" else self.mask(rng)
            w = self.w_of(y, aux)
            d = self.digest(c, w)
            g = self.challenge(d)
            if self.method == "gpg":
                par = g
                kk = np.where(par == 0, self.kg0.sample(rng, self.n), self.kg1.sample(rng, self.n))
                x = 2 * kk + par
                v = self.ring_mul_small(x, aux["rbar"])
                z = y + v
            elif lifted:
                v = self.ring_mul_small(g, aux["rbar"])
                sgn = 1 if rng.random() < 0.5 else -1
                z = y + sgn * v
            else:
                v = self.ring_mul_small(g, aux["r"])
                z = y + v
            if not self.accept(z, v, rng):
                continue
            if not self.norm_ok(z):
                continue
            return d + self.encode(z), attempts

    def verify(self, c, proof):
        d = proof[:32]
        count = (self.D + (1 if self.method in LIFTED else 0)) * self.n
        try:
            z = self.decode(proof[32:], count)
        except (ValueError, IndexError):
            return False
        z = z.reshape(-1, self.n)
        if not self.norm_ok(z):
            return False
        g = self.challenge(d)
        if self.method in LIFTED:
            q2 = 2 * self.q
            c0 = np.asarray(c[0], dtype=np.int64)
            A = np.zeros((self.mu, self.D + 1, self.n), dtype=np.int64)
            A[:, : self.D, :] = (2 * np.asarray(self.B0, dtype=np.int64)) % q2
            last = (-2 * c0) % q2
            last[:, 0] = (last[:, 0] + self.q) % q2
            A[:, self.D, :] = last
            Az = self.R2.matvec_packed(self.R2.packmat(A), self.R2.packvec(np.mod(z, q2)))
            w = np.mod(Az - self.q * g[None, :], q2)
        else:
            if self.big:
                Bz = self.R.matvec_packed(self.B0pk, self.R.packvec(self.R.reduce(z)))
                w = self.R.sub(Bz, self.R.scalar_vec(self.R.reduce(g), c[0]))
            else:
                Bz = self.R.matvec_packed(self.B0pk, self.R.packvec(np.mod(z, self.q)))
                w = self.R.sub(Bz, self.R.scalar_vec(np.mod(g, self.q), c[0]))
        return self.digest(c, w) == d


def benchmark(cfg, trials, seed=1):
    rng = np.random.default_rng(seed)
    S = Scheme(cfg)
    msg = S.R.uniform(rng, (K,))
    tc = tp = tv = 0.0
    reps = 0
    ok = 0
    sizes = []
    for _ in range(trials):
        t0 = time.perf_counter()
        c, aux = S.commit(msg, rng)
        t1 = time.perf_counter()
        proof, a = S.prove(c, aux, rng)
        t2 = time.perf_counter()
        good = S.verify(c, proof)
        t3 = time.perf_counter()
        tc += t1 - t0
        tp += t2 - t1
        tv += t3 - t2
        reps += a
        ok += int(good)
        sizes.append(len(proof))
    return {
        "commit_ms": 1000 * tc / trials,
        "prove_ms": 1000 * tp / trials,
        "verify_ms": 1000 * tv / trials,
        "reps": reps / trials,
        "accept": ok / trials,
        "proof_bytes_mean": float(np.mean(sizes)),
        "proof_bytes_max": int(np.max(sizes)),
        "commit_bytes": cfg["commit_bytes"],
    }
