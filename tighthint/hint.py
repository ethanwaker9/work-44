import math
import numpy as np
from .sampling import CenteredSampler


def smoothing_eta(eps, dim):
    return math.sqrt(math.log(2.0 * dim * (1.0 + 1.0 / eps)) / math.pi)


def posterior_width_sq(s1, s2, lam_max):
    return 1.0 / (1.0 / s1 ** 2 + lam_max / s2 ** 2)


def tight_sigma0_sq(post2, eta):
    disc = 1.0 - 4.0 * eta ** 2 / post2
    if disc < -1e-9:
        return float("nan")
    return 0.5 * post2 * (1.0 + math.sqrt(max(disc, 0.0)))


def klss_sigma0_sq(post2):
    return 0.5 * post2


def upper_width_sq(post2_min, eta):
    if post2_min <= eta ** 2:
        return float("nan")
    return post2_min + eta ** 2 * post2_min / (post2_min - eta ** 2)


def posterior(M, h, s1, s2):
    D = M.shape[1]
    prec = np.eye(D) / s1 ** 2 + M.T @ M / s2 ** 2
    Sigma = np.linalg.inv(prec)
    mu = Sigma @ (M.T @ h) / s2 ** 2
    return Sigma, mu


def brute_posterior(M, h, s1, s2, box):
    D = M.shape[1]
    grids = np.meshgrid(*[np.arange(-box, box + 1)] * D, indexing="ij")
    X = np.stack([g.ravel() for g in grids], axis=1)
    r = h[None, :] - X @ M.T
    logw = -math.pi * (np.sum(X * X, axis=1) / s1 ** 2 + np.sum(r * r, axis=1) / s2 ** 2)
    w = np.exp(logw - logw.max())
    return X, w / w.sum()


def gaussian_pmf_on(X, Sigma, mu):
    P = np.linalg.inv(Sigma)
    d = X - mu[None, :]
    logw = -math.pi * np.einsum("ij,jk,ik->i", d, P, d)
    w = np.exp(logw - logw.max())
    return w / w.sum()


def sample_dgauss_ellipse(rng, Sigma, center, r):
    D = Sigma.shape[0]
    C = Sigma - r ** 2 * np.eye(D)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    L = V * np.sqrt(w)
    p = center + L @ rng.normal(0.0, 1.0 / math.sqrt(2 * math.pi), D)
    return CenteredSampler(r).sample(rng, p)


def reduction_secret(rng, M, h, s1, s2, s0, x0, eta):
    Sigma, mu = posterior(M, h, s1, s2)
    return x0 + sample_dgauss_ellipse(rng, Sigma - s0 ** 2 * np.eye(Sigma.shape[0]), mu, eta)


def split_distance(P, s02, mu=0.3, dps=60):
    from mpmath import mp, mpf, exp, pi, fsum
    mp.dps = dps

    def dg(s2, c, lo, hi):
        xs = list(range(lo, hi + 1))
        w = [exp(-pi * (mpf(x) - c) ** 2 / s2) for x in xs]
        Z = fsum(w)
        return {x: wi / Z for x, wi in zip(xs, w)}

    R = int(12 * math.sqrt(P)) + 5
    a = dg(mpf(s02), mpf(0), -R, R)
    b = dg(mpf(P) - mpf(s02), mpf(mu), -R, R)
    conv = {}
    for x, px in a.items():
        for y, py in b.items():
            conv[x + y] = conv.get(x + y, 0) + px * py
    t = dg(mpf(P), mpf(mu), -2 * R, 2 * R)
    keys = set(conv) | set(t)
    return float(fsum(abs(conv.get(k, 0) - t.get(k, 0)) for k in keys) / 2)


def uniform_hint_attack(hints, u):
    hi = np.max(hints, axis=0) - u
    lo = np.min(hints, axis=0) + u
    return hi, hi == lo


def uniform_hint_count(u, dim, lam):
    return int(math.ceil((2 * u + 1) * (lam * math.log(2) + math.log(dim))))


def ring_slot_posterior(gamma_embed_sq, s1, s2):
    return 1.0 / (1.0 / s1 ** 2 + gamma_embed_sq / s2 ** 2)


def log_det_width_sq(gamma_embed_sq, s1, s2, eta):
    w = ring_slot_posterior(gamma_embed_sq, s1, s2) - 2 * eta ** 2
    return float(np.exp(np.mean(np.log(w))))


def jensen_width_sq(s1, s2, kappa, eta):
    return 1.0 / (1.0 / s1 ** 2 + kappa / s2 ** 2) - 2 * eta ** 2
